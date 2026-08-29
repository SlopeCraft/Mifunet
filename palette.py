import torch
import cv2
import numpy as np
import warnings


# Load palette from png. Read, dedup and into array [N,3]
def load_palette(filename: str) -> np.ndarray:
    img: np.ndarray | None = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to load palette image {filename}")

    assert img.ndim == 3
    if img.shape[2] != 4:
        raise RuntimeError(f"Palette image {filename} must be RGBA, but found {img.shape[2]} channel(s)")

    if img.dtype != np.uint8:
        raise RuntimeError(f"Palette image {filename} must be RGBA, uint8")

    img = img.reshape([-1, 4])

    colors = []
    for i in range(img.shape[0]):
        color = img[i, :]
        if color[3] == 0:
            continue
        if color[3] != 255:
            warnings.warn(f"Palette image {filename} contains semi-transparent color {color.tolist()} (in RGBA)")

        colors.append(color[0:3])

    colors = np.array(colors)
    colors = np.unique(colors, axis=0, sorted=False)
    return colors


# palette: [N,3]. images: [B,3,H,W]. Returns: [B,N,H,W]
def color_diff_RGB(palette: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    assert palette.shape[1] == 3
    assert images.shape[1] == 3
    # palette=
    # channel diff: [B,N,3,H,W]
    channel_diff = palette.view(1, -1, 3, 1, 1) - images.view(-1, 1, 3, images.shape[2], images.shape[3])

    color_diff = torch.sum(torch.square(channel_diff), dim=2, keepdim=False)
    return color_diff


class Palette:
    def __init__(self, palette: np.ndarray):
        self.palette = torch.tensor(palette, dtype=torch.float32) / 255.
        self.palette.requires_grad_(False)

    def forward(self, images: torch.Tensor, tau: float,
                details: bool = False, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
        """images: [B,3,H,W]。dtype=torch.bfloat16 时整段计算减半内存，输出统一回 fp32。

        forward 只依赖 color_diff 的接口，不依赖具体公式：
          - STE 放在 [B,3,H,W] 分辨率做，不再有 y/y_hard/temp 这些 [B,N,H,W] 副本
        """
        assert tau > 0
        images = images.to(dtype)
        pal = self.palette.to(dtype)          # [N,3]
        N = pal.shape[0]

        weight = -color_diff_RGB(pal, images)  # [B,N,H,W]

        hard_idx = weight.argmax(dim=1)       # [B,H,W]
        y_soft = torch.softmax(weight / tau, dim=1)  # [B,N,H,W]

        # STE 在 [B,3,H,W] 上做（与色差函数无关）：
        # einsum(y_hard + (y_soft - y_soft.detach()), pal)
        #   == pal[hard_idx] + (einsum(y_soft,pal) - einsum(y_soft,pal).detach())
        out_hard = pal[hard_idx].permute(0, 3, 1, 2)          # [B,3,H,W]
        out_soft = torch.einsum('bnhw,nc->bchw', y_soft, pal) # [B,3,H,W]
        converted = out_hard + (out_soft - out_soft.detach())
        converted = converted.to(torch.float32)

        result = {
            'converted_image': converted,
            'y_soft': y_soft,
            'hard_index': hard_idx,
        }
        if details:  # 调试/可视化才生成大张量
            result['color_diff'] = (-weight).to(torch.float32)  # 正距离（调试用）
            result['y_hard'] = torch.nn.functional.one_hot(hard_idx, N).permute(0, 3, 1, 2).to(torch.float32)
        return result

    def to_(self, device: torch.device):
        self.palette = self.palette.to(device)
        self.palette.requires_grad_(False)


def test():
    colors = load_palette("./binary/colorset-Slope.png")

    pal = Palette(colors)

    img = torch.randn((2, 3, 63, 129))

    results = pal.forward(img, tau=0.1)
    pass
    # pass


if __name__ == "__main__":
    test()
