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

    def forward(self, images: torch.Tensor, tau: float) -> dict[str, torch.Tensor]:
        assert tau > 0
        # [B,N,H,W]
        weight = -color_diff_RGB(self.palette, images)

        hard_idx = weight.argmax(dim=1)  # Should be [B,H,W]
        y_hard = torch.nn.functional.one_hot(hard_idx, num_classes=self.palette.shape[0]).permute(0, 3, 1,
                                                                                                  2)  # Should be [B,N,H,W]
        y_soft = torch.nn.functional.softmax(weight / tau, dim=1)
        # Forward: argmax; backward: softmax
        y = y_hard + (y_soft - y_soft.detach())

        out_img = torch.einsum("bnhw,nc->bchw", y, self.palette)
        return {
            'color_diff': -weight,
            'y_hard': y_hard,
            'y_soft': y_soft,
            'hard_index': hard_idx,
            'converted_image': out_img,
        }

    def to_(self, device: torch.device):
        self.palette = self.palette.to(device)


def test():
    colors = load_palette("./binary/colorset-Slope.png")

    pal = Palette(colors)

    img = torch.randn((2, 3, 63, 129))

    results = pal.forward(img, tau=0.1)
    pass
    # pass


if __name__ == "__main__":
    test()
