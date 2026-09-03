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


class PaletteDataSet(torch.utils.data.Dataset):
    def __init__(self):
        self.palette_arrays: list[torch.Tensor] = []

    def add_palette(self, pal: torch.Tensor):
        assert pal.ndim == 2
        assert pal.shape[0] > 0
        assert pal.shape[1] == 3
        p = pal.cpu().detach().type(torch.float32)
        p.requires_grad_(False)
        self.palette_arrays.append(p)

    def add_palette_numpy(self, pal: np.ndarray, data_range: float = 255.):
        self.add_palette(torch.tensor(pal, dtype=torch.float32) / data_range)

    def add_palette_from_file(self, filename: str):
        pal = load_palette(filename)
        self.add_palette_numpy(pal, data_range=255.)

    def __len__(self) -> int:
        return len(self.palette_arrays)

    def max_palette_len(self) -> int:
        assert len(self.palette_arrays) > 0
        ret = 0
        for pal in self.palette_arrays:
            ret = max(ret, pal.shape[0])
        return ret

    def merged_full_palette(self) -> tuple[torch.Tensor, torch.Tensor]:
        ret = torch.zeros([self.__len__(), self.max_palette_len(), 3], dtype=torch.float32)
        lens = []
        for i, pal in enumerate(self.palette_arrays):
            ret[i, 0:pal.shape[0], :] = pal
            lens.append(pal.shape[0])
        return ret, torch.tensor(lens, dtype=torch.int32)

    def __getitem__(self, index: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pals, lens = self.merged_full_palette()
        return pals[index, :, :], lens[index]


# palette: [B,N,3]. images: [B,3,H,W]. Returns: [B,N,H,W].
# Should always use RGB as input
def color_diff_RGB(palette: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    batch_size = images.shape[0]
    assert palette.shape[0] == batch_size
    assert palette.shape[2] == 3
    assert images.shape[1] == 3
    # palette=
    # channel diff: [B,N,3,H,W]
    channel_diff = palette.view(batch_size, -1, 3, 1, 1) - images.view(batch_size, 1, 3, images.shape[2],
                                                                       images.shape[3])

    color_diff = torch.sum(torch.square(channel_diff), dim=2, keepdim=False)
    return color_diff


RGB_to_LMS = torch.tensor(
    [[0.4122214708, 0.5363325363, 0.0514459929],
     [0.2119034982, 0.6806995451, 0.1073969566],
     [0.0883024619, 0.2817188376, 0.6299787005]])

cbrt_LMS_to_OKLab = torch.tensor([[0.2104542553, 0.7936177850, -0.0040720468],
                                  [1.9779984951, -2.4285922050, 0.4505937099],
                                  [0.0259040371, 0.7827717662, -0.8086757660]])


def rgb_to_OKLab_palette(palette_RGB: torch.Tensor) -> torch.Tensor:
    M1 = RGB_to_LMS.detach().clone().to(dtype=palette_RGB.dtype, device=palette_RGB.device)
    M2 = cbrt_LMS_to_OKLab.detach().clone().to(dtype=palette_RGB.dtype, device=palette_RGB.device)
    # Here, c refers rgb channel; m refers LMS; a refers OKLAB. b: batch, h: height, w: width, n: palette size
    pal_LMS = torch.einsum("mc,bnc->bnm", M1, palette_RGB)
    # assert torch.all(pal_LMS >= 0)
    pal_LMS = torch.nn.functional.relu(pal_LMS - 1e-6) + 1e-6
    pal_LMS = torch.pow(pal_LMS, 1. / 3.)
    pal_Lab = torch.einsum("am,bnm->bna", M2, pal_LMS)

    return pal_Lab


def rgb_to_OKLab_image(images_RGB: torch.Tensor) -> torch.Tensor:
    M1 = RGB_to_LMS.detach().clone().to(dtype=images_RGB.dtype, device=images_RGB.device)
    M2 = cbrt_LMS_to_OKLab.detach().clone().to(dtype=images_RGB.dtype, device=images_RGB.device)

    img_LMS = torch.einsum("mc,bchw->bmhw", M1, images_RGB)
    img_LMS = torch.nn.functional.relu(img_LMS - 1e-6) + 1e-6
    # assert torch.all(img_LMS >= 1e-6)
    img_LMS = torch.pow(img_LMS, 1. / 3.)
    img_Lab = torch.einsum("am,bmhw->bahw", M2, img_LMS)

    return img_Lab


# palette: [B,N,3]. images: [B,3,H,W]. Returns: [B,N,H,W].
# Should always use RGB as input
def color_diff_OKLAB(palette_RGB: torch.Tensor, images_RGB: torch.Tensor) -> torch.Tensor:
    batch_size = images_RGB.shape[0]
    assert palette_RGB.shape[0] == batch_size
    assert palette_RGB.shape[2] == 3
    assert images_RGB.shape[1] == 3

    pal_Lab = rgb_to_OKLab_palette(palette_RGB)
    img_Lab = rgb_to_OKLab_image(images_RGB)

    channel_diff = pal_Lab.view(batch_size, -1, 3, 1, 1) - img_Lab.view(batch_size, 1, 3, img_Lab.shape[2],
                                                                        img_Lab.shape[3])

    color_diff = torch.sum(torch.square(channel_diff), dim=2, keepdim=False)
    return color_diff


# Returns: [B,N]
def make_palette_mask(palette_lens: torch.Tensor, palette_len_max: int) -> torch.Tensor:
    assert (palette_lens <= palette_len_max).all()
    mask = torch.arange(palette_len_max,
                        device=palette_lens.device,
                        dtype=palette_lens.dtype).reshape(1, -1) < palette_lens.view(-1, 1)
    return mask


def convert_images(palettes_RGB: torch.Tensor, palette_mask: torch.Tensor, images_RGB: torch.Tensor, tau: float,
                   details: bool = False, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    batch_size = images_RGB.shape[0]
    assert palettes_RGB.shape[0] == batch_size
    assert palettes_RGB.shape[2] == 3
    assert images_RGB.shape[1] == 3
    """images: [B,3,H,W]。dtype=torch.bfloat16 时整段计算减半内存，输出统一回 fp32。

    forward 只依赖 color_diff 的接口，不依赖具体公式：
      - STE 放在 [B,3,H,W] 分辨率做，不再有 y/y_hard/temp 这些 [B,N,H,W] 副本
    """
    assert tau > 0
    images = images_RGB.to(dtype)
    pal = palettes_RGB.to(dtype)  # [N,3]
    N = pal.shape[1]
    b_idx = torch.arange(batch_size, device=pal.device).view(batch_size, 1, 1)  # 广播成 [B,H,W]

    diff = color_diff_OKLAB(pal, images)  # [B,N,H,W]

    hard_idx = torch.masked.argmin(diff, dim=1, keepdim=False, mask=palette_mask, dtype=torch.int32)  # [B,H,W]
    color_diff_selected = torch.masked.amin(diff, dim=1, keepdim=False, mask=palette_mask)  # [B,H,W]
    y_soft = torch.masked.softmax(-diff / tau, dim=1, mask=palette_mask)  # [B,N,H,W]

    # STE 在 [B,3,H,W] 上做（与色差函数无关）：
    out_hard = pal[b_idx, hard_idx].permute(0, 3, 1, 2)  # [B, H, W, 3]

    out_soft = torch.einsum('bnhw,bnc->bchw', y_soft, pal)  # [B,3,H,W]
    converted = out_hard + (out_soft - out_soft.detach())
    converted = converted.to(torch.float32)

    result = {
        'converted_image': converted,
        'y_soft': y_soft,
        'hard_index': hard_idx,
        'color_diff_selected': color_diff_selected.to(torch.float32)
    }
    if details:  # 调试/可视化才生成大张量
        result['color_diff'] = diff.to(torch.float32)  # 正距离（调试用）
        result['y_hard'] = torch.nn.functional.one_hot(hard_idx, N).permute(0, 3, 1, 2).to(torch.float32)
    return result


def test():
    pal = PaletteDataSet()
    pal.add_palette_from_file("./binary/colorset-Slope.png")
    pal.add_palette_from_file("./binary/colorset-Slope-old.png")
    pal.add_palette_from_file("./binary/colorset-Flat.png")
    pal.add_palette_from_file("./binary/colorset-Flat-old.png")
    pal.add_palette_from_file("./binary/colorset-FileOnly.png")
    pal.add_palette_from_file("./binary/colorset-FileOnly-old.png")

    # full = pal.merged_full_palette()
    # pass
    # pal = Palette(colors)

    img = torch.randn((2, 3, 63, 129))
    pals, lens = pal[torch.tensor([0, 3])]

    palette_mask = make_palette_mask(lens, pals.size(1))

    results = convert_images(pals, palette_mask.view(2, -1, 1, 1), img, tau=0.1)
    pass


if __name__ == "__main__":
    test()
