# This file is part of Mifunet.
#
# Mifunet is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# Mifunet is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Mifunet.
# If not, see <https://www.gnu.org/licenses/>.

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
import argparse

import palette
from dataset import ImgDataSet
from sc_filter import SCFilter


def tensors2images(t: torch.Tensor) -> np.ndarray:
    im = t.detach().cpu().permute((0, 2, 3, 1)).numpy()
    im *= 255
    im = np.maximum(np.minimum(im, 255), 0).astype(np.uint8)
    return im


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset-key", type=str, required=True)
    parser.add_argument("--num", type=int, default=4)
    parser.add_argument("--export-onnx", action='store_true')
    args = parser.parse_args()

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"Using device {torch.cuda.get_device_name(device)}")

    fig_dir = Path("./binary/figures")
    os.makedirs(fig_dir, exist_ok=True)

    full_dataset = ImgDataSet(Path(args.dataset), key=args.dataset_key)

    ckpt = torch.load(
        args.checkpoint,
        weights_only=False)

    palette_dataset = palette.PaletteDataSet.from_files(["./binary/colorset-Slope.png"])
    # palette: Palette = ckpt['palette']
    sc_filter: SCFilter = ckpt['scFilter'].to(device)
    sc_filter.eval()

    N_imgs: int = args.num
    assert N_imgs > 0

    pal, pal_len = palette_dataset[torch.zeros([N_imgs], dtype=torch.int32)]
    pal = pal.to(device)
    pal_len = pal_len.to(device)
    pal_mask = palette.make_palette_mask(pal_len, pal.size(1))


    input_imgs = full_dataset[1:N_imgs + 1].to(device)
    output_imgs = sc_filter(input_imgs, pal, pal_mask)
    sc_converted_imgs = palette.convert_images(pal, pal_mask, output_imgs, tau=1e-2)['converted_image']
    naive_converted_imgs = palette.convert_images(pal, pal_mask, input_imgs, tau=1e-2)['converted_image']

    if args.export_onnx:
        from torch.export import Dim
        dim_batch_size = Dim("batch_size")
        dim_rows = Dim("image_rows")
        dim_cols = Dim("image_cols")
        dim_palette_length = Dim("palette_length")
        torch.onnx.export(
            sc_filter,
            (input_imgs, pal, pal_mask),
            fig_dir / f"{Path(args.checkpoint).stem}.onnx",
            input_names=["in image", "palette", "palette mask"],
            output_names=["out image"],
            dynamic_shapes={
                "images": {0: dim_batch_size, 2: dim_rows, 3: dim_cols},
                "palettes": {0: dim_batch_size, 1: dim_palette_length},
                "palette_mask": {0: dim_batch_size, 1: dim_palette_length},
            },
            dynamo=True,
        )

    input_imgs = tensors2images(input_imgs)
    output_imgs = tensors2images(output_imgs)
    sc_converted_imgs = tensors2images(sc_converted_imgs)
    naive_converted_imgs = tensors2images(naive_converted_imgs)

    rows = ["Input", "UNet Filtered", "Filter-Converted", "Naive-converted"]
    batches = [input_imgs, output_imgs, sc_converted_imgs, naive_converted_imgs]

    fig, axes = plt.subplots(len(batches), N_imgs,
                             figsize=(4 * N_imgs, 12))
    # 只有 1 张图时 axes 会是一维数组，这里统一成二维方便索引
    if N_imgs == 1:
        axes = axes[:, np.newaxis]

    for r, (row_name, batch) in enumerate(zip(rows, batches)):
        for c in range(N_imgs):
            ax = axes[r, c]
            ax.imshow(batch[c])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"img {c}", fontsize=14)
            if c == 0:
                ax.set_ylabel(row_name, fontsize=14)

    fig.suptitle("UNet Filter Result", fontsize=16)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{Path(args.checkpoint).stem}.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    run()
