import cv2
import argparse
import pathlib
import os
import numpy as np


def scale_and_crop(img: np.ndarray, dest_height: int, dest_width: int) -> np.ndarray:
    assert img.ndim == 3
    height, width, channels = img.shape
    assert channels == 3
    assert dest_height > 0 and dest_width > 0
    assert height > 0 and width > 0

    src_h, src_w = img.shape[:2]

    # 计算缩放因子：取宽和高的较大比例，保证覆盖目标尺寸
    scale = max(float(dest_width) / src_w, float(dest_height) / src_h)

    # 缩放后的新尺寸（向上取整，确保不小于目标尺寸）
    new_w = int(np.ceil(src_w * scale))
    new_h = int(np.ceil(src_h * scale))

    # 选择插值方法：缩小用 INTER_AREA，放大用 INTER_CUBIC，等尺寸用 INTER_LINEAR
    if scale < 1.0:
        interpolation = cv2.INTER_AREA
    elif scale > 1.0:
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_LINEAR

    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    # 居中裁剪
    x = (new_w - dest_width) // 2
    y = (new_h - dest_height) // 2

    return resized[y:y + dest_height, x:x + dest_width]



parser = argparse.ArgumentParser()
parser.add_argument('--image-dir', type=str)
parser.add_argument('--out-dir', type=str)
parser.add_argument("--img-height", type=int, default=128)
parser.add_argument("--img-width", type=int, default=64)

args = parser.parse_args()

image_dir = pathlib.Path(args.image_dir)
out_dir = pathlib.Path(args.out_dir)

os.makedirs(out_dir, exist_ok=True)

dest_height = args.img_height
dest_width = args.img_width

supported_image_suffixes = {'.jpg', '.jpeg', '.png'}

for entry in image_dir.glob("*"):
    if entry.is_dir():
        continue
    if not supported_image_suffixes.__contains__(entry.suffix):
        print(f"Ignoring {entry}")
        continue

    img: np.ndarray | None = cv2.imread(str(entry), cv2.IMREAD_COLOR_RGB)
    if img is None:
        print(f"Failed to load {entry}")
        continue

    new_img = scale_and_crop(img, dest_height, dest_width)

    # seems have to write as BGR
    new_img=cv2.cvtColor(new_img,cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_dir/f"{entry.stem}.png", new_img)

    pass
    # print(f"Processing {entry}")
