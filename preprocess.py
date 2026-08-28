import typing
import random

import cv2
import argparse
import pathlib
import os
import re
import h5py
import numpy as np


def parse_shape(string: str) -> tuple[int, int]:
    pattern = re.compile(r"^([0-9]+)x([0-9]+)$")
    matches = pattern.match(string)
    height = int(matches[1])
    width = int(matches[2])
    return height, width
    # pass


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


def main():
    supported_image_suffixes = {'.jpg', '.jpeg', '.png'}

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str)
    parser.add_argument('--out-file', type=str)
    parser.add_argument("--shape", type=str, default="128x128")

    args = parser.parse_args()

    image_dir = pathlib.Path(args.image_dir)
    out_file = pathlib.Path(args.out_file)

    all_files = []
    for entry in image_dir.glob("*"):
        if entry.is_dir():
            continue
        if not supported_image_suffixes.__contains__(entry.suffix):
            print(f"Ignoring {entry}")
            continue
        all_files.append(entry)
    random.shuffle(all_files)

    f = h5py.File(out_file, mode='w')

    ds_filenames = f.create_dataset("filename",
                                    shape=(len(all_files),),
                                    maxshape=(max(1024, len(all_files)),),
                                    chunks=(1024,),
                                    compression='gzip',
                                    compression_opts=8,
                                    dtype=h5py.string_dtype(),
                                    )

    ds_shapes: dict[tuple[int, int], typing.Any] = {}
    for shape_str in args.shape.split(':'):
        shape = parse_shape(shape_str)
        dest_height, dest_width = shape
        ds = f.create_dataset(shape_str,
                              shape=(len(all_files), dest_height, dest_width, 3),
                              chunks=(32, dest_height, dest_width, 3),
                              maxshape=(max(32, len(all_files)), dest_height, dest_width, 3),
                              dtype='uint8',
                              compression='gzip',
                              compression_opts=3,
                              )
        ds_shapes[shape] = ds

    dst_idx: int = 0
    for idx, entry in enumerate(all_files):
        if idx % 100 == 0:
            print(f"[{idx}/{len(all_files)}] Processing images")

        img: np.ndarray | None = cv2.imread(str(entry), cv2.IMREAD_COLOR_RGB)
        if img is None:
            print(f"Failed to load {entry}")
            continue
        ds_filenames[dst_idx] = str(entry)
        for (shape, ds) in ds_shapes.items():
            new_img = scale_and_crop(img, shape[0], shape[1])
            ds[dst_idx, :, :, :] = new_img

        dst_idx += 1

        # seems have to write as BGR
        # new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(out_dir / f"{entry.stem}.png", new_img)

    ds_filenames.resize((dst_idx,))
    for (shape, ds) in ds_shapes.items():
        ds.resize((dst_idx, shape[0], shape[1], 3))

    f.close()


main()
