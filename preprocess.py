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

import os
import re
import random
import argparse
import pathlib
import typing
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import cv2


def parse_shape(string: str) -> tuple[int, int]:
    pattern = re.compile(r"^([0-9]+)x([0-9]+)$")
    matches = pattern.match(string)
    height = int(matches[1])
    width = int(matches[2])
    return height, width
    # pass


def scale_and_crop(img: np.ndarray, dest_height: int, dest_width: int) -> np.ndarray:
    src_h, src_w = img.shape[:2]
    scale = max(float(dest_width) / src_w, float(dest_height) / src_h)
    new_w = int(np.ceil(src_w * scale))
    new_h = int(np.ceil(src_h * scale))

    if scale < 1.0:
        interpolation = cv2.INTER_AREA
    elif scale > 1.0:
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_LINEAR

    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    x = (new_w - dest_width) // 2
    y = (new_h - dest_height) // 2
    return resized[y:y + dest_height, x:x + dest_width]


def process_one(job):
    """子进程任务：读取一张图并处理成所有目标尺寸。
    job = (图片路径, [(h,w), ...])
    返回 (路径, {shape: 数组})，读取失败返回 (路径, None)。"""
    path_str, shapes = job
    img = cv2.imread(path_str, cv2.IMREAD_COLOR_RGB)
    if img is None:
        return path_str, None
    return path_str, {shape: scale_and_crop(img, shape[0], shape[1]) for shape in shapes}


def flush_batch(filenames_ds, img_datasets, buf, start):
    """把 buf 里的一批结果写入 h5。batch 大小对齐 chunk，压缩更高效。"""
    n = len(buf)
    filenames_ds[start:start + n] = [p for p, _ in buf]
    for shape, ds in img_datasets.items():
        batch = np.stack([imgs[shape] for _, imgs in buf])
        ds[start:start + n] = batch
    return start + n


def main():
    supported_image_suffixes = {'.jpg', '.jpeg', '.png'}

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dirs', type=str, required=True)
    parser.add_argument('--out-file', type=str, required=True)
    parser.add_argument("--shapes", type=str, default="128x128")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    all_files = []
    for image_dir in args.image_dirs.split(';'):
        for entry in pathlib.Path(image_dir).glob("*"):
            if entry.is_dir():
                continue
            if entry.suffix.lower() not in supported_image_suffixes:
                print(f"Ignoring {entry}")
                continue
            all_files.append(entry)
    random.shuffle(all_files)

    f = h5py.File(pathlib.Path(args.out_file), mode='w')
    ds_filenames = f.create_dataset("filename",
                                    shape=(len(all_files),),
                                    maxshape=(max(1024, len(all_files)),),
                                    chunks=(1024,),
                                    compression='gzip',
                                    compression_opts=8,
                                    dtype=h5py.string_dtype(),
                                    )

    ds_shapes: dict[tuple[int, int], typing.Any] = {}
    for shape_str in args.shapes.split(';'):
        shape = parse_shape(shape_str)
        dest_height, dest_width = shape
        ds = f.create_dataset(shape_str,
                              shape=(len(all_files), dest_height, dest_width, 3),
                              chunks=(128, dest_height, dest_width, 3),
                              maxshape=(max(128, len(all_files)), dest_height, dest_width, 3),
                              dtype='uint8',
                              compression='gzip',
                              compression_opts=3,
                              )
        ds_shapes[shape] = ds

    jobs = [(str(p), list(ds_shapes.keys())) for p in all_files]
    batch_size = 128  # 与 chunk 第一维一致
    buf = []
    dst_idx = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # pool.map 保持任务顺序，输出顺序 = 打乱后的顺序（确定性的）
        for path_str, imgs in pool.map(process_one, jobs, chunksize=16):
            if imgs is None:
                print(f"Failed to load {path_str}")
                continue
            buf.append((path_str, imgs))
            if len(buf) >= batch_size:
                dst_idx = flush_batch(ds_filenames, ds_shapes, buf, dst_idx)
                buf.clear()
                print(f"[{dst_idx}/{len(all_files)}] written")

    if buf:
        dst_idx = flush_batch(ds_filenames, ds_shapes, buf, dst_idx)

    ds_filenames.resize((dst_idx,))
    for shape, ds in ds_shapes.items():
        ds.resize((dst_idx, shape[0], shape[1], 3))
    f.close()


if __name__ == '__main__':
    main()
