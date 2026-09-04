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

import numpy as np
import h5py
import cv2
import pathlib
import torch


class ImgDataSet(torch.utils.data.Dataset):
    def __init__(self, file: pathlib.Path, key: str):
        reader = h5py.File(file, 'r')
        ds: h5py.Dataset = reader[key]
        assert len(ds.shape) == 4
        assert ds.shape[3] == 3
        assert ds.dtype == np.uint8
        # (N,H,W,C) -> (N,C,H,W)
        self.data = torch.tensor(np.array(ds).transpose((0, 3, 1, 2)), dtype=torch.uint8)

        pass

    def __getitem__(self, index) -> torch.Tensor:
        part = self.data[index, :, :, :].type(torch.float32) / 255
        return part

    def __len__(self) -> int:
        return self.data.size(0)


def test():
    ds = ImgDataSet(pathlib.Path('./binary/dataset.h5'), key='129x63')

    first = ds[0]

    some = ds[4:10]
    pass


if __name__ == "__main__":
    test()
