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
import torchinfo
from triton.language import dtype


class SCFilter(torch.nn.Module):
    def __init__(self, palette_embedding_size: int = 64):
        super(SCFilter, self).__init__()
        # Palette encoder
        self.palette_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=3, out_features=palette_embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=palette_embedding_size, out_features=palette_embedding_size),
        )
        # UNet
        self.encoder = torch.nn.ModuleList([
            torch.nn.Sequential(  # 3->64 channels
                torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential(  # 64->128 channels
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential(  # 128->256
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential(  # 256->512
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
            )
        ])
        # Encoder FiLM
        self.encoder_FiLMs = torch.nn.ModuleList([
            torch.nn.Linear(in_features=palette_embedding_size * 2, out_features=64 * 2),
            torch.nn.Linear(in_features=palette_embedding_size * 2, out_features=128 * 2),
            torch.nn.Linear(in_features=palette_embedding_size * 2, out_features=256 * 2),
            torch.nn.Linear(in_features=palette_embedding_size * 2, out_features=512 * 2),
        ])
        assert len(self.encoder) == len(self.encoder_FiLMs)

        self.middle = torch.nn.Sequential(  # 512->512->512
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.middle_FiLMs = torch.nn.Linear(in_features=palette_embedding_size * 2, out_features=512 * 2)

        self.decoder = torch.nn.ModuleList([
            torch.nn.Sequential(  # [512+512] -> 512
                torch.nn.Conv2d(in_channels=512 + 512, out_channels=512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ),
            torch.nn.Sequential(  # [512+256] -> 256
                torch.nn.Conv2d(in_channels=512 + 256, out_channels=256, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ),
            torch.nn.Sequential(  # [256+128] -> 128
                torch.nn.Conv2d(in_channels=256 + 128, out_channels=128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ),
            torch.nn.Sequential(  # [128+64] -> 3
                torch.nn.Conv2d(in_channels=128 + 64, out_channels=64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
            )
        ])
        self.to_RGB = torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, padding=0)
        self.decoder_FiLMs = torch.nn.ModuleList([
            torch.nn.Linear(in_features=palette_embedding_size * 2, out_features=512 * 2),
            torch.nn.Linear(in_features=palette_embedding_size * 2, out_features=256 * 2),
            torch.nn.Linear(in_features=palette_embedding_size * 2, out_features=128 * 2),
            torch.nn.Linear(in_features=palette_embedding_size * 2, out_features=64 * 2),
        ])
        assert len(self.decoder) == len(self.encoder_FiLMs)
        assert len(self.encoder) == len(self.decoder)

    # images: [B,C,H,W]
    # palettes: [B,N,3]
    # palette lens: [B]
    def forward(self, images: torch.Tensor, palettes: torch.Tensor, palette_mask: torch.Tensor) -> torch.Tensor:
        assert images.ndim == 4
        assert images.size(1) == 3

        batch_size = images.size(0)
        assert palettes.ndim == 3
        # assert palette_lens.ndim == 1
        assert palettes.size(0) == batch_size
        assert palettes.size(2) == 3
        # assert palette_lens.size(0) == batch_size

        palette_embedding = self.palette_encoder(palettes)  # Should be : [B,N,64]
        palette_len_max = palettes.size(1)
        # mask = torch.arange(palette_len_max,
        #                     device=palette_lens.device,
        #                     dtype=palette_lens.dtype).reshape(1, -1) < palette_lens.view(-1, 1)
        # [B,64*2]
        palette_embedding = torch.cat([torch.masked.mean(palette_embedding, dim=1,
                                                         mask=palette_mask.view(batch_size, palette_len_max, 1)),
                                       torch.masked.amax(palette_embedding, dim=1,
                                                         mask=palette_mask.view(batch_size, palette_len_max, 1)),
                                       ], dim=1)

        stack = []
        x = images
        for layer, film in zip(self.encoder, self.encoder_FiLMs):
            x = layer(x)
            gamma, beta = film(palette_embedding).chunk(2, dim=-1)
            x = x * gamma.view(batch_size, -1, 1, 1) + beta.view(batch_size, -1, 1, 1)
            stack.append(x)

        x = self.middle(x)
        gamma, beta = self.middle_FiLMs(palette_embedding).chunk(2, dim=-1)
        x = x * gamma.view(batch_size, -1, 1, 1) + beta.view(batch_size, -1, 1, 1)
        del gamma, beta

        for layer, film in zip(self.decoder, self.decoder_FiLMs):
            previous_x = stack[-1]
            stack.pop()
            x = torch.cat([x, previous_x], dim=1)
            x = layer(x)
            gamma, beta = film(palette_embedding).chunk(2, dim=-1)
            x = x * gamma.view(batch_size, -1, 1, 1) + beta.view(batch_size, -1, 1, 1)
            pass
        assert len(stack) == 0
        x = self.to_RGB(x)
        assert x.shape == images.shape
        return x


def test():
    model = SCFilter()
    torchinfo.summary(model,
                      input_data=[torch.rand((2, 3, 128, 128), dtype=torch.float32),
                                  torch.rand((2, 183, 3), dtype=torch.float32),
                                  torch.randint(0, 1, (2, 183)).to(torch.bool)
                                  ]
                      )
    pass


if __name__ == '__main__':
    test()
