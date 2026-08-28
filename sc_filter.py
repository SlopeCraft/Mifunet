import torch
import torchinfo


class SCFilter(torch.nn.Module):
    def __init__(self):
        super(SCFilter, self).__init__()

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

        self.middle = torch.nn.Sequential(  # 512->512->512
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

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
                torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, padding=0),
            )
        ])

        assert len(self.encoder) == len(self.decoder)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B,C,H,W]
        assert images.ndim == 4
        assert images.size(1) == 3

        stack = []
        x = images
        for layer in self.encoder:
            x = layer(x)
            stack.append(x)

        x = self.middle(x)
        for layer in self.decoder:
            previous_x = stack[-1]
            stack.pop()
            x = torch.cat([x, previous_x], dim=1)
            x = layer(x)
        assert len(stack) == 0
        assert x.shape == images.shape
        return x


def test():
    model = SCFilter()
    torchinfo.summary(model, input_size=(1, 3, 128, 128))
    pass


if __name__ == '__main__':
    test()
