import torch
import lpips
from pathlib import Path

from dataset import ImgDataSet
from palette import Palette, load_palette
from sc_filter import SCFilter


def main():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"Using device {torch.cuda.get_device_name(device)}")

    full_dataset = ImgDataSet(Path('./binary/dataset.h5'), key="128x128")

    train_ds, validate_ds, test_ds = torch.utils.data.random_split(full_dataset, [0.6, 0.2, 0.2],
                                                                   torch.Generator().manual_seed(42))
    # Student model
    sc_filter = SCFilter().to(device)

    palette = Palette(load_palette("./binary/colorset-Slope.png"))
    palette.to_(device)

    teacher = lpips.LPIPS(net='vgg', pretrained=True).to(device)
    teacher.eval()

    optimizer = torch.optim.Adam(sc_filter.parameters(), lr=5e-4)

    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=16,
                                               shuffle=True,
                                               # pin_memory=True,
                                               # pin_memory_device=device_str,
                                               )

    for batch_idx, src_img in enumerate(train_loader):
        src_img = src_img.to(device)
        img = sc_filter(src_img)
        convert_dict = palette.forward(img, tau=1.5)
        converted_img = convert_dict["converted_image"]

        loss = torch.mean(teacher(src_img, converted_img))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    pass


main()
