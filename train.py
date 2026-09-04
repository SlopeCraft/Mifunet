import torch
import lpips
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np
import os

from pytorch_msssim import ssim

from dataset import ImgDataSet
import palette
from sc_filter import SCFilter


def main():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"Using device {torch.cuda.get_device_name(device)}")

    fig_dir = Path("./binary/figures")
    os.makedirs(fig_dir, exist_ok=True)

    checkpoint_dir = Path("./binary/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    full_dataset = ImgDataSet(Path('./binary/dataset.h5'), key="128x128")
    N_test = int(len(full_dataset) * 0.2)
    N_validate = int(len(full_dataset) * 0.2)
    # N_train = len(full_dataset) - N_test - N_validate

    test_ds = torch.utils.data.Subset(full_dataset, range(0, N_test))
    validate_ds = torch.utils.data.Subset(full_dataset, range(N_test, N_test + N_validate))
    train_ds = torch.utils.data.Subset(full_dataset, range(N_test + N_validate, len(full_dataset)))

    # train_ds, validate_ds, test_ds = torch.utils.data.random_split(full_da/enerator().manual_seed(42))
    # Student model
    sc_filter = SCFilter().to(device)

    palette_dataset = palette.PaletteDataSet.from_files(
        [
            "./binary/colorset-Slope.png",
            "./binary/colorset-Slope-old.png",
            "./binary/colorset-Flat.png",
            "./binary/colorset-Flat-old.png",
            "./binary/colorset-FileOnly.png",
            "./binary/colorset-FileOnly-old.png",
        ]
    )
    # palette = Palette(load_palette("./binary/colorset-Slope.png"))
    # palette.to_(device)

    teacher = lpips.LPIPS(net='vgg', pretrained=True).to(device)
    teacher.eval()
    teacher.requires_grad_(False)

    optimizer = torch.optim.AdamW(sc_filter.parameters(), lr=5e-4, fused=True)

    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=16,
                                               shuffle=True,
                                               pin_memory=True,
                                               # num_workers=1,
                                               # persistent_workers=True,
                                               # prefetch_factor=2,
                                               )
    validation_loader = torch.utils.data.DataLoader(validate_ds,
                                                    batch_size=32,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    # num_workers=1,
                                                    # persistent_workers=True,
                                                    # prefetch_factor=2,
                                                    )
    train_history: list[float] = []
    validate_history: list[float] = [math.nan]

    def calc_loss(src_img_on_cpu: torch.Tensor,
                  palettes_on_cpu: torch.Tensor,
                  palette_lens_on_cpu: torch.Tensor,
                  tau: float,
                  train: bool) -> torch.Tensor:
        src_img_ = src_img_on_cpu.to(device)
        palettes_ = palettes_on_cpu.to(device)
        palette_lens_ = palette_lens_on_cpu.to(device)
        batch_size = src_img_.size(0)

        mask = palette.make_palette_mask(palette_lens_, palettes_.shape[1])

        img = sc_filter(src_img_, palettes_, mask)
        convert_dict = palette.convert_images(palettes_, mask.view(batch_size, -1, 1), img, tau=tau,
                                              dtype=torch.float32 if train else torch.bfloat16)
        converted_img = convert_dict["converted_image"]
        similarity_loss = torch.mean(teacher(src_img_, converted_img), dim=0)
        # Force UNet output to approach converted
        # color_diff_loss = torch.mean(convert_dict["color_diff_selected"])

        overflow_loss = torch.mean(torch.nn.functional.relu(img - 1.0) ** 2 + torch.nn.functional.relu(-img) ** 2)

        ssim_loss = (1 - ssim(src_img_, converted_img, data_range=1.0, size_average=True)) / 2

        loss_ = (similarity_loss
                 # + 0.2 * ssim_loss
                 + 0.1 / tau * overflow_loss
                 # + 1e-2 * color_diff_loss
                 )

        return loss_

    N_epochs = 16
    for epoch in range(N_epochs):
        tau = max(1 - 0.1 * epoch, 1e-2)
        print(f"tau = {tau}")
        sc_filter.train()
        for batch_idx, src_img in enumerate(train_loader):
            rand_idx = torch.randint(0, len(palette_dataset), [src_img.size(0)])
            pals, pal_lens = palette_dataset[rand_idx]
            loss = calc_loss(src_img, tau=tau, palettes_on_cpu=pals, palette_lens_on_cpu=pal_lens, train=True)
            if batch_idx % 5 == 0:
                train_history.append(float(loss.item()))

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, batch [{batch_idx}/{len(train_loader)}]: {loss.item()}")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        sc_filter.eval()
        val_loss_sum = 0.
        val_samples_count = 0
        # Calc loss on evaluation set
        with torch.no_grad():
            begin_idx = 0
            for src_img in validation_loader:
                idx = torch.arange(start=begin_idx, end=begin_idx + src_img.size(0), dtype=torch.int32) % len(
                    palette_dataset)
                begin_idx += src_img.size(0)
                pals, pal_lens = palette_dataset[idx]
                loss = calc_loss(src_img, tau=tau, palettes_on_cpu=pals, palette_lens_on_cpu=pal_lens, train=False)

                val_loss_sum += float(loss.item()) * src_img.size(0)
                val_samples_count += src_img.size(0)
        validate_loss = val_loss_sum / val_samples_count
        print(f"Epoch {epoch}, Validation loss: {validate_loss}")
        validate_history.append(validate_loss)

        torch.cuda.empty_cache()

        torch.save({
            'scFilter': sc_filter,
            # 'teacher': teacher,
            'optimizer': optimizer,
            'palette_dataset': palette_dataset,
        }, checkpoint_dir / f"epoch{epoch + 1}.pth")

    plt.figure("history")
    plt.plot(np.linspace(0., float(N_epochs), len(train_history)), np.array(train_history), linewidth=1, label="Train")
    plt.plot(np.arange(len(validate_history)), validate_history, 'o', label="Validate")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1)
    # plt.yscale("log")
    plt.legend()

    plt.savefig(fig_dir / "history.svgz", transparent=True)
    # plt.show()


main()
