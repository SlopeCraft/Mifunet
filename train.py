import torch
import lpips
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np
import os

from pytorch_msssim import ssim

from dataset import ImgDataSet
from palette import Palette, load_palette
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

    palette = Palette(load_palette("./binary/colorset-Slope.png"))
    palette.to_(device)

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
                                                    batch_size=128,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    # num_workers=1,
                                                    # persistent_workers=True,
                                                    # prefetch_factor=2,
                                                    )
    train_history: list[float] = []
    validate_history: list[float] = [math.nan]

    def calc_loss(src_img_on_cpu: torch.Tensor, tau: float, train: bool) -> torch.Tensor:
        src_img = src_img_on_cpu.to(device)
        img = sc_filter(src_img)
        convert_dict = palette.forward(img, tau=tau, dtype=torch.float32 if train else torch.bfloat16)
        converted_img = convert_dict["converted_image"]
        similarity_loss = torch.mean(teacher(src_img, converted_img), dim=0)
        # Force UNet output to approach converted
        # color_diff_loss = torch.mean(convert_dict["color_diff_selected"])

        overflow_loss = torch.mean(torch.nn.functional.relu(img - 1.0) ** 2 + torch.nn.functional.relu(-img) ** 2)

        ssim_loss = (1 - ssim(src_img, converted_img, data_range=1.0, size_average=True)) / 2

        loss = (similarity_loss
                + 0.4 * ssim_loss
                + 0.1 / tau * overflow_loss
                # + 1e-2 * color_diff_loss
                )

        return loss


    N_epochs = 16
    for epoch in range(N_epochs):
        tau = 0.5 * math.pow(0.75, epoch) + 1e-2
        print(f"tau = {tau}")
        sc_filter.train()
        for batch_idx, src_img in enumerate(train_loader):
            loss = calc_loss(src_img, tau, True)
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
            for src_img in validation_loader:
                loss = calc_loss(src_img, tau, False)

                val_loss_sum += float(loss.item()) * src_img.size(0)
                val_samples_count += src_img.size(0)
        validate_loss = val_loss_sum / val_samples_count
        print(f"Epoch {epoch}, Validation loss: {validate_loss}")
        validate_history.append(validate_loss)

        torch.save({
            'scFilter': sc_filter,
            # 'teacher': teacher,
            'optimizer': optimizer,
            'palette': palette,
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
