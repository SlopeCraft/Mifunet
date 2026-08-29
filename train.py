import torch
import lpips
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np
from triton.profiler.viewer import width

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
                                               batch_size=8,
                                               shuffle=True,
                                               # pin_memory=True,
                                               # pin_memory_device=device_str,
                                               )
    validation_loader = torch.utils.data.DataLoader(validate_ds,
                                                    batch_size=8,
                                                    shuffle=False,
                                                    )
    train_history: list[float] = []
    validate_history: list[float] = [math.nan]

    N_epochs = 1
    for epoch in range(N_epochs):
        train_loss_sum = 0.
        train_samples_count = 0
        tau = 1.5
        sc_filter.train()
        for batch_idx, src_img in enumerate(train_loader):
            src_img = src_img.to(device)
            img = sc_filter(src_img)
            convert_dict = palette.forward(img, tau=tau)
            converted_img = convert_dict["converted_image"]
            loss = torch.mean(teacher(src_img, converted_img), dim=0)

            train_loss_sum += loss.item() * src_img.size(0)
            train_samples_count += src_img.size(0)
            train_history.append(loss.item())

            loss.backward()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, batch [{batch_idx}/{len(train_loader)}]: {loss.item()}")
            optimizer.step()
            optimizer.zero_grad()

        trains_loss = train_loss_sum / train_samples_count
        print(f"Epoch {epoch}, Train loss: {trains_loss}")

        sc_filter.eval()
        val_loss_sum = 0.
        val_samples_count = 0
        # Calc loss on evaluation set
        for src_img in validation_loader:
            src_img = src_img.to(device)
            img = sc_filter(src_img)
            convert_dict = palette.forward(img, tau=tau)
            converted_img = convert_dict["converted_image"]
            loss = torch.mean(teacher(src_img, converted_img), dim=0)

            val_loss_sum += loss.item() * src_img.size(0)
            val_samples_count += src_img.size(0)
        validate_loss = val_loss_sum / val_samples_count
        print(f"Epoch {epoch}, Validation loss: {validate_loss}")
        validate_history.append(validate_loss)

    plt.plot(np.linspace(0., float(N_epochs), len(train_history)), np.array(train_history), linewidth=1, label="Train")
    plt.plot(np.arange(len(validate_history)), validate_history, 'o', label="Validate")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


main()
