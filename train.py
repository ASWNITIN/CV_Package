import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

from dataset import get_dataloaders
from model import Retinexformer


def calculate_ssim_loss(enhanced, target):
    """
    Calculate the SSIM loss for a batch of images.
    enhanced, target: (B, C, H, W)
    """
    enhanced_np = enhanced.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    batch_ssim = 0.0
    for i in range(enhanced_np.shape[0]):
        enh_img = enhanced_np[i].transpose(1, 2, 0)
        tar_img = target_np[i].transpose(1, 2, 0)
        batch_ssim += ssim(enh_img, tar_img, multichannel=True,
                           data_range=1.0)

    batch_ssim /= enhanced_np.shape[0]
    return 1 - batch_ssim  # SSIM loss


def train(root_dir,
          epochs=50,
          batch_size=4,
          lr=1e-4,
          num_workers=4,
          save_dir='checkpoints'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loaders
    train_loader, _ = get_dataloaders(root_dir=root_dir,
                                      batch_size=batch_size,
                                      num_workers=num_workers)

    # Model, optimizer, and loss
    model = Retinexformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    l1_loss_fn = nn.L1Loss()

    # Logging using TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))

    os.makedirs(save_dir, exist_ok=True)

    step = 0
    best_loss = float('inf')

    for epoch in range(epochs):
        epoch_l1 = 0.0
        epoch_ssim = 0.0
        model.train()

        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, targets, names in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")

                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                outputs, illum = model(inputs)

                # L1 loss
                l1_loss = l1_loss_fn(outputs, targets)

                # SSIM Loss
                ssim_loss = calculate_ssim_loss(outputs, targets)

                # Combined loss
                total_loss = l1_loss + 0.1 * ssim_loss

                total_loss.backward()
                optimizer.step()

                epoch_l1 += l1_loss.item()
                epoch_ssim += ssim_loss

                # Logging
                writer.add_scalar('Loss/L1', l1_loss.item(), step)
                writer.add_scalar('Loss/SSIM', ssim_loss, step)
                writer.add_scalar('Loss/Total', total_loss.item(), step)
                step += 1

                tepoch.set_postfix({
                    'L1 Loss': f"{l1_loss.item():.4f}",
                    'SSIM Loss': f"{ssim_loss:.4f}"
                })

        avg_epoch_l1 = epoch_l1 / len(train_loader)
        avg_epoch_ssim = epoch_ssim / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} - Avg L1: {avg_epoch_l1:.4f}, Avg SSIM Loss: {avg_epoch_ssim:.4f}"
        )

        # Save best model
        if avg_epoch_l1 < best_loss:
            best_loss = avg_epoch_l1
            save_path = os.path.join(save_dir, 'Model.yml')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model at {save_path}")

        # Save example output image
        if (epoch + 1) % 5 == 0:
            save_image(outputs, os.path.join(save_dir,
                                             f'output_epoch_{epoch+1}.png'))

    writer.close()


if __name__ == '__main__':
    train(root_dir='dataset',
          epochs=100,
          batch_size=4,
          lr=1e-4,
          num_workers=2,
          save_dir='Model')
