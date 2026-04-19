import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision import models

from dataloader.dataset_loader import get_train_scenes, SPCDataset
from model.attention_resunet import AttentionResUNet

DATASET = "Data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

# -------------------------------
# VGG FEATURE EXTRACTOR
# -------------------------------

vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].to(device)

for param in vgg.parameters():
    param.requires_grad = False

vgg.eval()

imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

# create folder to store results and TensorBoard writer
os.makedirs("attention_resunet_01", exist_ok=True)
writer = SummaryWriter(log_dir="runs/attention_resunet_01_experiment")

# initialize model
model = AttentionResUNet().to(device)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

scaler = torch.amp.GradScaler("cuda")

# Cosine Annealing LR Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,
    eta_min=1e-6
)

# -------------------------------
# DataLoader setup
# -------------------------------

batch_size = 2
num_workers = 4
prefetch_factor = 2 

train_scenes = get_train_scenes(DATASET)
print(f"Loading {len(train_scenes)} training scenes...")
train_dataset = SPCDataset(train_scenes,mode="train")
print(f"Found {len(train_dataset)} training samples.")

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=prefetch_factor
)

train_losses = []
global_step = 0

# -------------------------------
# TRAINING LOOP
# -------------------------------

for epoch in range(50):

    model.train()

    total_loss = 0
    total_ssim_loss = 0
    total_l1_loss = 0

    for batch_idx, (noisy, clean) in enumerate(train_loader):

        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        noisy = noisy.float() / 255.0
        clean = clean.float() / 255.0

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):

            output = model(noisy)

            # pixel loss
            l1 = F.l1_loss(output, clean)

            # structural loss
            ssim_loss = 1 - structural_similarity_index_measure(
                output, clean, data_range=1.0
            )

            # -------------------------------
            # VGG LOSS
            # -------------------------------

            output_vgg = (output - imagenet_mean) / imagenet_std
            clean_vgg = (clean - imagenet_mean) / imagenet_std

            vgg_out = vgg(output_vgg)
            vgg_gt = vgg(clean_vgg)

            vgg_loss = F.l1_loss(vgg_out, vgg_gt)

            # final combined loss
            loss = 0.7 * l1 + 0.2 * ssim_loss + 0.1 * vgg_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()

        total_loss += loss_val
        total_ssim_loss += ssim_loss.item()
        total_l1_loss += l1.item()

        if batch_idx % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/50] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss_val:.4f} | LR: {lr:.6f}")

        if global_step % 50 == 0:
            writer.add_scalar("Batch/Total_Loss", loss_val, global_step)
            writer.add_scalar("Batch/L1_Loss", l1.item(), global_step)
            writer.add_scalar("Batch/SSIM_Loss", ssim_loss.item(), global_step)
            writer.add_scalar("Batch/VGG_Loss", vgg_loss.item(), global_step)

        # Log visual progress
        if global_step % 500 == 0:
            writer.add_image("Images/Noisy_Input_Frame0", noisy[0, :3].clamp(0, 1), global_step)
            writer.add_image("Images/Clean_Target", clean[0].clamp(0, 1), global_step)
            writer.add_image("Images/Model_Output", output[0].clamp(0, 1), global_step)

        global_step += 1

    avg_loss = total_loss / len(train_loader)
    avg_ssim = total_ssim_loss / len(train_loader)
    avg_l1 = total_l1_loss / len(train_loader)

    train_losses.append(avg_loss)

    writer.add_scalar("Epoch/Total_Loss", avg_loss, epoch + 1)
    writer.add_scalar("Epoch/L1_Loss", avg_l1, epoch + 1)
    writer.add_scalar("Epoch/SSIM_Loss", avg_ssim, epoch + 1)

    print(
        f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | "
        f"L1: {avg_l1:.4f} | SSIM: {avg_ssim:.4f}"
    )

    scheduler.step()

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"attention_resunet_01/model_epoch_{epoch+1}.pth")

writer.close()

# -------------------------------
# AFTER TRAINING
# -------------------------------

epochs = range(1, 51)

plt.figure()
plt.plot(epochs, train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.xlim(1, 50)
plt.grid(True)
plt.savefig("attention_resunet_01/loss_curve.png")

# save loss values to csv
with open("attention_resunet_01/training_loss.csv", "w") as f:
    f.write("Epoch,Loss\n")
    for epoch, loss in enumerate(train_losses, start=1):
        f.write(f"{epoch},{loss}\n")

print("Training complete. Run `tensorboard --logdir=runs` to view the logs.")
