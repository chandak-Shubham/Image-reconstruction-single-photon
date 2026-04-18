import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

from modified.dataloader.dataset_loader_final import get_test_scenes, SPCDataset
from modified.models.attention_resunet_01 import AttentionResUNet


DATASET = "/iitgn/home/arjun.badola/projects/spc/spc/Data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load test scenes
# -----------------------------
test_scenes = get_test_scenes(DATASET)

print(f"Found {len(test_scenes)} test frames.")

test_dataset = SPCDataset(test_scenes, mode="test")

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"Total test samples: {len(test_dataset)}")

# -----------------------------
# Create output folder
# -----------------------------
os.makedirs("attention_resunet_01_results", exist_ok=True)

# -----------------------------
# Load trained model
# -----------------------------
model = AttentionResUNet().to(device)

state_dict = torch.load(
    "attention_resunet_01/model_epoch_50.pth",
    map_location=device
)

state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()

# -----------------------------
# Inference
# -----------------------------
count = 0

with torch.no_grad():

    for idx, noisy in enumerate(test_loader):

        noisy = noisy.to(device).float() / 255.0

        with torch.amp.autocast("cuda"):
            pred = model(noisy)

        pred = pred.cpu()[0].permute(1, 2, 0).numpy()
        pred = np.clip(pred, 0, 1)

        # -----------------------------
        # Extract scene + frame name
        # -----------------------------
        frame_path = test_dataset.samples[idx]

        scene_name = os.path.basename(os.path.dirname(frame_path))
        frame_name = os.path.splitext(os.path.basename(frame_path))[0]

        # -----------------------------
        # Create scene folder
        # -----------------------------
        scene_folder = os.path.join(
            "attention_resunet_01_results",
            scene_name
        )

        os.makedirs(scene_folder, exist_ok=True)

        save_path = os.path.join(
            scene_folder,
            frame_name + ".png"
        )

        # -----------------------------
        # Save prediction
        # -----------------------------
        pred_img = (pred * 255).astype(np.uint8)
        Image.fromarray(pred_img).save(save_path)

        count += 1


print("Total test frames processed:", count)
print("All predictions saved in folder: attention_resunet_01_results")