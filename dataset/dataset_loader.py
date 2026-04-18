import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from modified.utils.unpack_final import unpack


def get_train_scenes(dataset_path):
    train_scenes = []

    for folder in sorted(Path(dataset_path).glob("train_*")):
        scene_root = folder / "train"

        if scene_root.exists():
            for scene in scene_root.iterdir():
                if scene.is_dir():
                    train_scenes.append(scene)

    print("Total train scenes:", len(train_scenes))
    print("First 5 train scenes:", train_scenes[:5])

    return train_scenes


def get_test_scenes(dataset_path):
    test_scenes = []

    for folder in sorted(Path(dataset_path).glob("test_*")):
        scene_root = folder / "test"

        if scene_root.exists():
            for scene in scene_root.iterdir():
                if scene.is_dir():
                    test_scenes.append(scene)

    print("Total test scenes:", len(test_scenes))
    print("First 5 test scenes:", test_scenes[:5])

    return test_scenes


class SPCDataset(Dataset):

    def __init__(self, scenes, mode="train"):

        self.samples = []
        self.mode = mode

        for scene in scenes:

            scene_path = str(scene)

            if not os.path.isdir(scene_path):
                continue

            files = os.listdir(scene_path)

            # collect npy files
            npy_files = {Path(f).stem: f for f in files if f.endswith(".npy")}

            if self.mode == "train":

                png_files = {Path(f).stem: f for f in files if f.endswith(".png")}

                common_keys = sorted(set(npy_files.keys()) & set(png_files.keys()))

                if len(common_keys) == 0:
                    raise RuntimeError(f"No matching npy/png pairs found in {scene_path}")

                for key in common_keys:

                    npy_file = npy_files[key]
                    png_file = png_files[key]

                    self.samples.append((
                        os.path.join(scene_path, npy_file),
                        os.path.join(scene_path, png_file)
                    ))

            else:
                # test mode → only npy files
                for key in sorted(npy_files.keys()):

                    npy_file = npy_files[key]

                    self.samples.append(
                        os.path.join(scene_path, npy_file)
                    )

        print("Total dataset samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        if self.mode == "train":
            npy_path, png_path = self.samples[idx]
        else:
            npy_path = self.samples[idx]

        # load photon data
        photon = np.load(npy_path)

        # unpack compressed frames
        photon = unpack(photon)   # (128,800,800,3)

        # convert noisy tensor
        noisy = torch.from_numpy(photon)

        # (128,800,800,3) → (128,3,800,800)
        noisy = noisy.permute(0, 3, 1, 2)

        # flatten frames to channels
        noisy = noisy.reshape(384, noisy.shape[2], noisy.shape[3])

        if self.mode == "train":

            # load clean image
            clean = np.array(Image.open(png_path))  # (800,800,3)

            clean_tensor = torch.from_numpy(clean)
            clean_tensor = clean_tensor.permute(2, 0, 1)

            return noisy, clean_tensor

        else:
            return noisy


# quick debug test for dataset loader
if __name__ == "__main__":

    DATASET = "/iitgn/home/arjun.badola/projects/spc/spc/Data"

    train_scenes = get_train_scenes(DATASET)

    dataset = SPCDataset(train_scenes, mode="train")

    print("Dataset size:", len(dataset))

    noisy, clean = dataset[0]

    print("Noisy shape:", noisy.shape)
    print("Clean shape:", clean.shape)

    print("\nTesting test dataset loader")

    test_scenes = get_test_scenes(DATASET)

    test_dataset = SPCDataset(test_scenes, mode="test")

    print("Test dataset size:", len(test_dataset))

    noisy = test_dataset[0]

    print("Test noisy shape:", noisy.shape)