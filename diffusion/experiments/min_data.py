from diffusion.helper import run_and_save_experiment
from diffusion.main import ExperimentManager

"""# Research Minimum Dataset"""

import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import re

class EgyptianCharacterDataset(Dataset):
    def __init__(self, image_dir, angle_range=None, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = self._select_images(angle_range)

    def _select_images(self, angle_range):
        images = os.listdir(self.image_dir)
        if angle_range is None:
            return [os.path.join(self.image_dir, img) for img in images if not re.search(r"_-?\d+deg_rotated", img)]
        else:
            angles = set(f"{angle}deg_rotated" for angle in angle_range)
            return [
                os.path.join(self.image_dir, img)
                for img in images
                if any(f"_{angle}" in img for angle in angles)
            ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
      # Change to convert to grayscale ('L') instead of RGB ('RGB')
      image = Image.open(self.image_paths[idx]).convert("L")
      if self.transform:
          image = self.transform(image)
      return image

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add normalization to match the LocalDataset transformation
    # Assuming LocalDataset normalized to [-1, 1]
    # This is achieved by ToTensor (0-1) followed by lambda x: x * 2 - 1
    transforms.Lambda(lambda x: x * 2 - 1)
])

def run(manager: ExperimentManager):
    # Variants
    angles_none = None
    angles_small = list(range(-5, 6, 5))
    angles_large = list(range(-15, 16, 5))

    dataset_none = EgyptianCharacterDataset("data", angle_range=angles_none, transform=transform)
    dataset_small = EgyptianCharacterDataset("data", angle_range=angles_small, transform=transform)
    dataset_large = EgyptianCharacterDataset("data", angle_range=angles_large, transform=transform)

    dataloader_none = DataLoader(dataset_none, batch_size=manager.config.train_batch_size, shuffle=True)
    dataloader_small = DataLoader(dataset_small, batch_size=manager.config.train_batch_size, shuffle=True)
    dataloader_large = DataLoader(dataset_large, batch_size=manager.config.train_batch_size, shuffle=True)

    for test_name, dl in {
        "dataset_none": dataloader_none,
        "dataset_small": dataloader_small,
        "dataset_large": dataloader_large,
    }.items():
        print(f"Starting experiment: {test_name}")

        run_and_save_experiment(
            manager,
            exp_name="exp1",
            test_name=test_name,
            dataloader=dl,
        )

