from dataclasses import dataclass
from pathlib import Path

import torchvision.transforms as T
from PIL import Image
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from diffusion.model import create_model


def normalize_neg_one_to_one(img):
    return img * 2 - 1

# UPDATED LocalDataset class
class LocalDataset(Dataset):
    def __init__(self, folder, image_size, exts=None):
        super().__init__()
        if exts is None:
            exts = ['png']
        self.folder = folder
        self.image_size = image_size

        # Match extensions case-insensitively
        self.paths = []
        for ext in exts:
            self.paths.extend([
                p for p in Path(folder).rglob(f'*.{ext}')
                if p.suffix.lower() == f'.{ext.lower()}'
            ])

        assert len(self.paths) > 0, f"No images found in {folder}. Check path and extensions."

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Lambda(normalize_neg_one_to_one),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('L')  # Grayscale
        return self.transform(img)

@dataclass
class TrainingConfig:
    image_size: int = 128
    train_batch_size: int = 32
    eval_batch_size: int = 32
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 1
    save_model_epochs: int = 10
    mixed_precision: str = "fp16"
    output_dir: str = "glyffuser-unconditional"
    overwrite_output_dir: bool = True
    seed: int = 0
    dataset_name: str = "data128"


class ExperimentManager:
    def __init__(self, data_path: str, font_path=None, smokeConfig=None):
        # ---------- configuration ----------
        if smokeConfig is None:
            self.config = TrainingConfig()
        else:
            self.config = smokeConfig
        self.font_path = font_path#"NotoSansEgyptianHieroglyphs-Regular.ttf"

        # ---------- data ----------
        self.dataset = LocalDataset(data_path, image_size=self.config.image_size)
        self.train_dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
        )

        # ---------- model ----------
        self.model = create_model(self.config)

        # ---------- schedulers / optimiser ----------
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        total_steps = len(self.train_dataloader) * self.config.num_epochs
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=total_steps,
        )

    # ----------------------------------------
    # Experiment wrappers
    # ----------------------------------------
    def run_min_data(self, data_path):
        from .experiments.min_data import run
        run(self)

    def run_attention(self):
        from .experiments.attention import run
        run(self)

    def run_hyperparameters(self):
        from .experiments.hyper import run
        run(self)

    def run_scheduler(self):
        from .experiments.scheduler import run
        run(self)
