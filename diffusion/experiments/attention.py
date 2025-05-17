
"""# Research with Unet model architecture"""
from diffusers import UNet2DModel

from diffusion.helper import run_and_save_experiment
from diffusion.main import ExperimentManager, TrainingConfig
from diffusion.model import create_model


def run(manager: ExperimentManager):
    """
    Compares UNet architectures:
        - with attention in one layer
        - without attention

    Uses manager defaults for config, dataloader, schedulers, etc.
    """

    # ---- 1. UNet without attention ----
    model_without_attention = UNet2DModel(
        sample_size=manager.config.image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(128, 256, 256, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"
        ),
    )

    run_and_save_experiment(
        manager,
        exp_name="attention",
        test_name="no_attention",
        model=model_without_attention,
    )

    # ---- 2. UNet with attention ----
    model_attn = create_model(manager.config.image_size)

    run_and_save_experiment(
        manager,
        exp_name="attention",
        test_name="with_attention",
        model=model_attn,
    )