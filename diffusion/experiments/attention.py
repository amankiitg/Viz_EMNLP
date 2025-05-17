
"""# Research with Unet model architecture"""
from diffusers import UNet2DModel

from diffusion.helper import run_and_save_experiment
from diffusion.main import ExperimentManager, TrainingConfig


def run(manager: ExperimentManager):
    """
    Compares UNet architectures:
        - with attention in one layer
        - without attention

    Uses manager defaults for config, dataloader, schedulers, etc.
    """

    # ---- 1. UNet without attention ----
    model_no_attn = UNet2DModel(
        sample_size=manager.config.image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",  # no attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",  # no attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    config_no_attn = TrainingConfig(**vars(manager.config))
    config_no_attn.output_dir = "exp3/unet_no_attention"

    run_and_save_experiment(
        manager,
        exp_name="exp3",
        test_name="no_attention",
        model=model_no_attn,
        config=config_no_attn,
    )

    # ---- 2. UNet with attention ----
    model_attn = UNet2DModel(
        sample_size=manager.config.image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # attention added
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",  # attention added
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    config_attn = TrainingConfig(**vars(manager.config))
    config_attn.output_dir = "exp3/unet_attention"

    run_and_save_experiment(
        manager,
        exp_name="exp3",
        test_name="with_attention",
        model=model_attn,
        config=config_attn,
    )
