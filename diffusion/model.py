from diffusers import UNet2DModel


# def create_model(config):
#     return UNet2DModel(
#         sample_size=config.image_size,
#         in_channels=1,
#         out_channels=1,
#         layers_per_block=1,
#         block_out_channels=(128, 128, 256, 256, 512, 512),
#         down_block_types=(
#             "DownBlock2D", "DownBlock2D", "DownBlock2D",
#             "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
#         ),
#         up_block_types=(
#             "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
#             "UpBlock2D", "UpBlock2D", "UpBlock2D",
#         ),
#     )

# Inside diffusion/experiments/scheduler.py

# ... other imports ...
from diffusers import UNet2DModel

def create_model(
    image_size, # <--- Add image_size here
    layers_per_block=1,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D", "DownBlock2D", "DownBlock2D",
        "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
        "UpBlock2D", "UpBlock2D", "UpBlock2D",
    ),
    # ... other potential arguments ...
):
    """
    Creates a UNet2DModel based on provided configuration.
    """
    model = UNet2DModel(
        sample_size=image_size, # <--- Use image_size here
        in_channels=1,
        out_channels=1,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        # ... pass other relevant args ...
    )
    return model

# ... rest of the file ...
