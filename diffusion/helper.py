

"""### Step 5: Create Local Dataset Class and helper functions"""
from __future__ import annotations

from diffusion.train import train_loop, make_grid


"""### Step 12: Visualize model generated images"""


def make_labeled_grid(images, prompt, steps, font_path=None, font_size=20, margin=10):
    assert len(images) == len(steps), "The number of images must match the number of steps"

    w, h = images[0].size
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    # Calculate the height of the grid including the margin for text
    total_height = h + margin + font_size
    total_width = w * len(images)
    grid_height = total_height + margin + font_size  # Add extra margin for the prompt
    grid = Image.new('RGB', size=(total_width, grid_height), color=(255, 255, 255))
    # Draw the text prompt at the top
    draw = ImageDraw.Draw(grid)
    prompt_text = f"Prompt: \"{prompt}\""
    prompt_width, prompt_height = draw.textbbox((0, 0), prompt_text, font=font)[2:4]
    prompt_x = (total_width - prompt_width) / 2
    prompt_y = margin / 2
    draw.text((prompt_x, prompt_y), prompt_text, fill="black", font=font)

    for i, (image, step) in enumerate(zip(images, steps)):
        # Calculate position to paste the image
        x = i * w
        y = margin + font_size

        # Paste the image
        grid.paste(image, box=(x, y))

        # Draw the step text
        step_text = f"Steps: {step}"
        text_width, text_height = draw.textbbox((0, 0), step_text, font=font)[2:4]
        text_x = x + (w - text_width) / 2
        text_y = y + h + margin / 2 - 8
        draw.text((text_x, text_y), step_text, fill="black", font=font)
    return grid

"""### Step 13: Create animated visualization"""

from diffusers import DPMSolverMultistepScheduler, DDPMPipeline
from PIL import ImageDraw, ImageFont, Image
import torch
import numpy as np
import os



# Create a grid showing selected frames
def create_process_grid(frames, num_to_show=8):
    # Select frames evenly throughout the process
    if len(frames) <= num_to_show:
        selected_frames = frames
    else:
        indices = np.linspace(0, len(frames)-1, num_to_show, dtype=int)
        selected_frames = [frames[i] for i in indices]

    # Resize frames
    width, height = 256, 256
    selected_frames = [frame.resize((width, height)) for frame in selected_frames]

    # Create grid image
    cols = min(4, num_to_show)
    rows = (num_to_show + cols - 1) // cols

    grid = Image.new('RGB', (width * cols, height * rows))

    for i, frame in enumerate(selected_frames):
        row = i // cols
        col = i % cols
        grid.paste(frame, (col * width, row * height))

    return grid


"""# Wrapper for research experiment"""

def run_and_save_experiment(
    manager,
    *,
    exp_name: str,
    test_name: str,
    model=None,
    optimizer=None,
    noise_scheduler=None,
    lr_scheduler=None,
    dataloader=None,
    num_processes: int = 1,
):
    """
    Shared utility to train / evaluate and dump artifacts.

    Parameters
    ----------
    manager : ExperimentManager
        Object that already holds the default config, model, schedulers, etc.
    exp_name : str
        Top-level experiment folder (e.g., "exp1").
    test_name : str
        Sub-folder for this run (e.g., "dataset_none").
    model, optimizer, noise_scheduler, lr_scheduler, dataloader : optional
        If omitted, defaults are taken from `manager`.
    num_processes : int
    """
    # -------- fetch defaults from manager --------
    config          = manager.config
    model           = model           or manager.model
    optimizer       = optimizer       or manager.optimizer
    noise_scheduler = noise_scheduler or manager.noise_scheduler
    lr_scheduler    = lr_scheduler    or manager.lr_scheduler
    dataloader      = dataloader      or manager.train_dataloader
    # Set output path for this experiment/test combo
    config.output_dir = os.path.join("experiments", exp_name, test_name)
    os.makedirs(config.output_dir, exist_ok=True)

    font_path = manager.font_path

    # Train model
    args = (config, model, noise_scheduler, optimizer, dataloader, lr_scheduler)
    from accelerate import notebook_launcher
    notebook_launcher(train_loop, args, num_processes=num_processes)

    # Load model pipeline from final saved epoch
    model_dir = os.path.join(config.output_dir, f"epoch{config.num_epochs - 1}")
    pipeline = DDPMPipeline.from_pretrained(model_dir).to("cuda")
    pipeline.scheduler = DPMSolverMultistepScheduler()

    print("🌀 Generating image samples from trained model...")
    images = pipeline(
        batch_size=16,
        generator=torch.Generator(device='cuda').manual_seed(config.seed),
        num_inference_steps=50
    ).images

    # Save sample grid
    base_path = config.output_dir
    grid = make_grid(images, rows=4, cols=4)
    grid_image = grid.permute(1, 2, 0).mul(255).byte().cpu().numpy()
    from PIL import Image
    Image.fromarray(grid_image).save(os.path.join(base_path, "samples.png"))

    # Labeled grid from step-wise generations
    print("🌀 Creating labeled grid...")
    num_inference_steps_list = [1, 2, 3, 5, 10, 20, 50]
    stepwise_images = []

    for steps in num_inference_steps_list:
        result = pipeline(
            batch_size=1,
            generator=torch.Generator(device='cuda').manual_seed(config.seed),
            num_inference_steps=steps
        ).images
        stepwise_images.append(result[0])

    labeled_grid = make_labeled_grid(
        stepwise_images,
        prompt="Unconditional Diffusion Model Output",
        steps=num_inference_steps_list,
        font_path=font_path
    )
    labeled_grid.save(os.path.join(base_path, "labeled_grid.png"))

    # Generate GIF from denoising steps
    print("🌀 Creating diffusion process GIF and grid...")
    frames = []
    for step in range(0, 51, 2):
        current_step = step if step > 0 else 1
        img = pipeline(
            batch_size=1,
            generator=torch.Generator(device='cuda').manual_seed(config.seed),
            num_inference_steps=current_step
        ).images[0]
        frames.append(img)

    frames[0].save(
        os.path.join(base_path, "diffusion_process.gif"),
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0
    )

    process_grid = create_process_grid(frames)
    process_grid.save(os.path.join(base_path, "process_grid.png"))

    # # Create ZIP of top-level files
    # zip_filename = f"{config.output_dir}.zip"
    # with zipfile.ZipFile(zip_filename, 'w') as zipf:
    #     for file in os.listdir(config.output_dir):
    #         filepath = os.path.join(config.output_dir, file)
    #         if os.path.isfile(filepath):
    #             zipf.write(filepath, arcname=file)
    #
    # # Download in Colab only
    # if is_colab():
    #     from google.colab import files
    #     files.download(zip_filename)

    print(f"✅ Completed experiment: {exp_name}/{test_name}")