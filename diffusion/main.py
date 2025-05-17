import shutil
from google.colab import files

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 128  # assumes images are square
    train_batch_size = 32
    eval_batch_size = 32
    num_epochs = 10 #Only 10 epochs
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 10
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = None  # the model name
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
    dataset_name="data128"

config = TrainingConfig()


def run_and_save_experiment(
    exp_name,
    test_name,
    config,
    model,
    noise_scheduler,
    optimizer,
    dataloader,
    lr_scheduler,
    train_loop,
    make_labeled_grid,
    create_process_grid,
    num_processes=1,
    font_path=None
):
    # Set output path for this experiment/test combo
    config.output_dir = os.path.join("experiments", exp_name, test_name)
    os.makedirs(config.output_dir, exist_ok=True)

    # Train model
    args = (config, model, noise_scheduler, optimizer, dataloader, lr_scheduler)
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
    grid.save(os.path.join(base_path, "samples.png"))

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

    # Create ZIP of the output directory
    zip_path = f"{config.output_dir}_{test_name}.zip"
    print(f"📦 Zipping results to {zip_path}...")
    shutil.make_archive(base_name=zip_path.replace(".zip", ""), format="zip", root_dir=config.output_dir)

    # Automatically download the zip file
    files.download(zip_path)

    print(f"✅ Completed experiment: {exp_name}/{test_name}")