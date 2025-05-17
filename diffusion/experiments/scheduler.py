import os

import torch
from diffusers import DDPMScheduler, DDIMScheduler
from transformers import get_scheduler

from diffusion.helper import run_and_save_experiment
from diffusion.main import ExperimentManager
from diffusion.model import create_model
from torch.optim import AdamW


def run(manager: ExperimentManager):
    # Fixed configuration
    # default_lr = 1e-4
    # default_epochs = 10

    # Variants to test
    scheduler_variants = [
        ("DDPMScheduler", DDPMScheduler),
        ("DDIMScheduler", DDIMScheduler),
    ]

    lr_scheduler_types = [
        "constant",
        "cosine",
        "linear",
    ]

    # Debug the paths before we start
    print("\n===== DEBUG: Path Information =====")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Manager config output_dir: {manager.config.output_dir}")

    # Create base output directory to avoid problems
    base_output_dir = "/content/experiments/scheduler"
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Created base output directory: {base_output_dir}")
    print("===================================\n")

    # Load model once for each experiment — inference only
    for noise_schedule_name, noise_scheduler_class in scheduler_variants:
        for lr_scheduler_type in lr_scheduler_types:
            print(f"\n🧪 Experiment: {noise_schedule_name} + LR Scheduler: {lr_scheduler_type}")

            # Setup training config
            # config = TrainingConfig(**vars(manager.config))
            # config.learning_rate = default_lr
            # config.num_epochs = default_epochs

            # Let run_and_save_experiment set the output directory correctly
            # But keep the test name clear and consistent
            test_name = f"{noise_schedule_name}_{lr_scheduler_type}"
            experiment_output_dir = os.path.join(base_output_dir, test_name)
            print(f"Expected output directory will be: {experiment_output_dir}")

            # Create a fresh model
            model = create_model(**vars(manager.config))

            # Setup scheduler
            noise_scheduler = noise_scheduler_class(num_train_timesteps=1000)

            # Create a custom optimizer with the experiment's learning rate
            optimizer = AdamW(model.parameters(), lr=manager.config.learning_rate)

            # Create a custom LR scheduler
            total_steps = len(manager.train_dataloader) * manager.config.num_epochs
            lr_scheduler = get_scheduler(
                name=lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=manager.config.lr_warmup_steps,
                num_training_steps=total_steps,
            )

            # Evaluate using the experiment-specific noise and lr schedulers
            run_and_save_experiment(
                manager,
                model=model,
                optimizer=optimizer,
                exp_name="scheduler",  # This will be the top-level folder
                test_name=test_name,  # The subfolder for this specific experiment
                # noise_scheduler=noise_scheduler,
                # lr_scheduler=lr_scheduler,
            )

            # Print the path where we should find the model after training
            final_model_dir = os.path.join(experiment_output_dir, f"epoch{manager.config.num_epochs - 1}")
            print(f"Model should be saved at: {final_model_dir}")
            print(f"Directory exists: {os.path.exists(final_model_dir)}")