
"""# Scheduler"""
import os

from diffusers import DDPMScheduler, DDIMScheduler, DDPMPipeline
from transformers import get_scheduler
import torch

from diffusion.helper import run_and_save_experiment
from diffusion.main import TrainingConfig, ExperimentManager
from diffusion.model import create_model


def run(manager: ExperimentManager):
    # Fixed configuration
    default_lr = 1e-4
    default_epochs = 10

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

    # Load model once for each experiment — inference only
    for noise_schedule_name, noise_scheduler_class in scheduler_variants:
        for lr_scheduler_type in lr_scheduler_types:
            print(f"\n🧪 Experiment: {noise_schedule_name} + LR Scheduler: {lr_scheduler_type}")

            # Setup training config
            config = TrainingConfig(**vars(manager.config))
            config.learning_rate = default_lr
            config.num_epochs = default_epochs
            config.output_dir = f"scheduler_test/{noise_schedule_name}_{lr_scheduler_type}"

            model = create_model(**vars(manager.config))

            # Setup scheduler
            noise_scheduler = noise_scheduler_class(num_train_timesteps=1000)

            dummy_optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            total_steps = len(manager.train_dataloader) * config.num_epochs
            lr_scheduler = get_scheduler(
                name=lr_scheduler_type,
                optimizer=dummy_optimizer,
                num_warmup_steps=config.lr_warmup_steps,
                num_training_steps=total_steps,
            )

            # Evaluate using the experiment-specific noise and lr schedulers

            run_and_save_experiment(
                manager,
                model=model,
                exp_name="scheduler",
                test_name=f"{noise_schedule_name}_{lr_scheduler_type}",
                noise_scheduler=noise_scheduler,
                lr_scheduler=lr_scheduler,
                # let optimizer default from manager
                # dataloader defaults from manager too
            )
