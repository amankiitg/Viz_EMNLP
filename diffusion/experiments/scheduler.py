
"""# Scheduler"""
import os

from diffusers import DDPMScheduler, DDIMScheduler, DDPMPipeline
from transformers import get_scheduler
import torch

from diffusion.helper import run_and_save_experiment
from diffusion.main import TrainingConfig, ExperimentManager


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
    for noise_sched_name, noise_sched_class in scheduler_variants:
        for lr_sched_type in lr_scheduler_types:
            print(f"\n🧪 Experiment: {noise_sched_name} + LR Scheduler: {lr_sched_type}")

            # Setup training config
            config = TrainingConfig(**vars(manager.config))
            config.learning_rate = default_lr
            config.num_epochs = default_epochs
            config.output_dir = f"exp4_scheduler_test/{noise_sched_name}_{lr_sched_type}"

            # Load model saved from epoch 9 of prior training
            model_dir = os.path.join(config.output_dir, "epoch9")
            pipeline = DDPMPipeline.from_pretrained(model_dir)

            # Setup scheduler
            noise_scheduler = noise_sched_class(num_train_timesteps=1000)

            dummy_optimizer = torch.optim.Adam(pipeline.unet.parameters(), lr=config.learning_rate)
            total_steps = len(manager.train_dataloader) * config.num_epochs
            lr_scheduler = get_scheduler(
                name=lr_sched_type,
                optimizer=dummy_optimizer,
                num_warmup_steps=config.lr_warmup_steps,
                num_training_steps=total_steps,
            )

            # Evaluate using the experiment-specific noise and lr schedulers

            run_and_save_experiment(
                manager,
                model=pipeline.unet,
                exp_name="exp4",
                test_name=f"{noise_sched_name}_{lr_sched_type}",
                noise_scheduler=noise_scheduler,
                lr_scheduler=lr_scheduler,
                # let optimizer default from manager
                # dataloader defaults from manager too
            )
