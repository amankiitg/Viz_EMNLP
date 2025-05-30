from torch.optim import AdamW

from diffusion.helper import run_and_save_experiment
from diffusion.main import ExperimentManager, TrainingConfig
from diffusion.model import create_model


def run(manager: ExperimentManager):
    """
    Hyperparameter grid:
        • learning_rate  ∈ {1e-4, 5e-4, 1e-3}
        • num_epochs     ∈ {10, 25, 50}

    Uses manager-stored defaults for dataset / schedulers, but overrides
    config.learning_rate and config.num_epochs on each grid point.
    """

    learning_rates = [1e-4, 5e-4, 1e-3]
    epoch_counts = [10, 25, 50]

    for lr in learning_rates:
        for ep in epoch_counts:
            print(f"\n🔧 Running Experiment: LR={lr}, Epochs={ep}")

            # --- clone the base config and override knobs ---
            cfg = TrainingConfig(**vars(manager.config))  # shallow copy
            cfg.learning_rate = lr
            cfg.num_epochs = ep
            # Remove the output_dir setting since it will be overwritten
            # cfg.output_dir  = f"exp3/lr{lr}_ep{ep}"

            # --- fresh model instance ---
            model = create_model(cfg.image_size)
            # model = create_model(manager.config.image_size)

            # --- create a fresh optimizer for this experiment ---
            optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

            # --- hand off to shared utility ---
            run_and_save_experiment(
                manager,
                exp_name="hyperparams",
                test_name=f"lr{lr}_ep{ep}",
                model=model,
                optimizer=optimizer,  # Use our custom optimizer with the correct learning rate
                # Let other parameters default from manager
            )
