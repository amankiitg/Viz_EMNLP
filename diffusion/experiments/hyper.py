learning_rates = [1e-4, 5e-4, 1e-3]
epoch_counts = [10, 25, 50]  # Note: still only loading from `epoch9`

for use_attention in [True, False]:
    model_type = "with_attention" if use_attention else "no_attention"

    for lr in learning_rates:
        for ep in epoch_counts:
            print(f"\n🔧 Running Experiment: {model_type}, LR={lr}, Epochs={ep}")

            config = TrainingConfig()
            config.learning_rate = lr
            config.num_epochs = ep
            config.output_dir = f"exp3/{model_type}_lr{lr}_ep{ep}"

            model = get_model(with_attention=use_attention)

            # Assume training was already done and model saved in epoch9
            run_and_save_experiment(config, model, dataloader, test_name=f"{model_type}_lr{lr}_ep{ep}")