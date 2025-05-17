import os

import torch
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator
from diffusers import DDPMPipeline, DPMSolverMultistepScheduler
from tqdm.auto import tqdm
# Add gradient scaling and clipping
from torch.cuda.amp import GradScaler

def make_grid(images, rows, cols):
    # Helper function for making a grid of images
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline):
    # Sample from the model and save the images in a grid
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed),
        num_inference_steps=50
    ).images

    image_grid = make_grid(images, rows=4, cols=4)

    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        # mixed_precision="no",
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs")
    )

    inference_scheduler = DPMSolverMultistepScheduler()
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=inference_scheduler)
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=inference_scheduler)
                save_dir = os.path.join(config.output_dir, f"epoch{epoch}")
                pipeline.save_pretrained(save_dir)

# # =====================================================================
# # ❻ Training loop – checkpoints + metrics + plots
# # ---------------------------------------------------------------------

# def train_loop(cfg, model, noise_scheduler, optimizer, dataloader, lr_scheduler, start_epoch=0):
#     accelerator = Accelerator(mixed_precision=cfg.mixed_precision,
#                               gradient_accumulation_steps=cfg.gradient_accumulation_steps,
#                               log_with="tensorboard",
#                               project_dir=str(Path(cfg.output_dir) / "logs"))
#     if accelerator.is_main_process:
#         (Path(cfg.output_dir) / "samples").mkdir(parents=True, exist_ok=True)

#     # Add error handling and optimizer state restoration
#     if start_epoch > 0:
#         if cfg.resume_checkpoint:
#             try:
#                 pipe = DDPMPipeline.from_pretrained(cfg.resume_checkpoint)
#                 model.load_state_dict(pipe.unet.state_dict())
#                 noise_scheduler = pipe.scheduler

#                 # Optionally load optimizer state if saved
#                 optimizer_path = Path(cfg.resume_checkpoint) / "optimizer.pt"
#                 if optimizer_path.exists():
#                     optimizer.load_state_dict(torch.load(optimizer_path))

#                 print(f"🔄 Resumed from {cfg.resume_checkpoint} (epoch {start_epoch})")
#             except Exception as e:
#                 print(f"⚠️ Failed to load checkpoint: {e}")
#                 print("Starting from scratch instead.")
#                 start_epoch = 0
#         else:
#             print(f"⚠️ start_epoch > 0 but no resume_checkpoint provided. Starting from epoch 0.")
#             start_epoch = 0

#     model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)

#     loss_hist, kl_hist = [], []
#     global_step, total_start = start_epoch*len(dataloader), time.time()

#     for epoch in range(start_epoch, cfg.num_epochs):
#         epoch_loss = 0.0
#         epoch_kl_sum = 0.0
#         epoch_start = time.time()
#         pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", dynamic_ncols=True,
#                      disable=not accelerator.is_local_main_process)

#         for batch in pbar:
#             clean = batch; noise = torch.randn_like(clean)
#             t = torch.randint(0, noise_scheduler.num_train_timesteps, (clean.size(0),), device=clean.device).long()
#             noisy = noise_scheduler.add_noise(clean, noise, t)
#             with accelerator.accumulate(model):
#                 pred = model(noisy, t, return_dict=False)[0]
#                 loss = F.mse_loss(pred, noise)
#                 kl   = 0.5*loss
#                 accelerator.backward(loss)
#                 accelerator.clip_grad_norm_(model.parameters(), 1.0)
#                 optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()

#             # loss bookkeeping
#             epoch_loss += loss.item()
#             epoch_kl_sum += kl.item()
#             kl_hist.append((global_step, kl.item()))

#             pbar.set_postfix(loss=f"{loss.item():.4f}", kl=f"{kl.item():.4f}")
#             accelerator.log({
#                 "loss": loss.item(),
#                 "kl": kl.item(),
#                 "lr": lr_scheduler.get_last_lr()[0]  # Added learning rate logging
#             }, step=global_step)
#             global_step += 1

#         # ---- epoch end ---- (properly indented outside the batch loop)
#         if accelerator.is_main_process:
#             mean_loss = epoch_loss / len(dataloader)
#             mean_kl   = epoch_kl_sum / len(dataloader)

#             loss_hist.append((epoch + 1, mean_loss))
#             epoch_time = time.time() - epoch_start
#             print(f"⏱️ Epoch {epoch+1} finished in {epoch_time:.1f}s – mean loss {mean_loss:.4f} | mean KL {mean_kl:.4f}")

#             # 1) sample grid every N epochs
#             if (epoch + 1) % cfg.save_image_epochs == 0:
#                 try:
#                     pipe = DDPMPipeline(unet=accelerator.unwrap_model(model),
#                                         scheduler=inference_scheduler)
#                     evaluate(cfg, epoch + 1, pipe)
#                 except Exception as e:
#                     print(f"⚠️ Error generating samples: {e}")

#             # 2) checkpoint every N epochs
#             if (epoch + 1) % cfg.save_model_epochs == 0 or (epoch + 1) == cfg.num_epochs:
#                 try:
#                     pipe = DDPMPipeline(unet=accelerator.unwrap_model(model),
#                                         scheduler=inference_scheduler)
#                     ckpt_dir = Path(cfg.output_dir) / f"epoch{epoch+1}"
#                     pipe.save_pretrained(ckpt_dir)

#                     # Save optimizer state
#                     torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

#                     print(f"✅ Saved checkpoint → {ckpt_dir}")
#                 except Exception as e:
#                     print(f"⚠️ Error saving checkpoint: {e}")

#     # Training complete
#     if accelerator.is_main_process:
#         total_time = time.time() - total_start
#         print(f"✅ Training complete in {total_time/60:.1f} min")

#         # Generate plots if we have data
#         try:
#             if loss_hist:
#                 # Loss plot
#                 ep, l = zip(*loss_hist)
#                 plt.figure(figsize=(6, 4))
#                 plt.plot(ep, l, marker='o')
#                 plt.xlabel('Epoch')
#                 plt.ylabel('Loss')
#                 plt.title('Training Loss')
#                 plt.grid(True)
#                 plt.savefig(Path(cfg.output_dir) / 'loss_curve.png')
#                 plt.close()  # Close figure to free memory

#                 # KL plot
#                 if kl_hist:
#                     st, kl = zip(*kl_hist)
#                     plt.figure(figsize=(6, 4))
#                     plt.plot(st, kl)
#                     plt.xlabel('Global step')
#                     plt.ylabel('KL')
#                     plt.title('KL divergence')
#                     plt.grid(True)
#                     plt.savefig(Path(cfg.output_dir) / 'kl_curve.png')
#                     plt.close()  # Close figure to free memory
#         except Exception as e:
#             print(f"⚠️ Error generating plots: {e}")
