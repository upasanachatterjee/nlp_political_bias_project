
import torch, torch.nn as nn, torch.nn.functional as F
from accelerate import Accelerator
from dataset import build_dataloaders
from model import MultiTaskRoberta
from utils import TrainArgs, TaskSpec
from transformers import AutoTokenizer
import time 
from datetime import datetime, timedelta
from transformers.optimization import get_cosine_schedule_with_warmup


# --- Training prep ---
# RTX 5090 optimized Accelerator setup
accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=4,  # Optimized for RTX 5090 and 5M dataset
    project_dir="./logs"
)

args = TrainArgs()

# Multi-GPU information
if accelerator.is_main_process:
    print(f"🚀 Multi-GPU Training Setup:")
    print(f"   - Number of processes: {accelerator.num_processes}")
    print(f"   - Distributed type: {accelerator.distributed_type}")
    print(f"   - Mixed precision: {accelerator.mixed_precision}")
    print(f"   - Main process: {accelerator.is_main_process}")
    if torch.cuda.is_available():
        print(f"   - CUDA devices: {torch.cuda.device_count()}")
        print(f"   - Current device: {accelerator.device}")

print(f"✅ TrainArgs loaded: {args.num_epochs} epochs, {args.max_length} max_length, {args.model_name}")

# Check if themes file exists
import os
themes_path = "top_themes.txt"
theme_count = 0

with open(themes_path, 'r') as f:
        theme_count = len(f.readlines())


print(f"\n🤖 Initializing MultiTaskRoberta model with theme_path: {themes_path}")
model = MultiTaskRoberta(theme_path=themes_path)
print(f"✅ Model initialized: {model.__class__.__name__}")

effective_batch_size = args.batch_mlm * accelerator.num_processes * accelerator.gradient_accumulation_steps
base_lr = 5e-5  # Start with a more conservative base learning rate
adjusted_lr = base_lr * (effective_batch_size / 32)  # Scale based on effective batch size
optimizer = torch.optim.AdamW(model.parameters(), lr=adjusted_lr, weight_decay=0.01)
print("✅ Optimizer created")

print(f"\n📊 Building dataloaders for multi-task training...")
print("   - Loading tokenizer for roberta-base...")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
print("   ✅ Tokenizer loaded")

task_spec = TaskSpec(dataset_name="dragonslayer631/bignewsalign-with-gdelt", themes_path="top_themes.txt")

print("   - Building dataloaders (this may take a moment)...")
dataloaders = build_dataloaders(tok=tokenizer, task_spec=task_spec, args=args)
print("   ✅ Dataloaders built")

# Calculate dataset sizes and steps per epoch
dataset_sizes = {}
steps_per_epoch = {}

for task_name, dataloader in dataloaders.items():
    if dataloader is not None:
        dataset_sizes[task_name] = len(dataloader.dataset)
        steps_per_epoch[task_name] = len(dataloader)
        print(f"   📋 {task_name:12s}: {dataset_sizes[task_name]:,} samples, {steps_per_epoch[task_name]:,} steps per epoch")
    else:
        print(f"   ❌ {task_name:12s}: No dataloader created (skipped)")

# Calculate total training steps based on epochs and largest dataset
max_steps_per_epoch = max(steps_per_epoch.values()) if steps_per_epoch else 1000
TOTAL_STEPS = args.num_epochs * max_steps_per_epoch
print(f"\n🎯 Training configuration:")
print(f"   - Epochs: {args.num_epochs}")
print(f"   - Max steps per epoch: {max_steps_per_epoch:,}")
print(f"   - Total training steps: {TOTAL_STEPS:,}")

print(f"\n📅 Creating learning rate scheduler...")
print(f"   - Warmup ratio: {args.warmup_ratio}")
print(f"   - Warmup steps: {int(TOTAL_STEPS * args.warmup_ratio):,}")
# Add learning rate scheduler

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(TOTAL_STEPS * args.warmup_ratio),
    num_training_steps=TOTAL_STEPS,
    num_cycles=0.5  # This creates a gentle decay
)
print(f"\n🔄 Preparing objects with Accelerator...")
# Prepare dataloaders for distributed training as well
prepared_dataloaders = {}
for name, dl in dataloaders.items():
    if dl is not None:
        prepared_dataloaders[name] = accelerator.prepare(dl)
    else:
        prepared_dataloaders[name] = None

(model, optimizer, scheduler) = accelerator.prepare(model, optimizer, scheduler)
dataloaders = prepared_dataloaders  # Use prepared dataloaders

if accelerator.is_main_process:
    print("✅ Model, optimizer, scheduler, and dataloaders prepared with Accelerator")
    print(f"   - Model device: {next(model.parameters()).device}")
    print(f"   - Effective batch size per GPU: {args.batch_mlm}")
    print(f"   - Total effective batch size: {args.batch_mlm * accelerator.num_processes}")

# Initialize native logging configuration
if accelerator.is_main_process:
    print(f"\n📊 Training Configuration Summary:")
    print(f"   - Model: {args.model_name}")
    print(f"   - Epochs: {args.num_epochs}")
    print(f"   - Max length: {args.max_length}")
    print(f"   - Learning rate: 0.0005")
    print(f"   - Weight decay: 0.01")
    print(f"   - Mixed precision: fp16")
    print(f"   - Number of GPUs: {accelerator.num_processes}")
    print(f"   - Batch sizes per GPU - MLM: {args.batch_mlm}, Triplet: {args.batch_triplet}, Themes: {args.batch_multilabel}, Tone: {args.batch_reg}")
    print(f"   - Gradient accumulation steps: {accelerator.gradient_accumulation_steps}")
    print(f"   - Effective batch sizes - MLM: {args.batch_mlm * accelerator.num_processes * accelerator.gradient_accumulation_steps}")
    print(f"   - Dataset: {task_spec.dataset_name}")
    print(f"   - Total training steps: {TOTAL_STEPS:,}")
    
    # Create logs directory for native logging
    os.makedirs("./logs", exist_ok=True)
    log_file = "./logs/training_log.txt"
    with open(log_file, "w") as f:
        f.write("Training Log - RTX 5090 Multi-GPU Setup\n")
        f.write("="*50 + "\n")
        f.write(f"Start Time: {__import__('datetime').datetime.now()}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset: {task_spec.dataset_name}\n")
        f.write(f"Total Steps: {TOTAL_STEPS:,}\n")
        f.write(f"GPUs: {accelerator.num_processes}\n")
        f.write("="*50 + "\n\n")
    print(f"   ✅ Native logging initialized: {log_file}")

print(f"\n📊 Initializing loss functions...")
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
bce_loss     = nn.BCEWithLogitsLoss()
mse_loss     = nn.MSELoss()
print("✅ Loss functions initialized:")
print("   - Triplet loss (margin=1.0)")
print("   - BCE with logits loss (for themes)")
print("   - MSE loss (for tone regression)")

print(f"\n🔄 Initializing task iterators and training order...")
iters = {}
available_tasks = []

# Initialize iterators for available dataloaders
if dataloaders.get("triplet"):
    iters["triplet"] = iter(dataloaders["triplet"])
    available_tasks.append("triplet")
    print("   ✅ Triplet iterator initialized")

if dataloaders.get("mlm"):
    iters["mlm"] = iter(dataloaders["mlm"])
    available_tasks.append("mlm")
    print("   ✅ MLM iterator initialized")

if dataloaders.get("regression"):
    iters["tone"] = iter(dataloaders["regression"])
    available_tasks.append("tone")
    print("   ✅ Tone (regression) iterator initialized")

if dataloaders.get("multilabel"):
    iters["themes"] = iter(dataloaders["multilabel"])
    available_tasks.append("themes")
    print("   ✅ Themes (multilabel) iterator initialized")

# Build training order based on available tasks
order = []
if "triplet" in available_tasks:
    order.extend(["triplet", "triplet"])  # Triplet appears twice as in original
if "mlm" in available_tasks:
    order.append("mlm")
if "tone" in available_tasks:
    order.insert(-1 if order else 0, "tone")
if "themes" in available_tasks:
    order.insert(-1 if order else 0, "themes")


## manually setting order for testing

order = ["triplet", "triplet", "themes", "mlm"]

print(f"\n📋 Training configuration complete:")
print(f"   - Training order: {order}")
print(f"   - Available tasks: {available_tasks}")
print(f"   - Tasks per cycle: {len(order)}")

if not available_tasks:
    print("❌ ERROR: No tasks available for training!")
    exit(1)

step = 0
epoch = 0
steps_in_current_epoch = 0
model.train()

print(f"\n🎯 Setting model to training mode...")
print("✅ Model ready for training")

# Loss tracking
loss_accumulator = {
    "triplet": [],
    "mlm": [],
    "tone": [], 
    "themes": []
}

print(f"\n📊 Loss tracking initialized for {len(loss_accumulator)} tasks")

def get_next_batch(task_name, iterators, dataloaders):
    """Get next batch, reinitializing iterator if exhausted."""
    try:
        return next(iterators[task_name])
    except StopIteration:
        print(f"   🔄 Reinitializing {task_name} iterator (end of dataset reached)")
        # Map task names to dataloader names
        dataloader_mapping = {
            "triplet": "triplet",
            "mlm": "mlm", 
            "tone": "regression",
            "themes": "multilabel"
        }
        dl_name = dataloader_mapping.get(task_name, task_name)
        iterators[task_name] = iter(dataloaders[dl_name])
        return next(iterators[task_name])

# Add timing variables for ETC calculation
training_start_time = time.time()
step_times = []  # Store recent step times for averaging
eta_window_size = 50  # Number of recent steps to average for ETC

print(f"\n" + "="*80)
print(f"🚀 STARTING TRAINING: {args.num_epochs} EPOCHS")
print(f"="*80)

while epoch < args.num_epochs:
    epoch_start_step = step
    epoch_start_time = time.time()  # Track epoch start time
    print(f"\n📅 EPOCH {epoch + 1}/{args.num_epochs}")
    print(f"   Target steps this epoch: {max_steps_per_epoch:,}")
    print("-" * 50)
    
    while steps_in_current_epoch < max_steps_per_epoch and step < TOTAL_STEPS:
        step_start_time = time.time()  # Track individual step time
        task = order[step % len(order)]
        optimizer.zero_grad(set_to_none=True)

        loss = None  # Initialize loss variable

        if task == "triplet":
            b = get_next_batch("triplet", iters, dataloaders)
            if b is None or b.get("_skip", False):
                # Skip this step if no valid triplet batch could be formed - use dummy loss
                print(f"   ⚠️  Skipping triplet step {step} - no valid triplets found, using dummy loss")
                loss = torch.tensor(0.0, requires_grad=True, device=accelerator.device)
            else:
                za,_ = model.forward_embed(b["a_ids"], b["a_mask"])
                zp,_ = model.forward_embed(b["p_ids"], b["p_mask"])
                zn,_ = model.forward_embed(b["n_ids"], b["n_mask"])
                loss = triplet_loss(za, zp, zn)
                if loss is not None:
                    loss_accumulator["triplet"].append(loss.item())

        elif task == "themes":
            b = get_next_batch("themes", iters, dataloaders)
            if b is None or b.get("_skip", False):
                # Skip this step if no valid themes batch - use dummy loss
                print(f"   ⚠️  Skipping themes step {step} - no valid samples, using dummy loss")
                loss = torch.tensor(0.0, requires_grad=True, device=accelerator.device)
            else:
                z, pooled = model.forward_embed(b["input_ids"], b["attention_mask"])
                logits = model.theme_head(pooled)
                loss = bce_loss(logits, b["labels"].float())
                if loss is not None:
                    loss_accumulator["themes"].append(loss.item())

        elif task == "tone":
            b = get_next_batch("tone", iters, dataloaders)
            if b is None or b.get("_skip", False):
                # Skip this step if no valid tone batch - use dummy loss
                print(f"   ⚠️  Skipping tone step {step} - no valid samples, using dummy loss")
                loss = torch.tensor(0.0, requires_grad=True, device=accelerator.device)
            else:
                z, pooled = model.forward_embed(b["input_ids"], b["attention_mask"])
                pred = model.tone_head(pooled)
                loss = mse_loss(pred, b["targets"].float())
                if loss is not None:
                    loss_accumulator["tone"].append(loss.item())

        elif task == "mlm":
            b = get_next_batch("mlm", iters, dataloaders)
            out = model.forward_mlm(**b)
            loss = out.loss
            if loss is not None:
                loss_accumulator["mlm"].append(loss.item())

        # Only proceed with backpropagation if loss is not None
        if loss is not None:
            # Use gradient accumulation context
            with accelerator.accumulate(model):
                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()  # Update learning rate
        else:
            print(f"   ⚠️  Warning: Loss is None at step {step} for task {task}, skipping backward pass")

        if accelerator.is_main_process and step % args.log_every == 0:
            if loss is not None:
                # Calculate timing statistics
                current_time = time.time()
                elapsed_time = current_time - training_start_time
                elapsed_str = str(timedelta(seconds=int(elapsed_time)))
                
                # Calculate ETC based on recent step times
                if len(step_times) >= 5:  # Need at least 5 steps for reliable estimate
                    avg_step_time = sum(step_times) / len(step_times)
                    remaining_steps = TOTAL_STEPS - step
                    eta_seconds = remaining_steps * avg_step_time
                    eta_str = str(timedelta(seconds=int(eta_seconds)))
                    completion_time = datetime.now() + timedelta(seconds=eta_seconds)
                    completion_str = completion_time.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    eta_str = "Calculating..."
                    completion_str = "Calculating..."
                
                # Calculate epoch ETC
                epoch_elapsed = current_time - epoch_start_time
                epoch_progress = steps_in_current_epoch / max_steps_per_epoch
                if epoch_progress > 0.01:  # Avoid division by very small numbers
                    epoch_eta_seconds = (epoch_elapsed / epoch_progress) - epoch_elapsed
                    epoch_eta_str = str(timedelta(seconds=int(epoch_eta_seconds)))
                else:
                    epoch_eta_str = "Calculating..."

                    
                # Calculate average losses for each task
                current_loss = float(loss.item())
                current_lr = scheduler.get_last_lr()[0]

                # Calculate steps per second
                if len(step_times) >= 5:
                    steps_per_second = 1.0 / avg_step_time
                    samples_per_second = steps_per_second * args.batch_mlm * accelerator.num_processes
                else:
                    steps_per_second = 0
                    samples_per_second = 0

                progress_pct = (step / TOTAL_STEPS) * 100
                epoch_progress = steps_in_current_epoch / max_steps_per_epoch
                epoch_progress_pct = epoch_progress * 100
                
                # Prepare logging data
                log_entry = f"Step {step:5d} | Task: {task:8s} | Loss: {current_loss:.4f} | LR: {current_lr:.2e} | Epoch: {epoch + 1} | Progress: {epoch_progress*100:.1f}%"
                
                # Add task-specific losses if they have data
                task_losses = {}
                for task_name in ["triplet", "mlm", "tone", "themes"]:
                    if loss_accumulator[task_name]:
                        avg_loss = sum(loss_accumulator[task_name]) / len(loss_accumulator[task_name])
                        task_losses[task_name] = avg_loss
                        loss_accumulator[task_name] = []  # Reset accumulator
                
                # Log to console
                # Log to console with rich timing info
                print(f"  {log_entry}")
                print(f"    ⏱️  Elapsed: {elapsed_str} | ETC: {eta_str} | Completion: {completion_str}")
                print(f"    📊 Speed: {steps_per_second:.2f} steps/s | {samples_per_second:.0f} samples/s")
                print(f"    🔄 Epoch {epoch + 1}: {epoch_progress_pct:.1f}% | Epoch ETC: {epoch_eta_str}")
                if task_losses:
                    task_loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in task_losses.items()])
                    print(f"    Avg Task Losses: {task_loss_str}")
                
                # Log to file
                with open("./logs/training_log.txt", "a") as f:
                    f.write(f"{log_entry}\n")
                    f.write(f"  Timing: Elapsed={elapsed_str}, ETC={eta_str}, Completion={completion_str}\n")
                    f.write(f"  Speed: {steps_per_second:.2f} steps/s, {samples_per_second:.0f} samples/s\n")
                    f.write(f"  Epoch: {epoch + 1} ({epoch_progress_pct:.1f}%), Epoch ETC: {epoch_eta_str}\n")
                    if task_losses:
                        f.write(f"  Task Averages: {task_losses}\n")
                    f.write("\n")
            else:
                print(f"Warning: Loss is None at step {step} for task {task}")

        step += 1
        steps_in_current_epoch += 1
    
    # End of epoch
    epoch_steps = step - epoch_start_step
    print(f"\n✅ COMPLETED EPOCH {epoch + 1}")
    print(f"   Steps completed: {epoch_steps:,}")
    print(f"   Total steps so far: {step:,}")
    
    # Save checkpoint at end of epoch
    if accelerator.is_main_process:
        epoch_checkpoint_path = f"{args.output_dir}/epoch-{epoch + 1}"
        print(f"💾 Saving epoch checkpoint to {epoch_checkpoint_path}...")
        model.save_checkpoint(f'{epoch_checkpoint_path}.pt')
        print(f"✅ Epoch checkpoint saved")
        
    
    # Reset epoch counter and reinitialize all iterators for next epoch
    steps_in_current_epoch = 0
    epoch += 1

    if epoch < args.num_epochs:  # Don't reinitialize on the last epoch
        print(f"🔄 Preparing for next epoch...")
        # Reinitialize iterators for next epoch
        for task_name in iters.keys():
            dataloader_mapping = {
                "triplet": "triplet",
                "mlm": "mlm", 
                "tone": "regression",
                "themes": "multilabel"
            }
            dl_name = dataloader_mapping.get(task_name, task_name)
            if dl_name and dataloaders.get(dl_name):
                iters[task_name] = iter(dataloaders[dl_name])
        print(f"✅ All iterators reinitialized for epoch {epoch + 1}")
    
    print("="*50)

# Clean shutdown
print(f"\n🏁 Training completed! Cleaning up...")
accelerator.end_training()
print("✅ Accelerator shutdown complete")

# Save final model
if accelerator.is_main_process:
    final_model_path = f"{args.output_dir}/final_model"
    print(f"💾 Saving final model to {final_model_path}...")
    model.save_checkpoint(f'{final_model_path}.pt')
    print(f"✅ Final model saved")

print("\n" + "="*80)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print(f"📊 FINAL STATISTICS:")
print(f"   - Total epochs completed: {args.num_epochs}")
print(f"   - Total steps executed: {step:,}")
print(f"   - Max steps per epoch: {max_steps_per_epoch:,}")
print(f"   - Tasks trained: {', '.join(available_tasks)}")
print(f"   - Final model saved to: {args.output_dir}/final_model")
print("="*80)