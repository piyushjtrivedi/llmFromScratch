import torch
import time
import os
import urllib.request
import ssl
import json
import urllib
import logging
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from src.models.registry import (get_model, get_weights_loader, save_weights,
                                  save_metrics, save_lora_config)

from src.utils.config import Model_Configs
from src.utils.lora import apply_lora, _DEFAULT_TARGET_MODULES

from src.data.instruction_dataset import InstructionDataset
from src.data.instruction_collator import InstructionCollator

from src.training.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():

    torch.manual_seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small (124M)", choices=list(Model_Configs.keys()))
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override the model config batch size")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Override the model config max sequence length")
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="Override the model config gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0004,
                        help="Peak learning rate (default: 0.0004)")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Number of linear LR warmup steps (default: 100)")
    parser.add_argument("--lora", action="store_true",
                        help="Apply LoRA adapters and train only the adapter weights")
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="LoRA rank r (default: 8)")
    parser.add_argument("--lora-alpha", type=float, default=16.0,
                        help="LoRA alpha scaling factor (default: 16.0)")
    parser.add_argument("--grad-clip", type=float, default=0.5,
                        help="Max gradient L2 norm for clipping (default: 0.5)")
    parser.add_argument("--eval-freq", type=int, default=20,
                        help="Evaluate every N optimizer steps (default: 20). "
                             "Low values (e.g. 5) make eval dominate runtime.")
    parser.add_argument("--eval-iter", type=int, default=30,
                        help="Number of batches used per evaluation (default: 30)")
    args = parser.parse_args()
    logger.info(f"[Arguments received]: {args}")

    model = get_model(args.model, Model_Configs[args.model])
    logger.info(f"[Instantiated Language Model]: {args.model}")

    tokenizer = model.get_tokenizer()
    logger.info("[Instantiated Model specific Tokenizer]")

    weights_loader = get_weights_loader(args.model)
    weights_loader(model)
    logger.info("[Loaded pretrained weights]")

    # Optionally apply LoRA: freeze the base weights and add small trainable
    # A/B adapter matrices to the target attention projections. When --lora is
    # not passed the model trains all parameters as normal.
    if args.lora:
        apply_lora(model, r=args.lora_rank, alpha=args.lora_alpha)
        save_lora_config(args.model, r=args.lora_rank, alpha=args.lora_alpha,
                         target_modules=list(_DEFAULT_TARGET_MODULES))

    # AdamW with transformer-standard hyperparameters.
    # beta2=0.95 (vs PyTorch default 0.999): the second moment estimate decays faster,
    # so stale gradient history from early training steps has less influence on later
    # updates. This is the value used in GPT-3, Chinchilla, and PaLM and works better
    # than the default for both fine-tuning GPT-2 and training Gemma3 from scratch.
    # eps=1e-8 is fine for fp32; 1e-8 is PyTorch default so no change needed here
    # unless switching to bf16/fp16 where 1e-9 adds numerical stability.
    # When LoRA is active, only the adapter matrices have requires_grad=True.
    # Passing only those to the optimizer avoids AdamW allocating moment buffers
    # for the frozen base weights, which would roughly double memory usage.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95)
    )

    # Fetch the instruction dataset for fine-tuning
    file_path = "data/instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    ssl_context = ssl.create_default_context()

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url, context=ssl_context) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        logger.info(f"Data length: {len(data)}")
        logger.info(f"Example entry: {data[50]}")

    # Split into train / test / validation
    train_ratio = 0.85
    test_ratio = 0.1
    train_portion = int(len(data) * train_ratio)
    test_portion = int(len(data) * test_ratio)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    logger.info(f"[Train-Validate Split] Train={train_portion}, Test={test_portion}, Val={val_portion}")

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    pad_token_id = tokenizer.eot_token if hasattr(tokenizer, "eot_token") else tokenizer.eos_token_id
    ignore_index = -100
    cfg = Model_Configs[args.model]
    batch_size = args.batch_size or cfg["batch_size"]
    allowed_max_length = args.max_length or cfg["allowed_max_length"]
    gradient_accumulation_steps = args.grad_accum or cfg.get("gradient_accumulation_steps", 1)
    num_workers = 0
    logger.info(
        f"[Training config] batch_size={batch_size}, max_length={allowed_max_length}, "
        f"grad_accum={gradient_accumulation_steps} "
        f"(effective_batch={batch_size * gradient_accumulation_steps})"
    )

    collator = InstructionCollator(
        pad_token_id=pad_token_id,
        ignore_index=ignore_index,
        allowed_max_length=allowed_max_length,
    )

    train_loader = DataLoader(
        InstructionDataset(train_data, tokenizer),
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        InstructionDataset(val_data, tokenizer),
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        InstructionDataset(test_data, tokenizer),
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    logger.info(
        f"[DataLoaders] Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}"
    )

    model.to(device)

    # LR scheduler: linear warmup for `warmup_steps` optimizer steps, then cosine
    # decay down to 10% of peak LR over the remaining steps.
    # - Warmup: prevents large gradient updates at the start when weights are either
    #   random (Gemma3) or just loaded and not yet adapted (GPT-2 fine-tuning).
    # - Cosine decay: smoothly reduces LR as training converges, avoiding oscillation
    #   around the minimum that a fixed LR can cause.
    # total_steps = optimizer updates per epoch × epochs (each epoch has
    # len(train_loader) / gradient_accumulation_steps optimizer steps).
    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(args.warmup_steps, total_steps)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            # Phase 1: ramp LR from ~0 up to args.lr over warmup_steps
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
            # Phase 2: cosine decay from args.lr down to 10% of args.lr
            CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=args.lr * 0.1),
        ],
        milestones=[warmup_steps],
    )
    logger.info(
        f"[Scheduler] warmup={warmup_steps} steps, cosine decay over {total_steps - warmup_steps} steps"
    )

    # Training
    start_time = time.time()
    num_epochs = args.epochs
    trainer_instance = ModelTrainer()
    train_losses, val_losses, tokens_seen = trainer_instance(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=args.eval_freq, eval_iter=args.eval_iter,
        start_context=InstructionDataset.format_input(val_data[0]),
        tokenizer=tokenizer,
        gradient_accumulation_steps=gradient_accumulation_steps,
        scheduler=scheduler,
        model_name=args.model,
        lora=args.lora,
        grad_clip=args.grad_clip,
    )
    execution_time_minutes = (time.time() - start_time) / 60
    logger.info(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # Save fine-tuned weights
    saved_path = save_weights(model, args.model, lora=args.lora)
    logger.info(f"[Checkpoint] Fine-tuned weights saved to {saved_path}")

    # Save training metrics for loss curve visualisation
    save_metrics(
        {
            "model_name": args.model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "tokens_seen": tokens_seen,
            "learning_rates": trainer_instance.learning_rates,
            "grad_norms": trainer_instance.grad_norms,
            "peak_memory_gb": trainer_instance.peak_memory_gb,
            "num_epochs": num_epochs,
            "eval_freq": args.eval_freq,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": args.lr,
            "lora_rank":  args.lora_rank  if args.lora else None,
            "lora_alpha": args.lora_alpha if args.lora else None,
            "execution_time_minutes": round(execution_time_minutes, 2),
            "tokens_per_sec": round(trainer_instance.tokens_seen / (execution_time_minutes * 60), 1),
            "grad_clip_norm": args.grad_clip,
            "step_times_sec": trainer_instance.step_times_sec,
            "gpu_memory_total_gb": (
                torch.cuda.get_device_properties(device).total_memory / 1e9
                if device.type == "cuda" else None
            ),
        },
        args.model,
        lora=args.lora,
    )

    # Sample inference on test entries
    model.eval()
    with torch.no_grad():
        for entry in test_data[:3]:
            input_text = InstructionDataset.format_input(entry)
            token_ids = ModelTrainer.generate_text_simple(
                model=model,
                idx=ModelTrainer.text_to_token_ids(input_text, tokenizer).to(device),
                max_new_tokens=256,
                context_size=Model_Configs[args.model]["context_length"],
                eos_id=pad_token_id,
            )
            generated_text = ModelTrainer.token_ids_to_text(token_ids, tokenizer)
            response_text = (
                generated_text[len(input_text):]
                .replace("### Response:", "")
                .strip()
            )
            logger.info(input_text)
            logger.info(f"\nCorrect response:\n>> {entry['output']}")
            logger.info(f"\nModel response:\n>> {response_text}")
            logger.info("-------------------------------------")


if __name__ == "__main__":
    main()
