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
from src.models.registry import get_model, get_weights_loader, save_weights
from src.utils.config import Model_Configs

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
    args = parser.parse_args()
    logger.info(f"[Arguments received]: {args}")

    model = get_model(args.model, Model_Configs[args.model])
    logger.info(f"[Instantiated Language Model]: {args.model}")

    tokenizer = model.get_tokenizer()
    logger.info("[Instantiated Model specific Tokenizer]")

    weights_loader = get_weights_loader(args.model)
    weights_loader(model)
    logger.info("[Loaded pretrained weights]")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    # Fetch the instruction dataset for fine-tuning
    file_path = "data/instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

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
    allowed_max_length = 128
    num_workers = 0
    batch_size = 8

    collator = InstructionCollator(
        pad_token_id=pad_token_id,
        ignore_index=ignore_index,
        allowed_max_length=allowed_max_length,
        device=device,
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

    # Training
    start_time = time.time()
    num_epochs = 2
    trainer_instance = ModelTrainer()
    train_losses, val_losses, tokens_seen = trainer_instance(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=InstructionDataset.format_input(val_data[0]),
        tokenizer=tokenizer,
    )
    execution_time_minutes = (time.time() - start_time) / 60
    logger.info(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # Save fine-tuned weights
    saved_path = save_weights(model, args.model)
    logger.info(f"[Checkpoint] Fine-tuned weights saved to {saved_path}")

    # Sample inference on test entries
    model.eval()
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
