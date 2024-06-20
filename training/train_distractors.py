import argparse
import random
import numpy as np
import torch
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from data_loader import DistractorDataset
from trainer import Trainer

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataloader_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning_rates", nargs="+", type=float, default=[2e-5, 1e-5])
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--pad_mask_id", type=int, default=-100)
    parser.add_argument("--distractor_model", type=str, default="VietAI/vit5-base")
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="./t5-base-distractor-generator")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=4)
    parser.add_argument("--log_file", type=str, default="result/qg_training_distractor_log.csv")
    parser.add_argument("--train_file", type=str, default="datasets/train/qg_train.json")
    parser.add_argument("--valid_file", type=str, default="datasets/validation/qg_valid.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_tokenizer(checkpoint: str) -> T5Tokenizer:
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<sep>']}
    )
    return tokenizer


def get_model(checkpoint: str, device: str, tokenizer: T5Tokenizer) -> T5ForConditionalGeneration:
    # config = T5Config(decoder_start_token_id=tokenizer.pad_token_id)
    # model = T5ForConditionalGeneration(config).from_pretrained(checkpoint)
    
    config = T5Config.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint, config=config)

    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    return model


if __name__ == "__main__":
    args = parse_args()
    
    # Set the seed for reproducibility
    set_seed(args.seed)
    
    tokenizer = get_tokenizer(args.distractor_model)
    
    train_set = DistractorDataset(
        json_file=args.train_file,
        max_length=args.max_length,
        pad_mask_id=args.pad_mask_id,
        tokenizer=tokenizer
    )

    valid_set = DistractorDataset(
        json_file=args.valid_file,
        max_length=args.max_length,
        pad_mask_id=args.pad_mask_id,
        tokenizer=tokenizer
    )
    
    for lr in args.learning_rates:
        print(f"Training with learning rate: {lr}")
        model = get_model(args.distractor_model, args.device, tokenizer)
        trainer = Trainer(
            dataloader_workers=args.dataloader_workers,
            device=args.device,
            epochs=args.epochs,
            learning_rate=lr,
            model=model,
            pin_memory=args.pin_memory,
            save_dir=args.save_dir,
            tokenizer=tokenizer,
            train_batch_size=args.train_batch_size,
            train_set=train_set,
            valid_batch_size=args.valid_batch_size,
            valid_set=valid_set,
            log_file=args.log_file
        )
        trainer.train()
