import argparse
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

from dataset import QGDataset
from trainer import Trainer
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataloader_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--pad_mask_id", type=int, default=-100)
    parser.add_argument("--qg_model", type=str, default="./t5-base-question-generator")
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--log_file", type=str, default="result/test_qg_log.csv")
    return parser.parse_args()


def get_tokenizer(checkpoint: str) -> T5Tokenizer:
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<answer>', '<context>']}
    )
    return tokenizer


def get_model(checkpoint: str, device: str, tokenizer: T5Tokenizer) -> T5ForConditionalGeneration:
    config = T5Config(decoder_start_token_id=tokenizer.pad_token_id)
    model = T5ForConditionalGeneration(config).from_pretrained(checkpoint)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    return model


if __name__ == "__main__":
    args = parse_args()
    tokenizer = get_tokenizer(args.qg_model)

    train_set = QGDataset(
        csv_file='datasets/train/qg_train.csv',
        max_length=args.max_length,
        pad_mask_id=args.pad_mask_id,
        tokenizer=tokenizer
    )

    test_set = QGDataset(
        csv_file='datasets/test/qg_test.csv',
        pad_mask_id=args.pad_mask_id,
        max_length=args.max_length,
        tokenizer=tokenizer
    )
    
    model = get_model(args.qg_model, args.device, tokenizer)
    trainer = Trainer(
        dataloader_workers=args.dataloader_workers,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model=model,
        tokenizer=tokenizer,
        pin_memory=args.pin_memory,
        save_dir="",
        train_batch_size=args.train_batch_size,
        train_set=train_set,
        valid_batch_size=args.test_batch_size,
        valid_set=test_set,
        log_file=args.log_file,
        evaluate_on_accuracy=True
    )
    
    trainer.evaluate(trainer.valid_loader)
    trainer.qg_accuracy(trainer.valid_loader)
