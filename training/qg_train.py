import argparse
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

from dataset import QGDataset
from trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataloader_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rates", nargs="+", type=float, default=[2e-5])
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--pad_mask_id", type=int, default=-100)
    parser.add_argument("--qg_model", type=str, default="VietAI/vit5-base")
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="./t5-base-question-generator")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=4)
    parser.add_argument("--log_file", type=str, default="result/qg_training_log.csv")
    parser.add_argument("--train_file", type=str, default="datasets/train/qg_train.json")
    parser.add_argument("--valid_file", type=str, default="datasets/validation/qg_valid.json")
    return parser.parse_args()


def get_tokenizer(checkpoint: str) -> T5Tokenizer:
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
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
        json_file=args.train_file,
        max_length=args.max_length,
        pad_mask_id=args.pad_mask_id,
        tokenizer=tokenizer
    )

    valid_set = QGDataset(
        json_file=args.valid_file,
        max_length=args.max_length,
        pad_mask_id=args.pad_mask_id,
        tokenizer=tokenizer
    )
    
    for lr in args.learning_rates:
        print(f"Training with learning rate: {lr}")
        model = get_model(args.qg_model, args.device, tokenizer)
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
