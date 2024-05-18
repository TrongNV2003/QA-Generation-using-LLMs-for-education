import torch
import numpy as np
import random
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer
import csv
from utils import AverageMeter

class Logger:
    def __init__(self, file_path: str, fieldnames: list):
        self.file_path = file_path
        self.fieldnames = fieldnames
        with open(self.file_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, data: dict):
        rounded_data = {key: round(value, 3) if isinstance(value, float) else value for key, value in data.items()}
        with open(self.file_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(rounded_data)

class Trainer:
    def __init__(
        self,
        dataloader_workers: int,
        device: str,
        epochs: int,
        learning_rate: float,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        pin_memory: bool,
        save_dir: str,
        train_batch_size: int,
        train_set: Dataset,
        valid_batch_size: int,
        log_file: str,
        valid_set: Dataset,
        evaluate_on_accuracy: bool = False
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        
        # Define the fieldnames for the CSV file
        self.fieldnames = ['epoch', 'train_loss', 'valid_loss', 'valid_accuracy']

        # Create a Logger instance
        self.logger = Logger(file_path=log_file, fieldnames=self.fieldnames)

        self.train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=True
        )
        self.valid_loader = DataLoader(
            valid_set,
            batch_size=train_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=False
        )
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.train_loss = AverageMeter()
        self.evaluate_on_accuracy = evaluate_on_accuracy
        if evaluate_on_accuracy:
            self.best_valid_score = 0
        else:
            self.best_valid_score = float("inf")

    def train(self) -> None:
        for epoch in range(1, self.epochs+1):
            self.model.train()
            self.train_loss.reset()

            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")
                for data in self.train_loader:
                    self.optimizer.zero_grad()
                    data = {key: value.to(self.device) for key, value in data.items()}
                    output = self.model(**data)
                    loss = output.loss
                    loss.backward()
                    self.optimizer.step()
                    self.train_loss.update(loss.item(), self.train_batch_size)
                    tepoch.set_postfix({"train_loss": self.train_loss.avg})
                    tepoch.update(1)

            if self.evaluate_on_accuracy:
                valid_accuracy = self.evaluate_accuracy(self.valid_loader)
                if valid_accuracy > self.best_valid_score:
                    print(
                        f"Validation accuracy improved from {self.best_valid_score:.4f} to {valid_accuracy:.4f}. Saving."
                    )
                    self.best_valid_score = valid_accuracy
                    self._save()
                valid_loss = self.evaluate(self.valid_loader)
                if valid_loss < self.best_valid_score:
                    print(
                        f"Validation loss decreased from {self.best_valid_score:.4f} to {valid_loss:.4f}. Saving.")
                    self.best_valid_score = valid_loss
                    self._save()
                self.logger.log({'epoch': epoch, 'train_loss': self.train_loss.avg,
                                    'valid_loss': valid_loss, 'valid_accuracy': valid_accuracy})
                
            else:
                valid_loss = self.evaluate(self.valid_loader)
                if valid_loss < self.best_valid_score:
                    print(
                        f"Validation loss decreased from {self.best_valid_score:.4f} to {valid_loss:.4f}. Saving.")
                    self.best_valid_score = valid_loss
                    self._save()
                self.logger.log({'epoch': epoch, 'train_loss': self.train_loss.avg,
                                    'valid_loss': valid_loss, 'valid_accuracy': None})

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        eval_loss = AverageMeter()
        with tqdm(total=len(dataloader), unit="batches") as tepoch:
            tepoch.set_description("validation")
            for data in dataloader:
                data = {key: value.to(self.device) for key, value in data.items()}
                output = self.model(**data)
                loss = output.loss
                eval_loss.update(loss.item(), self.valid_batch_size)
                tepoch.set_postfix({"valid_loss": eval_loss.avg})
                tepoch.update(1)
        return eval_loss.avg

    @torch.no_grad()
    def evaluate_accuracy(self, dataloader: DataLoader) -> float:
        self.model.eval()
        accuracy = AverageMeter()
        with tqdm(total=len(dataloader), unit="batches") as tepoch:
            tepoch.set_description("validation")
            for data in dataloader:
                data = {key: value.to(self.device) for key, value in data.items()}
                output = self.model(**data)
                preds = torch.argmax(output.logits, dim=1)
                score = accuracy_score(data["labels"].cpu(), preds.cpu())
                accuracy.update(score, self.valid_batch_size)
                tepoch.set_postfix({"valid_acc": accuracy.avg})
                tepoch.update(1)
        return accuracy.avg

    @torch.no_grad()
    def qg_accuracy(self, dataloader: DataLoader) -> float:
        self.model.eval()
        accuracies = []
        with tqdm(total=len(dataloader), unit="batches") as tepoch:
            tepoch.set_description("validation")
            for data in dataloader:
                data = {key: value.to(self.device) for key, value in data.items()}
                output = self.model(**data)
                preds = torch.argmax(output.logits, dim=-1)
                labels = data["labels"].to(preds.device)

                # Đảm bảo rằng preds và labels có cùng kích thước
                assert preds.shape == labels.shape, f"Shape mismatch: preds {preds.shape}, labels {labels.shape}"

                # Tính accuracy cho mỗi sample
                for i in range(labels.shape[1]):  # assuming the second dimension is the number of outputs
                    score = accuracy_score(labels[:, i].cpu(), preds[:, i].cpu())
                    accuracies.append(score)
                
                avg_accuracy = np.mean(accuracies)
                tepoch.set_postfix({"valid_acc": (avg_accuracy)*10})
                tepoch.update(1)

        overall_accuracy = np.mean(accuracies)
        return overall_accuracy

    def _save(self) -> None:
        self.tokenizer.save_pretrained(self.save_dir)
        self.model.save_pretrained(self.save_dir)
