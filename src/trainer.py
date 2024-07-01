import os
import time 
import logging

import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

class Trainer:

    def __init__(
            self,
            train_loader: DataLoader,
            test_loader: DataLoader,
            model: nn.Module,
            optimizer: Optimizer,
            loss_fn: nn.Module,
            n_epochs: int,
            save_location: str,
            training_log: str,
            early_stopping: bool = False,
            early_stopping_metric: str = "val_loss",
            min_or_max: str = "min",
            patience: int = 5,
            gradient_accumulation_steps: int = 1,
            scheduler: _LRScheduler = None,
            verbose: bool = True,
            additional_reporting: dict = None,
            device:str = None,
    ):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.early_stopping_metric = early_stopping_metric
        self.min_or_max = min_or_max
        self.patience = patience
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scheduler = scheduler
        self.save_location = save_location
        self.training_log = training_log
        self.verbose = verbose
        self.additional_reporting = additional_reporting
        if not device:
            self.device = (
                torch.device("cuda") if torch.cuda.is_available() 
                else torch.device("mps") if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = torch.device(device)
        self.patience_counter: int = 0
        self.best_metric: float = float("inf")
        self.best_metrics: dict[str, float] = None

    def _step(
            self,
            step_n: int,
            batch: dict,
    )->float:
        self.model.train()
        image = batch["image"].to(self.device)
        label = batch["label"].to(self.device)#.float()
        preds = self.model(image)
        loss = self.loss_fn(preds, label)
        loss.backward()
        if step_n % self.gradient_accumulation_steps == 0 and step_n > 1:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()
    
    def _epoch(
            self,
            n_epoch: int,
    ):
        self.model.to(self.device)
        self.model.train()
        total_loss = 0
        stop_training = False
        if self.verbose:
            pbar = tqdm(
                enumerate(self.train_loader), 
                total=len(self.train_loader), 
                desc="Training", 
                leave=False, 
                position=0
            )
        for step_n, batch in enumerate(self.train_loader):
            loss = self._step(step_n, batch)
            total_loss += loss
            if self.verbose:
                pbar.update(1)
        
        if self.scheduler:
            self.scheduler.step()

        val_loss, metrics = self._eval()
        train_loss_out = total_loss / len(self.train_loader)
    
        self._report(train_loss_out, val_loss, metrics)

        metrics["train_loss"] = train_loss_out
        metrics["val_loss"] = val_loss
        metrics["n_epoch"] = n_epoch

        if self.early_stopping:
            self._early_stopping(metrics)
            if self.patience_counter >= self.patience:
                stop_training = True

        return stop_training, metrics
    
    def _eval_step(
            self,
            batch: dict,
    ):
        image = batch["image"].to(self.device)
        label = batch["label"].to(self.device)
        preds = self.model(image)
        loss = self.loss_fn(preds, label)
        pred_labels = self.model.output_activation(preds)
        return loss.item(), label, pred_labels

    def _eval(
            self,
    ):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        if self.verbose:
            pbar = tqdm(
                enumerate(self.test_loader), 
                total=len(self.test_loader), 
                desc="Validation", 
                leave=False, 
                position=1
            )
        with torch.no_grad():
            for batch in self.test_loader:
                loss, label, pred_labels = self._eval_step(batch)
                total_loss += loss
                all_preds.append(pred_labels.cpu())
                all_labels.append(label.cpu())
                if self.verbose:
                    pbar.update(1)

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = self._compute_metrics(all_preds, all_labels)
        return total_loss / len(self.test_loader), metrics

    def _early_stopping(
            self,
            metrics: dict[str, float]
    ):
        if self.min_or_max == "min":
            metric = metrics[self.early_stopping_metric]
            if metric < self.best_metric:
                self.best_metric = metric
                self.patience_counter = 0
                self._save_model()
                self.best_metrics = metrics
            else:
                self.patience_counter += 1
        else:
            metric = metrics[self.early_stopping_metric]
            if metric > self.best_metric:
                self.best_metric = metric
                self.patience_counter = 0
                self._save_model()
                self.best_metrics = metrics
            else:
                self.patience_counter += 1

    def _save_best_metrics(
            self,
    ):
        if self.best_metrics:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            if os.path.exists(self.training_log):
                log = pd.read_csv(self.training_log)
            else:
                log = pd.DataFrame()
            self.best_metrics["timestamp"] = now
            for key in self.additional_reporting:
                self.best_metrics[key] = self.additional_reporting[key]
            log = pd.concat(
                [
                    log, 
                    pd.DataFrame({k:[v] for k,v in self.best_metrics.items()})
                ], 
                axis=0
            ).reset_index(drop=True)
            log.to_csv(self.training_log, index=False)


    def _save_model(
            self,
    ):
        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)
        self.model.save_pretrained(self.save_location)
        

    def _compute_metrics(
            self,
            preds: torch.Tensor,
            labels: torch.Tensor,
    )->dict[str, float]:
        preds = torch.argmax(preds, dim=1)
        labels = labels.numpy()
        preds = preds.numpy()
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="macro", zero_division=0)
        recall = recall_score(labels, preds, average="macro", zero_division=0)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        return {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
    def _report(
            self,
            train_loss_out: float, 
            val_loss: float, 
            metrics: dict[str, float]
    ):
        if self.verbose:
            text = "\n"
            text += "="*100
            text += f"\nTrain Loss: {train_loss_out:.4f} | "
            text += f"Val Loss: {val_loss:.4f} | "
            text += f"Accuracy: {metrics['acc']:.4f} | "
            text += f"Precision: {metrics['precision']:.4f} | "
            text += f"Recall: {metrics['recall']:.4f} | "
            text += f"F1: {metrics['f1']:.4f}\n"
            text += "="*100
            logger.info(text)

    def train(
            self,
    ):
        self.model.to(self.device)
        self.best_metric = float("inf")
        self.patience_counter = 0
        self.best_metrics = None
        try:
            for epoch in range(self.n_epochs):
                logger.info(f"Epoch {epoch+1}/{self.n_epochs}")
                stop_training, _ = self._epoch(epoch+1)
                if stop_training:
                    break
        except KeyboardInterrupt:
            logger.info("Training stopped by user")
        finally:
            if self.best_metrics:
                self._save_best_metrics()