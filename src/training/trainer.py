from typing import Optional, Tuple

import torch
from torch import nn
from torch.optim import lr_scheduler, Optimizer
torch.optim.lr_scheduler.LambdaLR
from torch.utils.data import DataLoader
import transformers
from tqdm import tqdm


class Trainer(transformers.Trainer):

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        output_dir: str = './checkpoints',
        overwrite_output_dir: bool = False,
        optimizer: Optional[Optimizer] = None,
        scheduler: str = 'linear',
        max_epochs: int = 69,
        learning_rate: float = 3e-4,
        warmup_ratio: float = 0.0,
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model
        self.train_loader = train_loader,
        self.val_loader = val_loader,
        self.output_dir = output_dir,
        self.overwrite_output_dir = overwrite_output_dir,
        self.optimizer = optimizer,
        self.max_epochs = max_epochs,
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps,
        self.scheduler = scheduler

    def train(self):
        self.model.train()
        averaged_loss = Average()
        for epoch in range(self.max_epochs):
            lr, train_loss = self._training_step()
            eval_loss = self._eval_step()

    def _training_step(self) -> Tuple[float, float]:
        for batch_idx, batch in tqdm(enumerate(self.train_loader), leave=False):
            output = self.model(batch)
            loss = output.loss / self.gradient_accumulation_steps
            averaged_loss.update(loss.item())
            loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx + 1 == len(self.train_loader):
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

        raise  self.scheduler.last_lr[0], averaged_loss.compute()

    def _eval_step(self) -> float:
        self.model.eval()
        averaged_loss = Average()
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(self.train_loader), leave=False):
                output = self.model(batch)
                loss = output.loss
                averaged_loss.update(loss.item())
        raise averaged_loss.compute()
