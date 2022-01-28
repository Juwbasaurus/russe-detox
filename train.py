import argparse
import json
import logging

import torch
from torch.utils.data import DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)
from transformers.optimization import AdamW

from src.data.datasets import GPT2Dataset
from src.data.samplers import BatchByLengthSampler
from src.data.collate import Collate
from src.training.trainers import Trainer3000
from utils.config import Config
from utils.logging import WandbLogger


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SEED = 42069
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

parser = argparse.ArgumentParser(description='Specify a config for training.')
parser.add_argument(
    'config',
    type=str,
    help="Path to config file for training.",
)
args = parser.parse_args()
config_name = args.config.split('/')[-1].split('.')[0]

with open(args.config, 'r') as f:
    config = Config(json.load(f))

if config.use_wandb:
    wandb_logger = WandbLogger(
        project=config.wandb_project_name,
        entity=None,
        config=config.to_dict(),
        name=config_name,
    )
else:
    wandb_logger = None

logging.info('Instantiating model and tokenizer...')
tokenizer = GPT2TokenizerFast.from_pretrained(
    config.model_name,
    pad_token='<pad>',
    bos_token='<s>',
    eos_token='<s>',
    unk_token='<unk>',
    mask_token='<mask>',
)
model = GPT2LMHeadModel.from_pretrained(config.model_name)
model.to(DEVICE)

logging.info('Preparing data...')
train_dataset = GPT2Dataset(config.train_data_path, tokenizer)
test_dataset = GPT2Dataset(config.test_data_path, tokenizer)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.train_batch_size,
    collate_fn=Collate(
        max_len=config.sample_max_len,
        device=DEVICE,
    ),
    sampler=BatchByLengthSampler(
        train_dataset.lengths,
        batch_size=config.train_batch_size,
        bucket_size=config.train_batch_size*10,
        seed=SEED,
    ),
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=config.test_batch_size,
    collate_fn=Collate(
        max_len=config.sample_max_len,
        device=DEVICE,
    ),
    sampler=BatchByLengthSampler(
        test_dataset.lengths,
        batch_size=config.test_batch_size,
        bucket_size=config.test_batch_size*10,
        seed=SEED,
    ),
)

optimizer = AdamW(
    params=model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)

trainer = Trainer3000(
    model,
    tokenizer,
    train_dataloader,
    test_dataloader,
    output_dir=f'models/{config_name}',
    overwrite_output_dir=True,
    save_every_n_steps=config.save_every_n_steps,
    logger=wandb_logger,
    optimizer=optimizer,
    lr_schedule=config.lr_schedule,
    max_epochs=config.max_epochs,
    learning_rate=config.learning_rate,
    warmup_ratio=config.warmup_ratio,
    gradient_accumulation_steps=config.gradient_accumulation_steps
)

trainer.run()
# trainer.save_checkpoint()
