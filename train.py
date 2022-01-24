import argparse
import json

import torch
from torch.utils.data import DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

from src.datasets import GPT2Dataset
from src.samplers import BatchByLengthSampler
from src.collate import Collate
from utils.config import Config


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

with open(args.config, 'r') as f:
    config = Config(json.load(f))

tokenizer = GPT2TokenizerFast.from_pretrained(config.model_name)
model = GPT2LMHeadModel.from_pretrained(config.model_name)

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
    shuffle=True
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
    shuffle=True,
)


