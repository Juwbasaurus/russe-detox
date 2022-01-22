import os

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    PreTrainedTokenizerFast,
)

from utils.data_utils import preprocess_source, preprocess_target


class GPT2Dataset(Dataset):

    def __init__(self,
                 source_path: str,
                 target_path: str,
                 tokenizer: PreTrainedTokenizerFast):
        self.source_path = source_path
        self.target_path = target_path
        self.tokenizer = tokenizer
        self.encodings = []
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        sep = ' === '

        with open(source_path, 'r') as source:
            with open(target_path, 'r') as target:
                for src, tgt in tqdm(zip(source, target)):
                    src, tgt = preprocess_source(src), preprocess_target(tgt)
                    src_ids = self.tokenizer(src, add_special_tokens=False)
                    tgt_ids = self.tokenizer(tgt, add_special_tokens=False)
                    input_ids = bos_id + src_ids +  + tgt_ids + eos_id
                    encoding = {
                        'input_ids': input_encoding.input_ids,
                        'attention_mask': input_encoding.attention_mask,
                        # 'labels': labels_encoding.input_ids,
                        # 'decoder_attention_mask': labels_encoding.attention_mask
                    }
                    self.encodings.append(encoding)


    def __len__(self):
        pass

    def __getitem__(self, idx):
        return self.