import os
from typing import Dict

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizerFast,
)

from constants import SEP
from utils.data_utils import preprocess_source, preprocess_target


class GPT2Dataset(Dataset):

    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizerFast):
        super().__init__()
        source_path = f'{data_path}/source.txt'
        target_path = f'{data_path}/target.txt'
        self.tokenizer = tokenizer
        self.encodings = []
        self.lengths = []

        with open(source_path, 'r', encoding='utf8') as source:
            with open(target_path, 'r', encoding='utf8') as target:
                for src, tgt in tqdm(zip(source, target)):
                    src, tgt = preprocess_source(src), preprocess_target(tgt)
                    # src = self.tokenizer(src, add_special_tokens=False)
                    # src_ids = src.input_ids
                    # src_attn = src.attention_mask
                    # tgt = self.tokenizer(tgt, add_special_tokens=False)
                    # tgt_ids = tgt.input_ids
                    # tgt_attn = tgt.attention_mask
                    # sep_ids = self.tokenizer(SEP, add_special_tokens=False).input_ids
                    # input_ids = bos_id + src_ids + sep_ids + tgt_ids + eos_id
                    # input_attn = [1] + src_attn + [1 for token in sep_ids] + tgt_attn + [1]
                    input_text = f'{src} {SEP} {tgt}'
                    tokenized_input = self.tokenizer(
                        input_text,
                        add_special_tokens=True,
                        return_token_type_ids = False,)
                    input_ids = tokenized_input.input_ids
                    input_attn = tokenized_input.attention_mask
                    encoding = {
                        'input_ids': input_ids,
                        'attention_mask': input_attn,
                        'labels': input_ids,
                    }
                    self.encodings.append(encoding)
                    self.lengths.append(len(encoding['input_ids']))

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Dict[str, list]:
        return self.encodings[idx]
