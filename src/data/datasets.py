from typing import Dict

from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizerFast,
)

from utils.data_utils import preprocess_source, preprocess_target


class PromptDataset(Dataset):

    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizerFast,
                 separator: str,):
        super().__init__()
        source_path = f'{data_path}/source.txt'
        target_path = f'{data_path}/target.txt'
        self.tokenizer = tokenizer
        self.separator = separator
        self.encodings = []
        self.lengths = []

        with open(source_path, 'r', encoding='utf8') as source:
            with open(target_path, 'r', encoding='utf8') as target:
                for src, tgt in tqdm(zip(source, target)):
                    src, tgt = preprocess_source(src), preprocess_target(tgt)
                    src_tokenized = self.tokenizer(
                        src,
                        return_token_type_ids=False,
                    )
                    tgt_tokenized = self.tokenizer(
                        tgt,
                        return_token_type_ids=False,
                    )
                    sep = tokenizer.encode(self.separator)
                    input_ids = src_tokenized.input_ids + sep + tgt_tokenized.input_ids + [self.tokenizer.eos_token_id]
                    input_attn = src_tokenized.attention_mask + [1] + tgt_tokenized.attention_mask + [1]
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


class Seq2SeqDataset(Dataset):

    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizerFast,):
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
                    source_encodings = self.tokenizer(src, return_token_type_ids=False)
                    target_encodings = self.tokenizer(tgt, return_token_type_ids=False)
                    encoding = {
                        'input_ids': source_encodings.input_ids,
                        'attention_mask': source_encodings.attention_mask,
                        'labels': target_encodings.input_ids,
                        'decoder_attention_mask': target_encodings.attention_mask,
                    }
                    self.encodings.append(encoding)
                    self.lengths.append(len(encoding['input_ids']))

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Dict[str, list]:
        return self.encodings[idx]
