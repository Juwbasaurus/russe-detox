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
        eos_id = tokenizer.eos_token_id

        with open(source_path, 'r', encoding='utf8') as source:
            with open(target_path, 'r', encoding='utf8') as target:
                for src, tgt in tqdm(zip(source, target)):
                    src, tgt = preprocess_source(src), preprocess_target(tgt)
                    input_text = f'{src} {SEP} {tgt}'
                    tokenized_input = self.tokenizer(
                        input_text,
                        return_token_type_ids=False,)
                    input_ids = tokenized_input.input_ids + [eos_id]
                    input_attn = tokenized_input.attention_mask + [1]
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
