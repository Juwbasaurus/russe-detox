from typing import Optional


class Collate:

    def __init__(
        self,
        pad_value: int = 0,
        max_len: Optional[int] = None,
        device: str = 'cpu',
    ):
        self.pad_value = pad_value
        self.max_len = max_len

    def __call__(
        self,
        batch,
    ):
        collated_batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
        }
        for sample in batch:
            for k, v in sample.items():
                collated_batch[k].append(v)
