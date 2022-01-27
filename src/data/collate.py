from typing import Dict, Optional, List

import torch


class Collate:

    def __init__(
        self,
        pad_value: int = 0,
        max_len: Optional[int] = None,
        device: str = 'cpu',
    ):
        self.pad_value = pad_value
        self.max_len = max_len
        self.device = device

    def __call__(self,
                 batch: List[Dict[str, List[int]]]) -> Dict[str, List[torch.tensor]]:
        collated_batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
        }
        max_batch_len = max([len(sample['input_ids']) for sample in batch])
        max_len = min(max_batch_len, self.max_len)
        for key in collated_batch:
            pad_value = -100 if key == 'labels' else 0
            for sample in batch:
                padded_sample = self._pad(sample[key], max_len, pad_value)
                collated_batch[key].append(padded_sample)
            collated_batch[key] = torch.tensor(collated_batch[key], device=self.device)
        return collated_batch

    @staticmethod
    def _pad(sample: List[int],
             max_len: int,
             pad_value: int) -> List[int]:
        sample = sample[:max_len]  # Truncate long
        delta = max_len - len(sample)
        sample += [pad_value] * delta
        return sample
