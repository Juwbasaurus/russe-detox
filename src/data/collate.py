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
        # collated_batch = {
        #     'input_ids': [],
        #     'attention_mask': [],
        #     'labels': [],
        #     'decoder_attention_mask': [],
        # }
        collated_batch = {}
        for key in batch[0]:
            max_batch_len = max([len(sample[key]) for sample in batch])
            max_len = min(max_batch_len, self.max_len)
            pad_value = -100 if key == 'labels' else 0
            samples = [sample[key] for sample in batch]
            padded_samples = self._pad(samples, max_len, pad_value)
            collated_batch[key] = torch.tensor(padded_samples, device=self.device)
        return collated_batch

    @staticmethod
    def _pad(samples: List[List[int]],
             max_len: int,
             pad_value: int = 0) -> List[List[int]]:
        padded = []
        max_len = min(max([len(sample) for sample in samples]), max_len)
        for sample in samples:
            sample = sample[:max_len]  # Truncate long
            delta = max_len - len(sample)
            sample += [pad_value] * delta
            padded.append(sample)
        return padded
