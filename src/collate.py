from typing import Optional, List


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
        max_batch_len = max([len(sample) for sample in batch])
        max_len = min(max_batch_len, self.max_len)
        for sample in batch:
            for k, v in sample.items():
                pad_value = -100 if k == 'labels' else 0
                collated_batch[k].append(self._pad(v, max_len, pad_value))
        return collated_batch

    def _pad(
        self,
        sample: List[int],
        max_len: int,
        pad_value: int,
    ):
        sample = sample[:max_len]  # Truncate long
        delta = len(sample) - max_len
        sample += [pad_value] * delta
        return sample
