import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Tuple

from prompt_pair_data import PromptPairRecords, collate_prompt_pairs

# --- Utility functions ---
def parse_ratios(ratio_strs: List[str]) -> List[Tuple[int, int]]:
    ratios = []
    for s in ratio_strs:
        res = s.split(' ')[0]
        w, h = map(int, res.split('x'))
        ratios.append((w, h))
    return ratios


def parse_prompt_key_pairs(spec: str) -> List[Tuple[str, str]]:
    """Parse a `student_key:teacher_key,student_key2:teacher_key2,...` spec into
    a list of (student_key, teacher_key) tuples.

    student_key -> p0 (short prompt), teacher_key -> p1 (enhanced prompt).
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(f"prompt_key_pairs must be a non-empty string, got {spec!r}")
    pairs: List[Tuple[str, str]] = []
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.count(':') != 1:
            raise ValueError(
                f"each prompt-key pair must be 'student_key:teacher_key', got {chunk!r}"
            )
        student_key, teacher_key = (p.strip() for p in chunk.split(':'))
        if not student_key or not teacher_key:
            raise ValueError(f"empty key in prompt-key pair {chunk!r}")
        pairs.append((student_key, teacher_key))
    if not pairs:
        raise ValueError(f"no valid prompt-key pairs parsed from {spec!r}")
    return pairs

# --- 1. Dataset ---
class PromptPairDataset(PromptPairRecords, Dataset):
    """Torch Dataset wrapper around the pure prompt-pair record implementation."""

    pass


# --- 2. Aspect Ratio Sampler ---
class AspectBatchSampler(Sampler):
    def __init__(self, buckets, target_resolutions, batch_size, prompt_key_pairs,
                 num_replicas=1, rank=0, seed=42, shuffle=True):
        self.buckets = buckets
        self.target_resolutions = target_resolutions
        self.batch_size = batch_size
        # list of (student_key, teacher_key) tuples; one pair is sampled per batch.
        self.prompt_key_pairs = prompt_key_pairs
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        all_batches = []
        for bucket_idx, indices in self.buckets.items():
            if self.shuffle:
                shuffled_indices = [indices[i] for i in torch.randperm(len(indices), generator=g).tolist()]
            else:
                shuffled_indices = indices

            target_res = self.target_resolutions[bucket_idx]
            for i in range(0, len(shuffled_indices), self.batch_size):
                batch_indices = shuffled_indices[i:i + self.batch_size]
                if len(batch_indices) == self.batch_size:
                    # Select which (student_key, teacher_key) pair to use for this batch at the sampler level
                    selected_pair = random.choice(self.prompt_key_pairs)
                    # Key point: output a list of tuples, where each element is (idx, res, (student_key, teacher_key))
                    batch_info = [(idx, target_res, selected_pair) for idx in batch_indices]
                    all_batches.append(batch_info)

        if self.shuffle:
            batch_perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in batch_perm]

        total_batches = (len(all_batches) // self.num_replicas) * self.num_replicas
        local_batches = all_batches[self.rank:total_batches:self.num_replicas]

        return iter(local_batches)

    def __len__(self):
        total_valid_batches = sum(len(indices) // self.batch_size for indices in self.buckets.values())
        return total_valid_batches // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch


# --- 3. DataLoader ---
def collate_fn(examples):
    return collate_prompt_pairs(examples)


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_sampler, batch_size, **kwargs):
        # Remove potentially conflicting arguments
        kwargs.pop('batch_size', None)
        kwargs.pop('shuffle', None)

        super().__init__(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, **kwargs)
        self._real_batch_size = batch_size

    @property
    def batch_size(self):
        return self._real_batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._real_batch_size = value

