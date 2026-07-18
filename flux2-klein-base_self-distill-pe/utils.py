import torch

from validation_seeds import create_validation_seeds


def create_validation_generators(num_samples: int, base_seed: int):
    """Create reproducible per-index generators shared across prompt variants."""
    return [
        torch.Generator().manual_seed(seed)
        for seed in create_validation_seeds(num_samples, base_seed)
    ]
