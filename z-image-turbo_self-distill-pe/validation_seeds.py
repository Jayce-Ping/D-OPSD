"""Pure validation-seed generation shared by all model branches."""


def create_validation_seeds(num_samples: int, base_seed: int) -> list[int]:
    if type(num_samples) is not int:
        raise TypeError(
            f"expected int for num_samples, got {type(num_samples).__name__}: "
            f"{num_samples!r}"
        )
    if num_samples <= 0:
        raise ValueError(f"expected num_samples > 0, got {num_samples}")
    if type(base_seed) is not int:
        raise TypeError(
            f"expected int for base_seed, got {type(base_seed).__name__}: "
            f"{base_seed!r}"
        )
    if base_seed < 0:
        raise ValueError(f"expected base_seed >= 0, got {base_seed}")
    return [base_seed + index for index in range(num_samples)]
