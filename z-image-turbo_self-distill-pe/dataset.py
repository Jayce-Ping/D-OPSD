import json
import random
import torch
import math
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
from typing import List, Tuple, Dict
from torchvision.transforms import functional as F
from local_paths import get_local_image_path, resolve_existing_path

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

def has_transparency(img: Image.Image) -> bool:
    if img.mode in ("RGBA", "LA"):
        return True
    if img.mode == "P" and "transparency" in img.info:
        return True
    return False

def to_rgb_safely(img: Image.Image, bg=(255, 255, 255)) -> Image.Image:
    if has_transparency(img):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, bg)
        background.paste(img, mask=img.getchannel("A"))
        return background
    return img.convert("RGB")


# --- 1. Dataset ---
class TextImageDataset(Dataset):
    def __init__(self, jsonl_path: str, target_resolutions: List[Tuple[int, int]], data_root: str | Path | None = None):
        self.data = []
        self.data_root = Path(data_root).expanduser().resolve() if data_root is not None else Path(__file__).resolve().parent
        self.jsonl_path = resolve_existing_path(jsonl_path, self.data_root)
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.target_resolutions = target_resolutions
        self.buckets: Dict[int, List[int]] = {i: [] for i in range(len(target_resolutions))}
        self._build_buckets()

    def _build_buckets(self):
        for idx, item in enumerate(self.data):
            if 'h*w' in item:
                h_str, w_str = item['h*w'].split('*')
            elif 'w*h' in item:
                w_str, h_str = item['w*h'].split('*')
            else:
                raise KeyError("Item contains neither 'h*w' nor 'w*h'")
            orig_h, orig_w = float(h_str), float(w_str)
            orig_ratio = orig_w / orig_h
            best_ratio_idx = 0
            min_diff = float('inf')
            for i, (tw, th) in enumerate(self.target_resolutions):
                target_ratio = tw / th
                diff = abs(math.log(orig_ratio) - math.log(target_ratio))
                if diff < min_diff:
                    min_diff = diff
                    best_ratio_idx = i
            self.buckets[best_ratio_idx].append(idx)

    def process_image(self, image: Image.Image, target_size: Tuple[int, int]):
        tw, th = target_size
        w, h = image.size
        scale = max(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
        left = (new_w - tw) // 2
        top = (new_h - th) // 2
        image = image.crop((left, top, left + tw, top + th))
        img_tensor = F.to_tensor(image)
        img_tensor = F.normalize(img_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        return img_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, info):
        # Normalize the input info format.
        # `prompt_key_pair` is a (student_key, teacher_key) tuple: student -> p0, teacher -> p1.
        if isinstance(info, (tuple, list)):
            if len(info) == 3:
                idx, target_res, prompt_key_pair = info
                retry_count = 0
            elif len(info) == 4:
                idx, target_res, prompt_key_pair, retry_count = info
            else:
                raise ValueError(f"Invalid info length: {len(info)}")
        else:
            # Handle corner cases where only an integer index is passed
            idx = info
            target_res = self.target_resolutions[0]
            prompt_key_pair = ("short_en", "detailed_en")  # default (p0, p1)
            retry_count = 0

        if retry_count > 10:
            raise RuntimeError(f"Retry limit reached for index {idx}")

        student_key, teacher_key = prompt_key_pair

        item = self.data[idx]
        try:
            img_path = get_local_image_path(item, jsonl_path=self.jsonl_path, data_root=self.data_root)
            with Image.open(img_path) as img:
                image = to_rgb_safely(img)
            pixel_values = self.process_image(image, target_res)
        except Exception as e:
            bucket_idx = self.target_resolutions.index(target_res)
            new_idx = random.choice(self.buckets[bucket_idx])
            return self.__getitem__((new_idx, target_res, prompt_key_pair, retry_count + 1))

        return {
            "pixel_values": pixel_values,
            "student_prompt": str(item.get(student_key, "")),   # p0
            "teacher_prompt": str(item.get(teacher_key, "")),   # p1 = PE(p0)
            "prompt_pair": f"{student_key}->{teacher_key}",
        }


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
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    student_prompts = [example["student_prompt"] for example in examples]   # p0
    teacher_prompts = [example["teacher_prompt"] for example in examples]   # p1 = PE(p0)
    prompt_pairs = [example["prompt_pair"] for example in examples]
    return {
        "pixel_values": pixel_values,
        "student_prompts": student_prompts,
        "teacher_prompts": teacher_prompts,
        "prompt_pairs": prompt_pairs,
    }


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

