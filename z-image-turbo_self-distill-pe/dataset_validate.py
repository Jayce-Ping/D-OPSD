from torch.utils.data import Dataset
import json
from pathlib import Path
from local_paths import resolve_existing_path


class TextPromptDataset(Dataset):
    """Validation prompts for the prompt-enhance teacher setting.

    Each item returns a pair (student_prompt p0, teacher_prompt p1) read from the
    same jsonl row, so the student can be sampled on p0 and the teacher on p1.
    """

    def __init__(self, dataset_path="a.jsonl", student_prompt_key="short_en",
                 teacher_prompt_key="detailed_en", num_prompts=16,
                 data_root: str | Path | None = None):
        # only read the first num_prompts lines
        self.data_root = Path(data_root).expanduser().resolve() if data_root is not None else Path(__file__).resolve().parent
        self.dataset_path = resolve_existing_path(dataset_path, self.data_root)
        with open(self.dataset_path, 'r') as f:
            all_data = [json.loads(line.strip()) for line in f.readlines()]
        self.student_prompts = []   # p0
        self.teacher_prompts = []   # p1 = PE(p0)
        for data in all_data:
            self.student_prompts.append(str(data.get(student_prompt_key, "")))
            self.teacher_prompts.append(str(data.get(teacher_prompt_key, "")))
            if len(self.student_prompts) >= num_prompts:
                break

    def __len__(self):
        return len(self.student_prompts)

    def __getitem__(self, idx):
        return self.student_prompts[idx], self.teacher_prompts[idx]
