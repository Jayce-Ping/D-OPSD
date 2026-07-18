from torch.utils.data import Dataset

from prompt_pair_data import read_validation_prompt_pairs


class TextPromptDataset(Dataset):
    """Validation prompts for the prompt-enhance teacher setting.

    Each item returns a pair (student_prompt p0, teacher_prompt p1) read from the
    same jsonl row, so the student can be sampled on p0 and the teacher on p1.
    """

    def __init__(self, dataset_path="a.jsonl", student_prompt_key="p0",
                 teacher_prompt_key="p1", num_prompts=16,
                 data_root=None):
        prompt_pairs = read_validation_prompt_pairs(
            dataset_path,
            student_prompt_key=student_prompt_key,
            teacher_prompt_key=teacher_prompt_key,
            num_prompts=num_prompts,
            data_root=data_root,
        )
        self.student_prompts = [pair[0] for pair in prompt_pairs]
        self.teacher_prompts = [pair[1] for pair in prompt_pairs]

    def __len__(self):
        return len(self.student_prompts)

    def __getitem__(self, idx):
        return self.student_prompts[idx], self.teacher_prompts[idx]
