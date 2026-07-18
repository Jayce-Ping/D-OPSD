# Z-Image-Turbo Tuning with D-OPSD (prompt-enhance teacher)

This variant distills a **prompt enhancer** into the DiT. It is exactly D-OPSD,
except the teacher context is the **enhanced prompt `p1`** instead of the
multimodal (prompt + target image) context:

```
student condition   c_s = f(p0)     # f = Z-Image text encoder, p0 = short prompt
teacher  condition  c_t = f(p1)     # p1 = PE(p0) = enhanced / detailed prompt
```

On-policy self-distillation is unchanged: the student rolls out a few-step
trajectory conditioned on `p0`; the teacher predicts on the **same** states
conditioned on `p1`; the student is trained to match the teacher (x0-space MSE,
stop-grad on the teacher). After training, inference takes **only `p0`** and no
LLM/MLLM rewriting is needed.

Difference vs `../z-image-turbo_self-distill-vlm`:

| | `*-vlm` (original D-OPSD) | `*-pe` (this variant) |
|---|---|---|
| teacher context | prompt + target image via Qwen3-VL `f_mm` | enhanced prompt `p1` via the same text encoder `f` |
| extra model at train | Qwen3-VL-4B | none |
| data per row | image + text | `(p0, p1)` text pair (image only sets resolution) |
| goal | inject target concept/style | internalize prompt enhancement |

## Data format

The default training data combines:

- `Flow-Factory-Private/dataset/geneval`: original prompts (`p0`)
- `Flow-Factory-Private/dataset/geneval_enhanced`: enhanced prompts (`p1`)

Build the aligned, self-contained dataset with:

```bash
python3 build_geneval_pe_dataset.py
```

This writes:

- `dataset/geneval_pe/train.jsonl` (33,199 rows)
- `dataset/geneval_pe/test.jsonl` (553 rows)

Each row contains `p0`, `p1`, Geneval metadata, `source_split`, and
`source_row`. The builder checks row counts, prompt alignment, non-empty
values, and uniqueness before writing anything.

The loader is text-only: it does not open or transfer target images. Rows
without `h*w`/`w*h` use `--train-height` and `--train-width` (both default to
512). Legacy rows with dimensions continue to use aspect-ratio buckets.

## Training

```bash
cd z-image-turbo_self-distill-pe
bash scripts/train_lora_pe.sh
```

`--ema-decay 1.0` keeps the teacher frozen at the base model (cleanest:
distill `base(p1)` → `student(p0)`); set `< 1.0` for an EMA teacher.
Training runs in 4 steps; inference keeps the standard 8-step Z-Image-Turbo
pipeline. Validation uses the same per-index noise for p0 and p1 branches.
The samples folder contains:

- `samples_base_p0.png` — fixed base model on original prompts
- `samples_base_p1.png` — fixed base model on enhanced prompts
- `samples_step_i_student_p0.png` — current student on original prompts
- `samples_step_i_teacher_p1.png` — current teacher on enhanced prompts

The two base grids are generated once before training. Student and teacher
grids are generated on each configured sampling checkpoint.

## Inference

Identical to the original Z-Image-Turbo pipeline: feed only `p0` and load the
student adapter from:

```text
exp_results/<exp-name>/checkpoints/lora_gen_step_<step>/student
```

Load that directory with `PeftModel.from_pretrained(pipe.transformer, path)`.
No prompt enhancer is needed at inference time.

## Acknowledgement

Built on top of [vvvvvjdy/D-OPSD](https://github.com/vvvvvjdy/D-OPSD),
[vvvvvjdy/dmdr](https://github.com/vvvvvjdy/dmdr) and
[Tongyi-MAI/Z-Image](https://github.com/Tongyi-MAI/Z-Image).
