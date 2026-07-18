# FLUX.2-Klein-base Tuning with D-OPSD (prompt-enhance teacher)

This variant distills a **prompt enhancer** into the FLUX.2-Klein base DiT (4B and
9B). It is D-OPSD, except the teacher context is the **enhanced prompt `p1`**
instead of the edit branch (reference image + edit prompt) used by
`../flux2-klein_self-distill-edit`.

```
student condition   c_s = f(p0)     # f = FLUX.2-Klein text encoder, p0 = original prompt
teacher  condition  c_t = f(p1)     # p1 = PE(p0) = enhanced / detailed prompt
```

On-policy self-distillation is unchanged: the student rolls out a few-step
trajectory conditioned on `p0`; the teacher predicts on the **same** states
conditioned on `p1` (no reference image, no edit prompt); the student is trained
to match the teacher (x0-space MSE, stop-grad on the teacher). After training,
inference takes **only `p0`** — no LLM/MLLM rewriting is needed.

`flux2-klein-base` and `flux2-klein` share the same `Flux2KleinPipeline`, so the
same pipeline drives both training and inference here.

Difference vs `../flux2-klein_self-distill-edit`:

| | `*-edit` | `*-base_self-distill-pe` (this variant) |
|---|---|---|
| teacher context | reference image (VAE latents) + edit prompt | enhanced prompt `p1` via the same text encoder `f` |
| data per row | image + text | `(p0, p1)` text pair (no image) |
| goal | inject reference identity | internalize prompt enhancement |

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
`source_row`. The builder checks row counts, prompt alignment
(`geneval.prompt == geneval_enhanced.orig_prompt`), non-empty values, and
uniqueness before writing anything.

The loader is text-only: it never opens or transfers images. Rows without
`h*w`/`w*h` use `--train-height` and `--train-width` (both default to 512); the
FLUX.2 `seqlen` and scheduler `mu` are derived from that resolution.

## Training

```bash
cd flux2-klein-base_self-distill-pe
bash scripts/train_lora_4b.sh     # FLUX.2-Klein-4B
# bash scripts/train_lora_9b.sh   # FLUX.2-Klein-9B
```

`--ema-decay 1.0` keeps the teacher frozen at the base model (cleanest: distill
`base(p1)` → `student(p0)`); set `< 1.0` for an EMA teacher. Training runs in 40
steps with `guidance_scale=1.0`. Validation uses the same per-index noise for the
p0 and p1 branches. The samples folder contains:

- `samples_base_p0.png` — fixed base model on original prompts
- `samples_base_p1.png` — fixed base model on enhanced prompts
- `samples_step_i_student_p0.png` — current student on original prompts
- `samples_step_i_teacher_p1.png` — current teacher on enhanced prompts

The two base grids are generated once before training; student and teacher grids
are generated on each configured sampling checkpoint. A student/teacher x0
trajectory grid is written under `samples_trajectory/`.

## Inference

Identical to the original FLUX.2-Klein pipeline: feed only `p0` and load the
student adapter.

```python
import torch
from diffusers import Flux2KleinPipeline
from peft import PeftModel

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-base-4B",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

lora_weights_path = "exp_results/<exp-name>/checkpoints/lora_gen_step_<step>/student"
pipe.transformer = PeftModel.from_pretrained(
    pipe.transformer,
    lora_weights_path,
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe(
    prompt="a photo of a black airplane and an orange apple",
    height=1024,
    width=1024,
    num_inference_steps=40,
    guidance_scale=1.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("sample.png")
```

## Acknowledgement

Built on top of [vvvvvjdy/D-OPSD](https://github.com/vvvvvjdy/D-OPSD),
[vvvvvjdy/dmdr](https://github.com/vvvvvjdy/dmdr) and
[black-forest-labs/flux2](https://github.com/black-forest-labs/flux2).
