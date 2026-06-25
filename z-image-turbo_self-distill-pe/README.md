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

Each `data.jsonl` row should contain a short prompt and an enhanced prompt (any
of the keys below). `p1` is produced offline by your LLM/MLLM prompt enhancer
(here the bundled `dataset/style_Millennium` toy set already ships
`short_*`/`detailed_*` pairs, usable as a smoke test).

- student key (`p0`): `short_en` / `short_zh` / `user_prompt_en` / `user_prompt_zh`
- teacher key (`p1`): `detailed_en` / `detailed_zh`
- `local_path_list`: one image path (only used to pick the generation resolution)
- `h*w` (or `w*h`): used for aspect-ratio bucketing

`--prompt-key-pairs "short_en:detailed_en,short_zh:detailed_zh,..."` selects one
`(p0, p1)` pair per batch for length/language variety.

## Training

```bash
cd z-image-turbo_self-distill-pe
bash scripts/train_lora_pe.sh
```

`--ema-decay 1.0` keeps the teacher frozen at the base model (cleanest:
distill `base(p1)` → `student(p0)`); set `< 1.0` for an EMA teacher.
Training runs in 4 steps; inference keeps the standard 8-step Z-Image-Turbo
pipeline. The samples folder logs, per step:

- `samples_original.png` — base model on `p0` (the weak starting point)
- `samples_step_i_student.png` — trained student on `p0` (should approach `img1`)
- `samples_step_i_teacher.png` — teacher on `p1` (the target)

## Inference

Identical to the original Z-Image-Turbo pipeline; just feed the short prompt
`p0` and load the trained student LoRA (see `../z-image-turbo_self-distill-vlm/README.md`
for the loading snippet — only the LoRA path changes).

## Acknowledgement

Built on top of [vvvvvjdy/D-OPSD](https://github.com/vvvvvjdy/D-OPSD),
[vvvvvjdy/dmdr](https://github.com/vvvvvjdy/dmdr) and
[Tongyi-MAI/Z-Image](https://github.com/Tongyi-MAI/Z-Image).
