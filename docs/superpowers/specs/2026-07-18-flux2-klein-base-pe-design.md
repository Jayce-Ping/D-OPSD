# FLUX2-Klein-base Prompt-Enhancement Self-Distillation Design

## Goal

Create a prompt-enhance (PE) self-distillation variant for the FLUX.2-Klein
base text-to-image model (4B and 9B), mirroring `z-image-turbo_self-distill-pe`
but running on `Flux2KleinPipeline`. The student is conditioned on the original
short prompt `p0`; the teacher is conditioned on the enhanced prompt `p1`. After
training, inference needs only `p0` plus the trained student LoRA.

## Relationship to Existing Variants

- `flux2-klein_self-distill-edit`: teacher context is the **edit branch**
  (`concat[student_latents, VAE(reference_image)]` + edit system prompt). It runs
  on `Flux2KleinPipeline` with `--edit-sys-prompt`.
- `z-image-turbo_self-distill-pe`: teacher context is the **enhanced prompt**
  `f(p1)`, text-only, with deterministic base/student/teacher validation images.

This variant combines both: the D-OPSD training skeleton and Flux2 transformer
call from `flux2-klein_self-distill-edit`, with the PE teacher, text-only
prompt-pair loader, deterministic validation, and tests from
`z-image-turbo_self-distill-pe`.

The only conceptual change vs the edit variant: the teacher forward drops the
image-latent concatenation and `edit_sys_prompt`, and instead conditions on the
enhanced prompt `p1` through the same text encoder as the student.

## Directory Structure

A single directory `flux2-klein-base_self-distill-pe/` (consistent with
`flux2-klein_self-distill-edit`, which ships both 4B and 9B scripts):

```
flux2-klein-base_self-distill-pe/
├── train_dopsd.py
├── arguments.py
├── dataset.py
├── dataset_validate.py
├── prompt_pair_data.py
├── validation_seeds.py
├── build_geneval_pe_dataset.py
├── utils.py
├── ema_utils.py
├── local_paths.py
├── configs/{default.yaml, z2.json}
├── scripts/{train_lora_4b.sh, train_lora_9b.sh}
├── tests/{test_build_geneval_pe_dataset.py, test_prompt_pair_data.py,
│          test_validation_seeds.py, test_arguments.py}
├── dataset/geneval_pe/{train.jsonl, test.jsonl}   # generated
└── README.md
```

No `eval/` directory (matching the flux2 family style).

## Shared / Reused Modules

These are ported (near-verbatim, framework-agnostic) from
`z-image-turbo_self-distill-pe`:

- `prompt_pair_data.py`: `PromptPairRecords`, `collate_prompt_pairs`,
  `read_validation_prompt_pairs`. Text-only, `default_resolution`, aspect and
  fixed-resolution buckets, fail-fast prompt validation.
- `validation_seeds.py`: `create_validation_seeds(num_samples, base_seed)`.
- `build_geneval_pe_dataset.py`: `merge_split`, `build_dataset`, CLI. Defaults
  point to `Flow-Factory-Private/dataset/geneval` and `geneval_enhanced`, output
  `dataset/geneval_pe`.
- `dataset.py`: `PromptPairDataset(PromptPairRecords, Dataset)`,
  `AspectBatchSampler`, `CustomDataLoader`, `collate_fn`, `parse_ratios`,
  `parse_prompt_key_pairs`.
- `dataset_validate.py`: `TextPromptDataset` returning `(p0, p1)` pairs.
- `ema_utils.py`, `local_paths.py`, `configs/*`: copied unchanged.

## Data

`build_geneval_pe_dataset.py` pairs the two Geneval sources row by row,
validating: equal split counts, `geneval.prompt == geneval_enhanced.orig_prompt`,
non-empty string prompts, and unique `p0` per split. It writes
`dataset/geneval_pe/train.jsonl` (33,199 rows) and `test.jsonl` (553 rows), each
row containing `p0`, `p1`, `tag`, `include`, `exclude`, `source_split`,
`source_row`. All splits are validated before anything is written.

The loader is text-only: it never opens or transfers images. Rows without
`h*w`/`w*h` use the configured fixed resolution; the Flux2 `seqlen` and `mu` are
derived from that resolution.

## Arguments

Start from the edit variant's `arguments.py`, then:

- Remove `--edit-sys-prompt`.
- Add `positive_int` type helper.
- Add `--train-height` and `--train-width`, both `positive_int`, default `512`.
- Set dataset defaults to `dataset/geneval_pe/train.jsonl` and
  `dataset/geneval_pe/test.jsonl`.
- Add `--prompt-key-pairs` default `"p0:p1"`, `--student-prompt-key` default
  `"p0"`, `--teacher-prompt-key` default `"p1"`.
- Keep `--ema-decay` (training default set to `1.0` via the scripts).

## Training Loop (`train_dopsd.py`)

Adapt the edit variant's loop:

- Keep Flux2 helpers: `compute_empirical_mu`, `_prepare_text_ids`,
  `_encode_prompt` (returns `(prompt_embeds, text_ids)`), `_patchify_latents`,
  `_unpatchify_latents`, `_unpack_latents_with_ids`,
  `decode_flux_packed_x0_to_images`, `save_student_teacher_trajectory`.
- Remove image-latent preparation (`prepare_batch_image_latents`) and all edit
  teacher context.
- Encode both prompts through the same text encoder:
  - student: `prompt_embeds, txt_ids = _encode_prompt(p0)`
  - teacher: `teacher_prompt_embeds, teacher_txt_ids = _encode_prompt(p1)`
- Flow matching unchanged from the edit variant: `t = timestep/1000`,
  `next_t = 0` at the final step, `x_0 = latents + (0 - t) * v_pred`.
- Teacher forward (no image latents, `no_grad`, `set_adapter("teacher")`):
  `hidden_states=latents_student`, `encoder_hidden_states=teacher_prompt_embeds`,
  `txt_ids=teacher_txt_ids`, `img_ids=latent_ids`.
- Student forward (`set_adapter("student")`): same states with `p0` embeds.
- Loss: `F.mse_loss(x_0_student, x_0_teacher.detach())`, averaged across steps.
- Dual LoRA via `init_dual_lora_transformer` with Flux2 target modules; only the
  student adapter is trainable; EMA update after each optimizer step.

## Validation Images (deterministic, shared noise)

- Before training, generate exactly once, with adapters disabled:
  - `samples_base_p0.png` (base on original prompts)
  - `samples_base_p1.png` (base on enhanced prompts)
- At each sampling checkpoint:
  - `samples_step_<step>_student_p0.png` (student adapter, `p0`)
  - `samples_step_<step>_teacher_p1.png` (teacher adapter, `p1`)
- Each branch rebuilds generators via `create_validation_generators(n, 2026)`
  immediately before sampling, so corresponding grid cells share initial noise.
  Base grids are not regenerated because the base model is fixed.
- Keep the student/teacher x0 trajectory grid (`save_student_teacher_trajectory`)
  driven by the training rollout latents.
- No `student(p1)` images.

The validation teacher is sampled through the plain base pipeline call on `p1`
(no `image=` argument), unlike the edit variant which called the edit pipeline.

## Scripts

`scripts/train_lora_4b.sh` and `scripts/train_lora_9b.sh` differ only in
`--pretrained_model` (`black-forest-labs/FLUX.2-klein-4B` vs `-9B`). Both set:

- `--data-path-train-jsonl dataset/geneval_pe/train.jsonl`
- `--data-path-test-jsonl dataset/geneval_pe/test.jsonl`
- `--prompt-key-pairs "p0:p1"`, `--student-prompt-key "p0"`,
  `--teacher-prompt-key "p1"`
- `--train-height 512`, `--train-width 512`
- `--num-training-steps 4`, `--use-lora 2`, `--ema-decay 1.0`
- No `--edit-sys-prompt`.

## Error Handling

All new data and loader validation is fail-fast: missing/empty/mistyped values
and mixed batch resolutions raise specific exceptions naming the file, split,
row, and field. No missing prompt becomes an empty string; no broad handler
substitutes another sample.

## Tests and Verification

Standard-library `unittest`, test-first. Mirror the z-image PE tests (all pure
Python, no torch needed):

1. builder: aligned merge, count mismatch, prompt mismatch, empty prompt,
   duplicate `p0`, atomic all-splits-before-write.
2. prompt-pair: text-only default resolution bucket, legacy aspect bucket,
   missing/empty prompt failure, collate pairs + single resolution, mixed
   resolution rejection, validation-pair reader.
3. validation seeds: reproducible list, distinct per index, invalid-type/count
   rejection.
4. arguments: `positive_int` accepts 512, rejects 0/-1/non-integer.

After implementation:

- run the unit suite;
- build the full dataset and validate counts, schema, non-empty, uniqueness,
  alignment;
- `py_compile` all modified/new Python files;
- inspect linter diagnostics.

A full GPU training run is out of scope (no local torch/GPU). The handoff notes
a smoke-training command for later runtime verification.

## Scope

Included: the new directory with all code, generated geneval_pe data, 4B/9B
scripts, README, and focused unit tests.

Excluded: online prompt enhancement, changes to the D-OPSD loss or teacher EMA
algorithm, an eval pipeline, real GPU training, and student-on-enhanced-prompt
validation images.
