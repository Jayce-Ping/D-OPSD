# FLUX2-Klein-base PE Self-Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `flux2-klein-base_self-distill-pe/` — a prompt-enhance (student=`f(p0)`, teacher=`f(p1)`) self-distillation variant of FLUX.2-Klein (4B/9B) on `Flux2KleinPipeline`, mirroring `z-image-turbo_self-distill-pe`.

**Architecture:** Port the D-OPSD Flux2 training skeleton from `flux2-klein_self-distill-edit`, replace the edit-branch teacher with a text-only enhanced-prompt teacher (no image latents, no `edit_sys_prompt`), and reuse the z-image PE prompt-pair loader, deterministic validation, dataset builder, and tests.

**Tech Stack:** Python 3, standard-library `unittest`, PyTorch, Accelerate, Diffusers (`Flux2KleinPipeline`), PEFT.

## Global Constraints

- Single directory `flux2-klein-base_self-distill-pe/` with `scripts/train_lora_4b.sh` and `scripts/train_lora_9b.sh`.
- Teacher is text-only `f(p1)`; no image-latent concatenation, no `--edit-sys-prompt`.
- Data via `build_geneval_pe_dataset.py` → `dataset/geneval_pe/{train,test}.jsonl` (train 33,199 / test 553).
- `p0` = `geneval.prompt`; `p1` = `geneval_enhanced.prompt`; verify `geneval.prompt == geneval_enhanced.orig_prompt`.
- Text-only resolution configurable via `--train-height`/`--train-width`, default 512.
- Base `p0`/`p1` images generated once; each checkpoint generates only student(p0) and teacher(p1); shared per-index noise via `create_validation_generators`.
- Default `--ema-decay 1.0` (teacher frozen at base(p1)); keep configurable.
- Flux2 flow-matching preserved from edit variant: `x_0 = latents + (0 - t) * v_pred`, `next_t = 0` at final step, scheduler timesteps via `compute_empirical_mu`.
- Fail-fast validation with specific exceptions; no commit unless the user explicitly asks.
- No local GPU run; verify via unit tests, dataset build, `py_compile`, lint.

---

### Task 1: Scaffold directory and port shared modules + tests + data

**Files:**
- Create dir: `flux2-klein-base_self-distill-pe/`
- Copy verbatim from `z-image-turbo_self-distill-pe/`: `prompt_pair_data.py`, `validation_seeds.py`, `build_geneval_pe_dataset.py`, `dataset.py`, `dataset_validate.py`, `arguments.py`, and `tests/{test_build_geneval_pe_dataset.py,test_prompt_pair_data.py,test_validation_seeds.py,test_arguments.py}`.
- Copy verbatim from `flux2-klein_self-distill-edit/`: `local_paths.py`, `ema_utils.py`, `configs/default.yaml`, `configs/z2.json`.
- Modify: `flux2-klein-base_self-distill-pe/arguments.py` — remove `--edit-sys-prompt` if present (it is not in z-image PE version, so nothing to remove after copy), keep wandb args and `--train-height/--train-width`; change `--wandb-project` default to `d-opsd-flux2-klein-pe`.
- Create: `flux2-klein-base_self-distill-pe/utils.py` (Flux2 `_encode_prompt` is defined in `train_dopsd.py`; `utils.py` provides `create_validation_generators`).

**Interfaces:**
- Produces: `build_dataset`, `merge_split`, `PromptPairRecords`, `PromptPairDataset`, `AspectBatchSampler`, `CustomDataLoader`, `parse_ratios`, `parse_prompt_key_pairs`, `TextPromptDataset`, `read_validation_prompt_pairs`, `create_validation_seeds`, `positive_int`.
- Produces: `create_validation_generators(num_samples: int, base_seed: int) -> list[torch.Generator]` in `utils.py`.

- [ ] **Step 1: Create directory and copy files** (via shell `cp`).
- [ ] **Step 2: Write `utils.py`** with `create_validation_generators` importing `create_validation_seeds`.
- [ ] **Step 3: Adjust `arguments.py` wandb project default** to `d-opsd-flux2-klein-pe`.
- [ ] **Step 4: Run the ported unit suite**

Run: `cd flux2-klein-base_self-distill-pe && python3 -m unittest discover -s tests -v`
Expected: all tests pass (builder, prompt-pair, validation seeds, arguments).

- [ ] **Step 5: Generate and validate the dataset**

Run: `python3 build_geneval_pe_dataset.py` → `train: 33199`, `test: 553`; then a schema/count/uniqueness/alignment check reports both splits OK.

---

### Task 2: Port `train_dopsd.py` to Flux2 base PE

**Files:**
- Create: `flux2-klein-base_self-distill-pe/train_dopsd.py`

**Interfaces:**
- Consumes: `PromptPairDataset`, `AspectBatchSampler`, `CustomDataLoader`, `parse_ratios`, `parse_prompt_key_pairs` (dataset.py); `TextPromptDataset` (dataset_validate.py); `create_validation_generators` (utils.py); `init_dual_lora_transformer`, `ema_update_lora_adapter` (ema_utils.py); `resolve_existing_path` (local_paths.py).
- Produces: training entrypoint `main(args)`.

Base this file on `flux2-klein_self-distill-edit/train_dopsd.py`, keeping all Flux2 helpers (`compute_empirical_mu`, `_prepare_text_ids`, `_encode_prompt` returning `(prompt_embeds, text_ids)`, `_patchify_latents`, `_unpatchify_latents`, `_unpack_latents_with_ids`, `decode_flux_packed_x0_to_images`, `save_student_teacher_trajectory`). Apply the z-image PE structural changes:

- [ ] **Step 1: Imports and dataset wiring** — import `PromptPairDataset`/`parse_prompt_key_pairs` and `TextPromptDataset` (p0/p1), `create_validation_generators`; build `prompt_key_pairs = parse_prompt_key_pairs(args.prompt_key_pairs)`; `test_h, test_w = args.train_height, args.train_width`; construct `PromptPairDataset(..., default_resolution=(args.train_width, args.train_height))`; `AspectBatchSampler(target_resolutions=train_dataset.target_resolutions, prompt_key_pairs=prompt_key_pairs, ...)`; `TextPromptDataset(student_prompt_key=args.student_prompt_key, teacher_prompt_key=args.teacher_prompt_key, ...)`. Keep wandb `log_with`/`init_trackers` block.
- [ ] **Step 2: Remove edit teacher context** — delete `prepare_batch_image_latents`, `teacher_image_latents`/`teacher_image_latent_ids`, and the `edit_sys_prompt` concatenations.
- [ ] **Step 3: Batch consumption** — `test_student_prompts, test_teacher_prompts = next(iter(test_dataloader))`; in the loop use `student_prompts = batch["student_prompts"]`, `teacher_prompts = batch["teacher_prompts"]`, `bsz = len(student_prompts)`, `h = batch["height"]`, `w = batch["width"]` (no `pixel_values`).
- [ ] **Step 4: Prompt encoding** — `prompt_embeds, txt_ids = _encode_prompt(text_encoder, tokenizer, student_prompts, ...)` and `teacher_prompt_embeds, teacher_txt_ids = _encode_prompt(..., teacher_prompts, ...)`.
- [ ] **Step 5: Teacher/student forwards** — keep Flux2 flow matching (`x_0 = latents + (0 - t) * v_pred`, `next_t = 0` at last step, scheduler timesteps from `compute_empirical_mu`). Teacher: `set_adapter("teacher")`, `gen_model(hidden_states=latents_student, timestep=t, guidance=None, encoder_hidden_states=teacher_prompt_embeds, txt_ids=teacher_txt_ids, img_ids=latent_ids, return_dict=False)[0][:, :latents_student.size(1)]` under `no_grad`. Student: `set_adapter("student")`, same call with `prompt_embeds`/`txt_ids`. Loss `F.mse_loss(x_0_student, x_0_teacher.detach())`.
- [ ] **Step 6: Base validation images (once)** — before the loop, `disable_adapter()` and run the base pipeline for `base_p0` (test_student_prompts) and `base_p1` (test_teacher_prompts), each with a fresh `create_validation_generators(len(prompts), 2026)`; save `samples_base_p0.png`, `samples_base_p1.png`.
- [ ] **Step 7: Checkpoint validation images** — at each sampling step, student adapter + `create_validation_generators(...)` on `test_student_prompts` → `samples_step_<n>_student_p0.png`; teacher adapter + fresh generators on `test_teacher_prompts` (plain base pipeline call, no `image=`) → `samples_step_<n>_teacher_p1.png`. Keep `save_student_teacher_trajectory` with `latent_ids`, `vae_dtype`, `latents_bn_mean`, `latents_bn_std`.
- [ ] **Step 8: Compile** — `python3 -m py_compile train_dopsd.py` exits 0.

---

### Task 3: Scripts and README

**Files:**
- Create: `flux2-klein-base_self-distill-pe/scripts/train_lora_4b.sh`
- Create: `flux2-klein-base_self-distill-pe/scripts/train_lora_9b.sh`
- Create: `flux2-klein-base_self-distill-pe/README.md`

- [ ] **Step 1: Write 4B script** — base on `flux2-klein_self-distill-edit/scripts/train_lora_4b.sh`; `--pretrained_model black-forest-labs/FLUX.2-klein-4B`; set `--data-path-*` to `dataset/geneval_pe/{train,test}.jsonl`; add `--prompt-key-pairs "p0:p1" --student-prompt-key "p0" --teacher-prompt-key "p1" --train-height 512 --train-width 512`; `--ema-decay 1.0`; remove `--edit-sys-prompt`.
- [ ] **Step 2: Write 9B script** — identical except `--pretrained_model black-forest-labs/FLUX.2-klein-9B`.
- [ ] **Step 3: Write README** — describe PE-on-base concept, data build command, text-only fixed resolution, validation image meanings (base_p0/base_p1 once + student_p0/teacher_p1 per checkpoint), 4B/9B training, and `PeftModel.from_pretrained(..., ".../lora_gen_step_<step>/student")` inference feeding only `p0`.

---

### Task 4: Final verification

- [ ] **Step 1: Full unit suite** — `python3 -m unittest discover -s tests -v` → all pass.
- [ ] **Step 2: Compile all modules** — `python3 -m py_compile build_geneval_pe_dataset.py prompt_pair_data.py validation_seeds.py dataset.py dataset_validate.py arguments.py utils.py train_dopsd.py` exits 0.
- [ ] **Step 3: Dataset integrity** — re-check counts (33,199/553), schema keys, non-empty p0/p1, unique p0, contiguous source_row.
- [ ] **Step 4: Lint** — inspect diagnostics for new/modified Python files (torch import-resolution warnings are expected offline).
- [ ] **Step 5: Report** — summarize; do not claim GPU runtime success.
