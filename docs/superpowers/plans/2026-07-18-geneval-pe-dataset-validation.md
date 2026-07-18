# Geneval PE Dataset and Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build aligned Geneval `p0`/`p1` datasets, make PE training natively support text-only fixed-resolution data, and generate deterministic base/student/teacher validation images.

**Architecture:** A standalone builder validates and materializes the two source datasets into self-contained JSONL files. The PE loader becomes a prompt-pair loader that returns dimensions instead of unused pixels. Validation uses fresh index-seeded generators for each model branch so corresponding images share initial noise.

**Tech Stack:** Python 3, standard-library `json`/`argparse`/`unittest`, PyTorch, Accelerate, Diffusers, PEFT.

## Global Constraints

- Generated train and test sizes must be exactly 33,199 and 553 for the supplied sources.
- `p0` is `geneval.prompt`; `p1` is `geneval_enhanced.prompt`.
- Pairing must verify `geneval.prompt == geneval_enhanced.orig_prompt`.
- Text-only resolution is configurable and defaults to 512×512.
- Base p0/p1 images are generated once; checkpoint validation generates only student(p0) and teacher(p1).
- Missing or malformed values raise specific informative exceptions; no broad exception substitutes data.
- Use test-first development and do not create a git commit unless the user explicitly requests one.

---

### Task 1: Reproducible Geneval Pair Builder

**Files:**
- Create: `z-image-turbo_self-distill-pe/build_geneval_pe_dataset.py`
- Create: `z-image-turbo_self-distill-pe/tests/test_build_geneval_pe_dataset.py`
- Generate: `z-image-turbo_self-distill-pe/dataset/geneval_pe/train.jsonl`
- Generate: `z-image-turbo_self-distill-pe/dataset/geneval_pe/test.jsonl`

**Interfaces:**
- Produces: `merge_split(original_path: Path, enhanced_path: Path, split: str) -> list[dict[str, object]]`
- Produces: `build_dataset(original_dir: Path, enhanced_dir: Path, output_dir: Path) -> dict[str, int]`
- Output rows contain `p0`, `p1`, `tag`, `include`, `exclude`, `source_split`, and `source_row`.

- [ ] **Step 1: Write failing builder tests**

Create fixture JSONL files in `tempfile.TemporaryDirectory` and test:

```python
def test_merge_split_builds_paired_rows(self):
    rows = merge_split(self.original_path, self.enhanced_path, "train")
    self.assertEqual(rows[0]["p0"], "a red cube")
    self.assertEqual(rows[0]["p1"], "A polished red cube on a table.")
    self.assertEqual(rows[0]["source_split"], "train")
    self.assertEqual(rows[0]["source_row"], 0)
```

Add separate tests asserting `ValueError` messages for count mismatch, prompt mismatch, empty prompt, and duplicate p0. Messages must include the split and row where applicable.

- [ ] **Step 2: Run builder tests and verify RED**

Run:

```bash
cd z-image-turbo_self-distill-pe
python3 -m unittest tests.test_build_geneval_pe_dataset -v
```

Expected: import failure because `build_geneval_pe_dataset` does not exist.

- [ ] **Step 3: Implement strict merge and CLI**

Implement line-aware JSONL loading, exact row-count comparison, string/non-empty validation, original-prompt alignment, duplicate detection, and UTF-8 JSONL writing with `ensure_ascii=False`. The CLI accepts:

```text
--original-dir
--enhanced-dir
--output-dir
```

Defaults point to the two supplied Flow-Factory-Private source directories and `dataset/geneval_pe`.

- [ ] **Step 4: Run builder tests and verify GREEN**

Run the same unittest command. Expected: all builder tests pass.

- [ ] **Step 5: Generate and validate the complete dataset**

Run:

```bash
cd z-image-turbo_self-distill-pe
python3 build_geneval_pe_dataset.py
```

Expected summary: `train: 33199`, `test: 553`.

Run an independent validation script that checks exact keys, counts, non-empty p0/p1, unique p0, and source-row continuity. Expected: both splits report OK.

---

### Task 2: Native Text-Only Prompt-Pair Loader

**Files:**
- Modify: `z-image-turbo_self-distill-pe/dataset.py`
- Modify: `z-image-turbo_self-distill-pe/train_dopsd.py`
- Create: `z-image-turbo_self-distill-pe/tests/test_prompt_pair_dataset.py`

**Interfaces:**
- Produces: `PromptPairDataset(jsonl_path, target_resolutions, default_resolution, data_root=None)`
- Produces dataset properties `target_resolutions: list[tuple[int, int]]` and `buckets: dict[int, list[int]]`.
- Dataset items contain `student_prompt`, `teacher_prompt`, `prompt_pair`, `width`, and `height`.
- `collate_fn` returns prompt lists plus scalar integer `width` and `height`.

- [ ] **Step 1: Write failing loader tests**

Test a text-only row with `p0`/`p1` and no image or size:

```python
dataset = PromptPairDataset(
    jsonl_path,
    target_resolutions=[(1024, 1024), (1152, 896)],
    default_resolution=(512, 512),
)
item = dataset[(0, (512, 512), ("p0", "p1"))]
self.assertEqual((item["width"], item["height"]), (512, 512))
```

Also test:

- a legacy `h*w` row is assigned to an existing aspect-ratio bucket rather than the 512 default bucket;
- missing/empty prompt keys raise `KeyError` or `ValueError` containing key, row, and file;
- `collate_fn` returns one width/height and rejects mixed dimensions.

- [ ] **Step 2: Run loader tests and verify RED**

Run:

```bash
cd z-image-turbo_self-distill-pe
python3 -m unittest tests.test_prompt_pair_dataset -v
```

Expected: import failure because `PromptPairDataset` does not exist.

- [ ] **Step 3: Implement the prompt-only dataset**

Remove image opening, transforms, retry substitution, and `pixel_values`. Keep ratio parsing and distributed aspect batching. Add the default resolution as a dedicated target bucket only when absent; rows with declared dimensions select only among the original aspect targets, while rows without dimensions select the default bucket.

Validate prompt keys with explicit membership, type, and stripped non-empty checks. Preserve prompt text exactly after validation.

- [ ] **Step 4: Update the training loop to consume dimensions**

Import `PromptPairDataset`, instantiate it with `(args.train_width, args.train_height)`, and pass `train_dataset.target_resolutions` to the sampler. Replace:

```python
images = batch["pixel_values"].to(...)
bsz = images.shape[0]
h, w = images.shape[2], images.shape[3]
```

with:

```python
student_prompts = batch["student_prompts"]
teacher_prompts = batch["teacher_prompts"]
bsz = len(student_prompts)
h = batch["height"]
w = batch["width"]
```

- [ ] **Step 5: Run loader and builder tests**

Run:

```bash
cd z-image-turbo_self-distill-pe
python3 -m unittest discover -s tests -v
```

Expected: all tests pass.

---

### Task 3: Deterministic Four-Way Validation References

**Files:**
- Modify: `z-image-turbo_self-distill-pe/utils.py`
- Modify: `z-image-turbo_self-distill-pe/train_dopsd.py`
- Create: `z-image-turbo_self-distill-pe/tests/test_validation_generators.py`

**Interfaces:**
- Produces: `create_validation_generators(num_samples: int, base_seed: int) -> list[torch.Generator]`
- Seeds depend only on sample index and base seed, not prompt text.

- [ ] **Step 1: Write failing deterministic-generator test**

```python
first = create_validation_generators(3, 2026)
second = create_validation_generators(3, 2026)
first_draws = [torch.randn(4, generator=g) for g in first]
second_draws = [torch.randn(4, generator=g) for g in second]
for a, b in zip(first_draws, second_draws):
    self.assertTrue(torch.equal(a, b))
```

Also assert that adjacent sample indices produce different streams and invalid `num_samples`/`base_seed` types raise informative exceptions.

- [ ] **Step 2: Run generator test and verify RED**

Run:

```bash
cd z-image-turbo_self-distill-pe
python3 -m unittest tests.test_validation_generators -v
```

Expected: import failure because `create_validation_generators` does not exist.

- [ ] **Step 3: Implement indexed validation generators**

Create one CPU `torch.Generator` per sample, seeded with `base_seed + index`, after validating integer types and positive sample count.

- [ ] **Step 4: Add one-time base p0 and base p1 generation**

Before the training loop, disable adapters and call the pipeline twice:

- original test prompts with freshly created validation generators;
- enhanced test prompts with a second freshly created set using the same seeds.

Save `samples_base_p0.png` and `samples_base_p1.png`. Use `args.train_height` and `args.train_width` for both.

- [ ] **Step 5: Make checkpoint sampling deterministic**

For student(p0) and teacher(p1), create a fresh same-seeded generator list immediately before each pipeline call. Save:

```text
samples_step_<step>_student_p0.png
samples_step_<step>_teacher_p1.png
```

Remove the reused mutable `generator_test` variable and the old ambiguous filenames.

- [ ] **Step 6: Run all tests**

Run:

```bash
cd z-image-turbo_self-distill-pe
python3 -m unittest discover -s tests -v
```

Expected: all tests pass.

---

### Task 4: Configuration, Documentation, and Final Verification

**Files:**
- Modify: `z-image-turbo_self-distill-pe/arguments.py`
- Modify: `z-image-turbo_self-distill-pe/scripts/train_lora_pe.sh`
- Modify: `z-image-turbo_self-distill-pe/README.md`

**Interfaces:**
- Adds CLI options `--train-height` and `--train-width`, both positive integers defaulting to 512.
- Training script selects `dataset/geneval_pe/{train,test}.jsonl` and `p0:p1`.

- [ ] **Step 1: Add argument-validation tests**

Extend the dataset or utility test module with a small pure helper test for positive dimensions, or refactor argument validation into:

```python
def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(...)
    return parsed
```

Test `512` succeeds and `0`, `-1`, and non-integers fail.

- [ ] **Step 2: Run the argument test and verify RED**

Expected: failure because `positive_int` is not defined.

- [ ] **Step 3: Add CLI arguments and update training script**

Add both arguments using `positive_int`, pass explicit 512 values in the script, switch dataset paths to Geneval PE, and set:

```bash
--prompt-key-pairs "p0:p1"
--student-prompt-key "p0"
--teacher-prompt-key "p1"
```

- [ ] **Step 4: Update README**

Document source-to-output construction, generated schema, builder command, fixed-resolution behavior, validation image meanings, training command, and student adapter inference path.

- [ ] **Step 5: Run final verification**

Run:

```bash
cd z-image-turbo_self-distill-pe
python3 -m unittest discover -s tests -v
python3 -m py_compile build_geneval_pe_dataset.py dataset.py dataset_validate.py arguments.py utils.py train_dopsd.py
```

Expected: all tests pass and compilation exits zero.

Validate generated data counts and schema again, then inspect linter diagnostics for every modified Python file. Do not claim GPU runtime success because no training run is included.
