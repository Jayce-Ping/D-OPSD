# Geneval Prompt-Enhancement Dataset and Validation Design

## Goal

Adapt the paired Geneval datasets for `z-image-turbo_self-distill-pe` so the
original prompt is used as the student condition `p0` and the enhanced prompt
is used as the teacher condition `p1`. Make the PE training path support
text-only datasets, and add deterministic validation images that make student
progress directly comparable to fixed base-model references.

## Source Data

- Original prompts:
  `/Users/bowenping/code/Flow-Factory-Private/dataset/geneval`
- Enhanced prompts:
  `/Users/bowenping/code/Flow-Factory-Private/dataset/geneval_enhanced`
- Splits:
  - train: 33,199 rows in each source
  - test: 553 rows in each source

The sources are currently aligned row by row. For every row,
`geneval.prompt` equals `geneval_enhanced.orig_prompt`.

## Dataset Construction

Add a reproducible builder script under the PE project and generate:

- `z-image-turbo_self-distill-pe/dataset/geneval_pe/train.jsonl`
- `z-image-turbo_self-distill-pe/dataset/geneval_pe/test.jsonl`

Each output row contains:

- `p0`: the original `geneval.prompt`
- `p1`: the enhanced `geneval_enhanced.prompt`
- `tag`, `include`, and `exclude`: retained Geneval metadata
- `source_split`: `train` or `test`
- `source_row`: zero-based row index within that split

The builder fails immediately with an informative exception when:

- either input file is missing;
- split row counts differ;
- either prompt is missing, not a string, or empty;
- `geneval.prompt` does not equal `geneval_enhanced.orig_prompt`;
- duplicate `p0` values occur within a split.

The training script uses `--prompt-key-pairs "p0:p1"` and points train and
validation paths at the generated files.

## Text-Only Prompt-Pair Loader

The PE variant does not use image contents; the current image tensor is only
used to recover batch height and width. Replace that dependency with explicit
resolution metadata in the prompt-pair dataset.

- Add `--train-height` and `--train-width`, both defaulting to `512`.
- Rows with `h*w` or `w*h` continue to use their declared aspect ratio for
  bucketing.
- The configured fixed resolution is added as a dedicated bucket. Rows without
  dimensions use that bucket; existing aspect-ratio buckets remain available
  for rows that declare dimensions.
- Dataset items return validated `student_prompt`, `teacher_prompt`, `height`,
  and `width`.
- Collation requires all items in a batch to have the same dimensions and
  returns the dimensions as integers.
- The training loop reads `h` and `w` from the batch and no longer loads or
  transfers image tensors.

This keeps the existing aspect-ratio behavior for the bundled toy dataset while
making Geneval a native text-only input without dummy images.

## Validation Images

Use paired prompts from the Geneval test split in a stable order.
Validation generation uses the configured `--train-height` and
`--train-width`, which default to 512×512.

Before training, generate exactly once:

- `samples_base_p0.png`: base model conditioned on original prompts
- `samples_base_p1.png`: base model conditioned on enhanced prompts

At every configured sampling checkpoint, generate:

- `samples_step_<step>_student_p0.png`
- `samples_step_<step>_teacher_p1.png`

Base images are generated with adapters disabled. Checkpoint images explicitly
activate the student or teacher adapter. Every branch recreates its generators
from the same per-prompt seeds before sampling, so corresponding grid cells
start from identical noise. Base images are not regenerated because the base
model is fixed.

No `student(p1)` images or combined comparison canvas are included.

## Error Handling

All new data and loader validation follows fail-fast behavior. Exceptions state
the expected condition, received value, source file, split, row number, and
relevant field where applicable. No missing prompt becomes an empty string, and
no broad exception handler substitutes another sample.

## Tests and Verification

Use standard-library `unittest` and test-first development.

Automated tests cover:

1. successful construction of aligned train and test fixtures;
2. row-count mismatch rejection;
3. original/enhanced prompt mismatch rejection;
4. empty and missing prompt rejection;
5. duplicate original prompt rejection;
6. text-only rows receiving the configured default resolution;
7. legacy rows retaining aspect-ratio bucketing;
8. collation returning paired prompts and one validated batch resolution;
9. fresh validation generators reproducing identical initial random streams.

After implementation:

- run the unit test suite;
- build the complete 33,199/553-row dataset;
- validate output counts, schemas, non-empty prompts, uniqueness, and alignment;
- compile modified Python files;
- inspect IDE lint diagnostics.

A full GPU training run is outside this change. The handoff will include a
short smoke-training command for subsequent runtime verification.

## Scope

Included:

- dataset builder and generated Geneval PE JSONL files;
- text-only loader and configurable fixed training resolution;
- training script and README updates;
- deterministic base/student/teacher validation image generation;
- focused unit tests.

Excluded:

- online prompt enhancement;
- changes to the D-OPSD loss or teacher EMA algorithm;
- full model training or quantitative evaluation;
- student-on-enhanced-prompt validation images.
