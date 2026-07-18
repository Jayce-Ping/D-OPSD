
CONFIG_FILE="configs/default.yaml"
MAIN_PORT=60213
NUM_PROCS=4
PYTHON_SCRIPT="train_dopsd.py"

# Prompt-enhance teacher (D-OPSD with c_t = f(p1) instead of f_mm(prompt, target_image)).
#   student is conditioned on the short prompt  p0  (e.g. short_en / user_prompt_en)
#   teacher is conditioned on the enhanced prompt p1 (e.g. detailed_en)
# Recommended: --ema-decay 1.0  => teacher stays frozen at the base model
#   (i.e. distill "base(p1)" into "student(p0)"). Use < 1.0 for an EMA teacher.

accelerate launch \
    --config_file       ${CONFIG_FILE} \
    --main_process_port ${MAIN_PORT} \
    --num_processes     ${NUM_PROCS} \
    ${PYTHON_SCRIPT} \
    --deepspeed-config  "configs/z2.json" \
    --output-dir        "exp_results/" \
    --exp-name          "dopsd_pe_geneval_ema1.0_4step_512_lora" \
    --sample-steps      100 \
    --checkpoint-steps  500 \
    --epochs            2001 \
    --max-train-steps   2000 \
    --pretrained_model  "Tongyi-MAI/Z-Image-Turbo" \
    --num-training-steps 4 \
    --train-height 512 \
    --train-width 512 \
    --use-lora 2 \
    --lora-rank 64 \
    --lora-alpha 128 \
    --data-path-train-jsonl "dataset/geneval_pe/train.jsonl" \
    --data-path-test-jsonl  "dataset/geneval_pe/test.jsonl" \
    --prompt-key-pairs "p0:p1" \
    --student-prompt-key "p0" \
    --teacher-prompt-key "p1" \
    --seed   30 \
    --mixed-precision "bf16" \
    --batch-size 1 \
    --batch-size-test 1 \
    --gradient-accumulation-steps 1 \
    --learning-rate-gen 1e-4 \
    --adam-weight-decay 0.0 \
    --enable-gc \
    --vae-dtype "bf16" \
    --ema-decay 1.0 \
