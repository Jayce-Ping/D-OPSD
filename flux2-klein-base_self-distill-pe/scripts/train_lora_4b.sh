CONFIG_FILE="configs/default.yaml"
MAIN_PORT=60214
NUM_PROCS=8
PYTHON_SCRIPT="train_dopsd.py"

# Prompt-enhance teacher on FLUX.2-Klein base (D-OPSD with c_t = f(p1) instead of the edit branch).
#   student is conditioned on the original prompt  p0
#   teacher is conditioned on the enhanced prompt  p1
# Recommended: --ema-decay 1.0 => teacher stays frozen at the base model (distill base(p1) -> student(p0)).

accelerate launch \
    --config_file      ${CONFIG_FILE} \
    --main_process_port ${MAIN_PORT} \
    --num_processes     ${NUM_PROCS} \
    ${PYTHON_SCRIPT} \
    --deepspeed-config  "configs/z2.json" \
    --output-dir     "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/d_opsd" \
    --exp-name     "dopsd_pe_teacher_ema1.0_40step_4b_geneval_bsz1_lora_lr2e-5" \
    --sample-steps      100 \
    --checkpoint-steps  500 \
    --epochs              3001 \
    --max-train-steps   3001 \
    --pretrained_model   "black-forest-labs/FLUX.2-klein-base-4B" \
    --num-training-steps  40 \
    --use-lora 2 \
    --lora-rank 64 \
    --lora-alpha 128 \
    --data-path-train-jsonl "dataset/geneval_pe/train.jsonl" \
    --data-path-test-jsonl "dataset/geneval_pe/test.jsonl" \
    --prompt-key-pairs "p0:p1" \
    --student-prompt-key "p0" \
    --teacher-prompt-key "p1" \
    --train-height 512 \
    --train-width 512 \
    --seed   30 \
    --mixed-precision "bf16" \
    --batch-size 4 \
    --batch-size-test 1 \
    --gradient-accumulation-steps 1 \
    --learning-rate-gen 2e-5 \
    --adam-weight-decay 0.0 \
    --enable-gc \
    --vae-dtype "bf16" \
    --ema-decay 1.0 \
