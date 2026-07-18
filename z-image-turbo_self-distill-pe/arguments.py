import argparse


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value!r}"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value!r}"
        )
    return parsed


def parse_args():

    parser = argparse.ArgumentParser(description="Training")

    #deepspeed
    parser.add_argument("--deepspeed-config", type=str, default=None, help="Path to deepspeed config file.")
    parser.add_argument("--enable-gc", action=argparse.BooleanOptionalAction, default=False, help="Enable model gradient checkpointing.")

    # logging:
    parser.add_argument("--output-dir", type=str, default="dopsd-exps")
    parser.add_argument("--logging-dir", type=str, default="logs")

    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--sample-steps", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--checkpoint-steps", type=int, default=200000)
    parser.add_argument("--max-train-steps", type=int, default=200000)


    # Gen model
    parser.add_argument("--pretrained_model", type=str, default="z-turbo")
    parser.add_argument("--use-lora",type=float, default=1, help="use if > 1")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--num-training-steps", type=int, default=8, help="number of diffusion steps for training.")
    parser.add_argument("--ema-decay", type=float, default=0.9, help="EMA decay for teacher model.")
    parser.add_argument("--train-height", type=positive_int, default=512,
                        help="Training and validation image height for text-only rows.")
    parser.add_argument("--train-width", type=positive_int, default=512,
                        help="Training and validation image width for text-only rows.")

    #vae
    parser.add_argument("--vae-dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="VAE precision.")

    # dataset
    parser.add_argument("--data-path-train-jsonl", type=str, default="dataset/geneval_pe/train.jsonl", help="Path to the training data jsonl file.")
    parser.add_argument("--data-path-test-jsonl", type=str, default="dataset/geneval_pe/test.jsonl", help="Path to the testing data jsonl file.")
    parser.add_argument("--batch-size", type=int, default=4, help="local batch size.")
    parser.add_argument("--batch-size-test", type=int, default=1, help="local batch size test.")

    # prompt-enhance teacher: student is conditioned on the original short prompt p0,
    # teacher is conditioned on the enhanced prompt p1 = PE(p0). Both go through the
    # SAME text encoder (no VLM / no target image is used as teacher context here).
    parser.add_argument(
        "--prompt-key-pairs",
        type=str,
        default="p0:p1",
        help="Comma-separated `student_key:teacher_key` pairs read from each jsonl row. "
             "student_key -> p0 (short prompt), teacher_key -> p1 (enhanced prompt). "
             "One pair is sampled per batch for length/language variety.",
    )
    parser.add_argument("--student-prompt-key", type=str, default="p0",
                        help="Validation: prompt key used for the student (p0).")
    parser.add_argument("--teacher-prompt-key", type=str, default="p1",
                        help="Validation: prompt key used for the teacher (p1).")

    # precision
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--use-8bit-adam", action=argparse.BooleanOptionalAction, default=False,)

    # optimization
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate-gen", type=float, default=1e-6)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=30)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)



    args = parser.parse_args()

    return args
