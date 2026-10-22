"""Run from repository root: python scripts/demo.py [--steps 8] [--device cpu]."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    p.add_argument("--output", default=None)
    args = p.parse_args()
    if args.steps < 1:
        p.error("steps must be positive")
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output or f"runs/demo-{datetime.now().strftime('%Y%m%d-%H%M%S')}").resolve()
    if out.exists():
        p.error("output already exists; choose a new path")
    out.mkdir(parents=True)

    def run(*arguments):
        cmd = [sys.executable, "-m", "hanlm.cli", *map(str, arguments)]
        print("\n>>> " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=root, check=True)

    tok = out / "tokenizer.json"
    run("tokenizer", "--input", "data/demo/pretrain.jsonl", "--output", tok,
        "--vocab-size", 768, "--val-fraction", 0.2)
    for name in ("pretrain", "cpt"):
        run("prepare", "--input", f"data/demo/{name}.jsonl", "--tokenizer", tok,
            "--output", out / name / "data", "--val-fraction", 0.2)

    def stage(name, objective, train, val, init=None, method="full", lr=3e-4):
        options = ["train", "--stage", objective, "--tokenizer", tok, "--train", train,
                   "--val", val, "--config", root / "configs/tiny.json", "--output", out / name / "model",
                   "--device", args.device, "--steps", args.steps, "--warmup-steps", min(2, args.steps),
                   "--seq-len", 128, "--batch-size", 2, "--grad-accum", 2,
                   "--eval-every", args.steps, "--save-every", args.steps,
                   "--eval-batches", 2, "--method", method, "--lr", lr]
        if init:
            options.extend(["--init", init])
        run(*options)
        return out / name / "model/last.pt"

    pt = stage("pretrain", "pretrain", out / "pretrain/data/train.bin", out / "pretrain/data/val.bin")
    cpt = stage("cpt", "cpt", out / "cpt/data/train.bin", out / "cpt/data/val.bin", pt, lr=1e-4)
    sft = stage("sft", "sft", root / "data/demo/sft_train.jsonl", root / "data/demo/sft_val.jsonl", cpt)
    lora = stage("lora", "sft", root / "data/demo/sft_train.jsonl", root / "data/demo/sft_val.jsonl",
                 sft, method="lora", lr=5e-4)
    merged = out / "merged.pt"
    run("merge", "--checkpoint", lora, "--output", merged)
    dpo = stage("dpo", "dpo", root / "data/demo/dpo_train.jsonl", root / "data/demo/dpo_val.jsonl",
                merged, method="lora", lr=1e-4)
    run("generate", "--checkpoint", dpo, "--tokenizer", tok, "--device", args.device,
        "--prompt", "请解释什么是语言模型。", "--max-new-tokens", 32, "--temperature", 0)
    print(f"\n完整演示已完成：{out}\n示例数据和短训练仅验证流程，不代表模型具有可用的中文能力。")


if __name__ == "__main__":
    main()
