import argparse
import json

from .data import prepare_corpus, train_tokenizer
from .engine import train, generate, export_merged, evaluate_checkpoint


def parser():
    p = argparse.ArgumentParser(description="HanLM 中文小语言模型全流程训练")
    commands = p.add_subparsers(dest="command", required=True)
    tok = commands.add_parser("tokenizer", help="从中文文本训练 byte-level BPE")
    tok.add_argument("--input", nargs="+", required=True)
    tok.add_argument("--output", required=True)
    tok.add_argument("--vocab-size", type=int, default=16384)
    tok.add_argument("--val-fraction", type=float, default=0.02)
    tok.add_argument("--seed", type=int, default=42)
    tok.set_defaults(func=lambda a: print(json.dumps({"vocab_size": train_tokenizer(
        a.input, a.output, a.vocab_size, a.val_fraction, a.seed)})))
    prep = commands.add_parser("prepare", help="文档去重、划分并打包预训练/CPT 数据")
    prep.add_argument("--input", nargs="+", required=True)
    prep.add_argument("--tokenizer", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--val-fraction", type=float, default=0.02)
    prep.add_argument("--seed", type=int, default=42)
    prep.set_defaults(func=lambda a: print(json.dumps(prepare_corpus(
        a.input, a.tokenizer, a.output, a.val_fraction, a.seed), ensure_ascii=False, indent=2)))

    def shared(sub):
        sub.add_argument("--tokenizer", required=True)
        sub.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="cpu")
        sub.add_argument("--threads", type=int, default=4)

    def evaluation(sub):
        sub.add_argument("--stage", choices=["pretrain", "cpt", "sft", "dpo"], required=True)
        sub.add_argument("--val", required=True)
        sub.add_argument("--seq-len", type=int, default=128)
        sub.add_argument("--batch-size", type=int, default=2)
        sub.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
        sub.add_argument("--eval-batches", type=int, default=20, help="0 表示完整验证集")
        sub.add_argument("--beta", type=float, default=0.1)

    tr = commands.add_parser("train", help="预训练、CPT、SFT 或 DPO；支持全参数和 LoRA")
    shared(tr)
    evaluation(tr)
    tr.add_argument("--train", required=True)
    tr.add_argument("--config", default="configs/tiny.json")
    tr.add_argument("--output", required=True)
    tr.add_argument("--init", help="上一阶段 checkpoint，重置优化器")
    tr.add_argument("--resume", help="当前阶段 checkpoint，恢复优化器/随机状态/步数")
    tr.add_argument("--method", choices=["full", "lora"], default="full")
    tr.add_argument("--lora-rank", type=int, default=8)
    tr.add_argument("--lora-alpha", type=float, default=16)
    tr.add_argument("--steps", type=int, default=100)
    tr.add_argument("--grad-accum", type=int, default=1)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--optimizer", choices=["adamw"], default="adamw")
    tr.add_argument("--lr-scheduler", choices=["cosine", "linear"], default="cosine",
                    help="warmup 后 cosine 衰减至峰值的 10%%，或 linear 衰减至 0")
    tr.add_argument("--weight-decay", type=float, default=0.1)
    tr.add_argument("--warmup-steps", type=int, default=5)
    tr.add_argument("--max-grad-norm", type=float, default=1.0)
    tr.add_argument("--eval-every", type=int, default=25)
    tr.add_argument("--save-every", type=int, default=25)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--gradient-checkpointing", action="store_true")
    tr.set_defaults(func=train)
    ev = commands.add_parser("eval", help="在独立验证集上评估")
    shared(ev)
    evaluation(ev)
    ev.add_argument("--checkpoint", required=True)
    ev.set_defaults(func=evaluate_checkpoint)
    merge = commands.add_parser("merge", help="合并 LoRA 并导出纯模型权重")
    merge.add_argument("--checkpoint", required=True)
    merge.add_argument("--output", required=True)
    merge.set_defaults(func=export_merged)
    gen = commands.add_parser("generate", help="KV cache 自回归生成")
    shared(gen)
    gen.add_argument("--checkpoint", required=True)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--system", default="")
    gen.add_argument("--raw", action="store_true", help="预训练模型文本续写，不套聊天模板")
    gen.add_argument("--max-new-tokens", type=int, default=64)
    gen.add_argument("--temperature", type=float, default=0.8, help="0 表示贪心解码")
    gen.add_argument("--top-k", type=int, default=40)
    gen.add_argument("--top-p", type=float, default=0.9)
    gen.add_argument("--seed", type=int, default=42)
    gen.set_defaults(func=generate)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
