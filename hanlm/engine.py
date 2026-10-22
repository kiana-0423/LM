import copy
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import torch

from .data import Codec, collate, dataset, sha256
from .model import DecoderLM, ModelConfig, add_lora, merge_lora
from .objectives import token_loss, sequence_logps, dpo_loss


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def amp_context(device, precision):
    if precision == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 currently requires a CUDA device with bf16 support; use fp32 on Mac/CPU")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_model(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = DecoderLM(ModelConfig(**payload["model_config"]))
    if payload.get("lora"):
        add_lora(model, **payload["lora"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device), payload


def check_tokenizer(payload, codec):
    if payload["tokenizer_sha256"] != codec.fingerprint:
        raise ValueError("Checkpoint/tokenizer fingerprint mismatch; retain the same tokenizer across stages")


def atomic_save(payload, path):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def preference_forward(model, reference, batch, beta):
    x, y = batch
    size = x.shape[0] // 2
    policy = sequence_logps(model(x), y)
    with torch.no_grad():
        ref = sequence_logps(reference(x), y)
    return dpo_loss(policy[:size], policy[size:], ref[:size], ref[size:], beta)


def make_batch(items, codec, device, stage):
    if stage == "dpo":
        items = [pair[0] for pair in items] + [pair[1] for pair in items]
    return collate(items, codec.pad, device)


@torch.no_grad()
def evaluate(model, reference, data, codec, args, device):
    was_training = model.training
    model.eval()
    total, count, correct = 0.0, 0, 0
    limit = len(data) if args.eval_batches == 0 else min(len(data), args.eval_batches * args.batch_size)
    for start in range(0, limit, args.batch_size):
        items = [data[i] for i in range(start, min(start + args.batch_size, limit))]
        batch = make_batch(items, codec, device, args.stage)
        with amp_context(device, args.precision):
            if args.stage == "dpo":
                losses, margins = preference_forward(model, reference, batch, args.beta)
                total += losses.sum().item()
                count += len(items)
                correct += margins.gt(0).sum().item()
            else:
                x, y = batch
                total += token_loss(model(x), y).item()
                count += y.ne(-100).sum().item()
    model.train(was_training)
    loss = total / count
    metrics = {"eval_loss": loss, "eval_count": count}
    if args.stage == "dpo":
        metrics["eval_preference_accuracy"] = correct / count
    else:
        metrics["perplexity"] = math.exp(min(loss, 80))
    return metrics


def lr_at(step, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    if args.lr_scheduler == "linear":
        remaining = args.steps - step - 1
        return args.lr * max(0.0, remaining / max(1, args.steps - args.warmup_steps))
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps - 1)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def train(args):
    if min(args.steps, args.batch_size, args.grad_accum, args.seq_len,
           args.eval_every, args.save_every, args.threads) < 1:
        raise ValueError("Steps, sizes, intervals, and threads must be positive")
    if args.lr <= 0 or args.beta <= 0 or args.weight_decay < 0 or args.max_grad_norm <= 0:
        raise ValueError("Invalid learning rate, beta, weight decay, or clipping norm")
    if not 0 <= args.warmup_steps <= args.steps or args.eval_batches < 0:
        raise ValueError("Invalid warmup_steps/eval_batches")
    if args.init and args.resume:
        raise ValueError("Use --init for a new stage, --resume for an interrupted stage")
    if args.stage != "pretrain" and not (args.init or args.resume):
        raise ValueError("CPT/SFT/DPO require --init or --resume")
    if args.stage == "pretrain" and (args.init or args.method != "full"):
        raise ValueError("Pretraining starts from scratch with --method full")
    device = select_device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    codec = Codec(args.tokenizer)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError("Output directory is not empty; choose a new directory or use --resume")
    out.mkdir(parents=True, exist_ok=True)
    lora = None
    loaded = None
    if args.init or args.resume:
        model, loaded = load_model(args.resume or args.init)
        check_tokenizer(loaded, codec)
        if args.resume:
            lora = loaded.get("lora")
        else:
            # Initialize a new stage from the effective previous-stage weights.
            merge_lora(model)
    else:
        cfg = json.loads(Path(args.config).read_text())
        if cfg["vocab_size"] == 0:
            cfg["vocab_size"] = codec.vocab_size
        if cfg["vocab_size"] != codec.vocab_size:
            raise ValueError(f"Config vocab_size={cfg['vocab_size']} != tokenizer size={codec.vocab_size}")
        model = DecoderLM(ModelConfig(**cfg))
    if not args.resume and args.method == "lora":
        lora = {"rank": args.lora_rank, "alpha": args.lora_alpha}
        add_lora(model, **lora)
    model.to(device)
    model.gradient_checkpointing = args.gradient_checkpointing
    if args.seq_len > model.cfg.max_seq_len:
        raise ValueError("seq_len exceeds model max_seq_len")
    # Validate precision before spending time preparing data.
    with amp_context(device, args.precision):
        pass
    train_data = dataset(args.train, codec, args.seq_len, args.stage)
    val_data = dataset(args.val, codec, args.seq_len, args.stage)
    if sha256(args.train) == sha256(args.val):
        raise ValueError("Training and validation files must differ")
    reference = None
    if args.stage == "dpo":
        reference = copy.deepcopy(model).eval().requires_grad_(False)
        reference.gradient_checkpointing = False
        if args.resume:
            reference.load_state_dict(loaded["reference"])
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay,
                                 betas=(0.9, 0.95))
    sampler = torch.Generator().manual_seed(args.seed)
    # Resume is deliberately strict: changing the LR horizon/data/device is a new run.
    excluded = {"output", "init", "resume", "tokenizer", "config", "command", "func", "train", "val"}
    signature = {k: v for k, v in vars(args).items() if k not in excluded}
    signature.update(train_sha256=sha256(args.train), val_sha256=sha256(args.val),
                     resolved_device=str(device))
    start, best, tokens_seen = 0, float("inf"), 0
    if args.resume:
        saved_signature = dict(loaded["signature"])
        # Older checkpoints used AdamW and cosine implicitly.
        saved_signature.setdefault("optimizer", "adamw")
        saved_signature.setdefault("lr_scheduler", "cosine")
        if signature != saved_signature:
            raise ValueError("Resume arguments/data differ from checkpoint; use the original arguments")
        optimizer.load_state_dict(loaded["optimizer"])
        sampler.set_state(loaded["sampler_rng"])
        torch.set_rng_state(loaded["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(loaded["device_rng"])
        elif device.type == "mps":
            torch.mps.set_rng_state(loaded["device_rng"])
        start, best = loaded["step"], loaded["best_loss"]
        tokens_seen = loaded["tokens_seen"]
    (out / "tokenizer.json").write_bytes(Path(args.tokenizer).read_bytes())
    (out / "run.json").write_text(json.dumps(signature, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
                      "trainable": sum(p.numel() for p in parameters), "train_samples": len(train_data),
                      "val_samples": len(val_data), "start_step": start}), flush=True)

    def save(step, filename):
        payload = {"format_version": 1, "model_config": asdict(model.cfg), "model": model.state_dict(),
                   "lora": lora, "tokenizer_sha256": codec.fingerprint, "stage": args.stage,
                   "step": step, "optimizer": optimizer.state_dict(), "signature": signature,
                   "best_loss": best, "tokens_seen": tokens_seen, "sampler_rng": sampler.get_state(),
                   "torch_rng": torch.get_rng_state(),
                   "reference": None if reference is None else reference.state_dict()}
        if device.type == "cuda":
            payload["device_rng"] = torch.cuda.get_rng_state_all()
        elif device.type == "mps":
            payload["device_rng"] = torch.mps.get_rng_state()
        atomic_save(payload, out / filename)

    model.train()
    for step in range(start, args.steps):
        tick = time.perf_counter()
        lr = lr_at(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        # Aggregate by actual target count across the entire accumulation window.
        windows = []
        denominator = 0
        step_tokens = 0
        for _ in range(args.grad_accum):
            indices = torch.randint(len(train_data), (args.batch_size,), generator=sampler).tolist()
            items = [train_data[i] for i in indices]
            batch = make_batch(items, codec, "cpu", args.stage)
            windows.append(batch)
            n = batch[1].ne(-100).sum().item()
            denominator += len(items) if args.stage == "dpo" else n
            step_tokens += n
        optimizer.zero_grad(set_to_none=True)
        loss_sum, correct = 0.0, 0
        for x, y in windows:
            x, y = x.to(device), y.to(device)
            with amp_context(device, args.precision):
                if args.stage == "dpo":
                    losses, margin = preference_forward(model, reference, (x, y), args.beta)
                    raw_loss = losses.sum()
                    correct += margin.gt(0).sum().item()
                else:
                    raw_loss = token_loss(model(x), y)
                loss = raw_loss / denominator
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            loss_sum += raw_loss.detach().item()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        tokens_seen += step_tokens
        metrics = {"step": step + 1, "loss": loss_sum / denominator, "lr": lr,
                   "grad_norm": float(grad_norm), "supervised_tokens_seen": tokens_seen,
                   "supervised_tokens_per_second": step_tokens / (time.perf_counter() - tick)}
        if args.stage == "dpo":
            metrics["preference_accuracy"] = correct / denominator
        improved = False
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            metrics.update(evaluate(model, reference, val_data, codec, args, device))
            if metrics["eval_loss"] < best:
                best, improved = metrics["eval_loss"], True
        if improved:
            save(step + 1, "best.pt")
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            save(step + 1, "last.pt")
        with open(out / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)


def evaluate_checkpoint(args):
    if min(args.batch_size, args.seq_len, args.threads) < 1 or args.eval_batches < 0 or args.beta <= 0:
        raise ValueError("Invalid evaluation sizes, threads, or beta")
    device = select_device(args.device)
    torch.set_num_threads(args.threads)
    model, payload = load_model(args.checkpoint, device)
    codec = Codec(args.tokenizer)
    check_tokenizer(payload, codec)
    if args.seq_len > model.cfg.max_seq_len:
        raise ValueError("seq_len exceeds model max_seq_len")
    reference = None
    if args.stage == "dpo":
        if payload.get("reference") is None:
            raise ValueError("DPO evaluation needs a training checkpoint containing the fixed reference")
        reference = copy.deepcopy(model).eval().requires_grad_(False)
        reference.load_state_dict(payload["reference"])
    data = dataset(args.val, codec, args.seq_len, args.stage)
    print(json.dumps(evaluate(model, reference, data, codec, args, device), indent=2))


def export_merged(args):
    model, payload = load_model(args.checkpoint)
    merge_lora(model)
    output = Path(args.output)
    if output.exists():
        raise ValueError("Export path already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Inference/stage-transition artifact, deliberately excludes old optimizer state.
    atomic_save({"format_version": 1, "model_config": asdict(model.cfg), "model": model.state_dict(),
                 "lora": None, "stage": payload["stage"],
                 "tokenizer_sha256": payload["tokenizer_sha256"]}, output)
    print(str(output))


@torch.no_grad()
def generate(args):
    if args.max_new_tokens < 1 or args.temperature < 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        raise ValueError("Invalid generation parameters")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    model, payload = load_model(args.checkpoint, device)
    model.eval()
    codec = Codec(args.tokenizer)
    check_tokenizer(payload, codec)
    if args.raw:
        ids = [codec.bos] + codec.encode(args.prompt)
    else:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        ids, _ = codec.chat(messages, generation=True)
    if len(ids) + args.max_new_tokens > model.cfg.max_seq_len:
        raise ValueError("Prompt + max_new_tokens exceeds model context; reduce one of them")
    x = torch.tensor([ids], dtype=torch.long, device=device)
    cache, output = None, []
    for _ in range(args.max_new_tokens):
        logits, cache = model(x, past=cache, use_cache=True)
        scores = logits[0, -1].float()
        # Prevent generation of role/control tokens; EOS remains available.
        for token_id in codec.special.values():
            if token_id != codec.eos:
                scores[token_id] = -float("inf")
        if args.temperature == 0:
            next_id = scores.argmax().item()
        else:
            scores = scores / args.temperature
            if args.top_k:
                threshold = torch.topk(scores, min(args.top_k, scores.numel())).values[-1]
                scores[scores < threshold] = -float("inf")
            if args.top_p < 1:
                sorted_scores, indices = scores.sort(descending=True)
                remove = sorted_scores.softmax(-1).cumsum(-1) > args.top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                scores[indices[remove]] = -float("inf")
            next_id = torch.multinomial(scores.softmax(-1), 1).item()
        if next_id == codec.eos:
            break
        output.append(next_id)
        x = torch.tensor([[next_id]], device=device)
    print(codec.decode(output))
