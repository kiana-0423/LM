import copy
from dataclasses import asdict
import json
import math
from pathlib import Path

import pytest
import torch

from hanlm.cli import main
from hanlm.data import Codec, ChatDataset, collate, prepare_corpus, train_tokenizer, PackedDataset
from hanlm.model import DecoderLM, ModelConfig, LoRALinear, add_lora, merge_lora
from hanlm.objectives import token_loss, sequence_logps, dpo_loss
from hanlm import engine

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(7)
    torch.set_num_threads(2)


@pytest.fixture
def model():
    return DecoderLM(ModelConfig(vocab_size=32, dim=32, layers=2, heads=4,
                                 hidden_dim=80, max_seq_len=64))


@pytest.fixture
def codec(tmp_path):
    path = tmp_path / "tokenizer.json"
    train_tokenizer([ROOT / "data/demo/pretrain.jsonl"], path, 768, 0.2)
    return Codec(path)


def test_100m_parameter_count():
    with torch.device("meta"):
        model = DecoderLM(ModelConfig(**json.loads((ROOT / "configs/100m.json").read_text())))
    assert sum(p.numel() for p in model.parameters()) == 97_536_768


def test_causality(model):
    ids = torch.randint(0, 32, (2, 12))
    changed = ids.clone()
    changed[:, 7:] = torch.randint(0, 32, (2, 5))
    torch.testing.assert_close(model(ids)[:, :7], model(changed)[:, :7])


def test_cache_matches_full_sequence(model):
    model.eval()
    ids = torch.randint(0, 32, (2, 12))
    full = model(ids)
    initial, cache = model(ids[:, :6], use_cache=True)
    continuation, cache = model(ids[:, 6:10], past=cache, use_cache=True)
    final, _ = model(ids[:, 10:], past=cache, use_cache=True)
    torch.testing.assert_close(torch.cat([initial, continuation, final], 1), full, atol=2e-6, rtol=1e-5)


def test_lora_freezes_base_and_merge_is_equivalent(model):
    original = copy.deepcopy(model)
    ids = torch.randint(0, 32, (2, 10))
    add_lora(model, rank=4, alpha=8)
    torch.testing.assert_close(model(ids), original(ids))
    assert all(p.requires_grad == (name.endswith(".a") or name.endswith(".b"))
               for name, p in model.named_parameters())
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    token_loss(model(ids), ids).backward()
    optimizer.step()
    assert any(not torch.equal(p, before[name]) for name, p in model.named_parameters() if p.requires_grad)
    assert all(torch.equal(p, before[name]) for name, p in model.named_parameters() if not p.requires_grad)
    logits = model(ids).detach()
    merge_lora(model)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    torch.testing.assert_close(model(ids), logits, atol=2e-6, rtol=1e-5)


def test_checkpointing_lora_backward(model):
    add_lora(model, rank=4, alpha=8)
    model.gradient_checkpointing = True
    ids = torch.randint(0, 32, (2, 10))
    token_loss(model(ids), ids).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in model.named_parameters() if name.endswith(".b"))


def test_tokenizer_roundtrip_and_sft_mask(codec):
    text = "中文测试：春天来了。罕见字龘，符号🙂！"
    assert codec.decode(codec.encode(text)) == text
    messages = [{"role": "system", "content": "请用中文回答。"},
                {"role": "user", "content": "你好。"},
                {"role": "assistant", "content": "你好！"},
                {"role": "user", "content": "什么是模型？"},
                {"role": "assistant", "content": "模型用于预测。"}]
    ids, labels = codec.chat(messages)
    supervised = [label for label in labels if label != -100]
    expected = codec.encode("你好！") + [codec.eos] + codec.encode("模型用于预测。") + [codec.eos]
    assert supervised == expected
    x, y = collate([(ids, labels)], codec.pad, "cpu")
    assert x[0].tolist() == ids[:-1]
    assert y[0].tolist() == labels[1:]


def test_padding_does_not_change_loss(model):
    a = ([1, 5, 7, 4], [-100, -100, 7, 4])
    b = ([1, 2, 3, 4, 5, 6], [-100, -100, -100, 4, 5, 6])
    x, y = collate([a], 0, "cpu")
    expected = token_loss(model(x), y)
    bx, by = collate([a, b], 0, "cpu")
    actual = token_loss(model(bx)[:1], by[:1])
    torch.testing.assert_close(actual, expected)


def test_dpo_formula_and_gradient_direction():
    chosen = torch.tensor([-3.0], requires_grad=True)
    rejected = torch.tensor([-4.0], requires_grad=True)
    refc, refr = chosen.detach(), rejected.detach()
    loss, margin = dpo_loss(chosen, rejected, refc, refr, 0.1)
    assert loss.item() == pytest.approx(math.log(2))
    assert margin.item() == 0
    loss.mean().backward()
    assert chosen.grad.item() < 0 < rejected.grad.item()
    improved, _ = dpo_loss(chosen.detach() + 1, rejected.detach() - 1, refc, refr, 0.1)
    assert improved.item() < loss.item()


def test_sequence_logps_excludes_prompt():
    logits = torch.randn(1, 5, 12)
    labels = torch.tensor([[-100, -100, 2, 3, 4]])
    original = sequence_logps(logits, labels)
    logits[:, :2] += torch.randn(1, 2, 12) * 50
    torch.testing.assert_close(original, sequence_logps(logits, labels))


def test_dpo_reference_has_no_grad(model):
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    before = {k: v.clone() for k, v in reference.state_dict().items()}
    x = torch.randint(0, 32, (4, 8))
    y = x.clone()
    y[:, :3] = -100
    losses, _ = engine.preference_forward(model, reference, (x, y), 0.1)
    losses.mean().backward()
    assert all(p.grad is None for p in reference.parameters())
    assert all(torch.equal(v, before[k]) for k, v in reference.state_dict().items())


def test_packing_dedup_and_tokenizer_identity(codec, tmp_path):
    source = ROOT / "data/demo/pretrain.jsonl"
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_bytes(source.read_bytes() * 2)
    meta = prepare_corpus([duplicate], codec.path, tmp_path / "packed", 0.2)
    assert sum(meta["documents"].values()) == 30
    assert min(meta["documents"].values()) > 0
    data = PackedDataset(tmp_path / "packed/train.bin", codec, 32)
    assert len(data[0][0]) == 33
    codec.fingerprint = "mismatch"
    with pytest.raises(ValueError, match="fingerprint"):
        PackedDataset(tmp_path / "packed/train.bin", codec, 32)


def test_overlong_samples_rejected_and_dpo_prompt_masked(codec):
    with pytest.raises(ValueError, match="exceeds"):
        ChatDataset(ROOT / "data/demo/sft_train.jsonl", codec, 8)
    data = ChatDataset(ROOT / "data/demo/dpo_train.jsonl", codec, 256, preference=True)
    for chosen, rejected in data.items:
        ci = next(i for i, x in enumerate(chosen[1]) if x != -100)
        ri = next(i for i, x in enumerate(rejected[1]) if x != -100)
        assert chosen[0][:ci] == rejected[0][:ri]
        assert chosen[1][-1] == codec.eos


def test_resume_exact_and_checkpoint_load(codec, tmp_path, monkeypatch):
    # Interrupt a four-step schedule after step two, then compare against uninterrupted training.
    corpus = tmp_path / "corpus"
    prepare_corpus([ROOT / "data/demo/pretrain.jsonl"], codec.path, corpus, 0.2)
    cfg = tmp_path / "tiny.json"
    cfg.write_text(json.dumps({"vocab_size": 0, "dim": 16, "layers": 1, "heads": 2,
                              "hidden_dim": 40, "max_seq_len": 32}))
    common = ["train", "--stage", "pretrain", "--tokenizer", codec.path,
              "--train", str(corpus / "train.bin"), "--val", str(corpus / "val.bin"),
              "--config", str(cfg), "--steps", "4", "--warmup-steps", "1",
              "--seq-len", "16", "--batch-size", "2", "--grad-accum", "2",
              "--eval-every", "2", "--save-every", "2", "--threads", "2", "--eval-batches", "1"]
    main(common + ["--output", str(tmp_path / "continuous")])
    save = engine.atomic_save

    def interrupt(payload, path):
        save(payload, path)
        if payload.get("step") == 2 and str(path).endswith("last.pt"):
            raise InterruptedError("simulate interruption")

    monkeypatch.setattr(engine, "atomic_save", interrupt)
    with pytest.raises(InterruptedError):
        main(common + ["--output", str(tmp_path / "interrupted")])
    monkeypatch.setattr(engine, "atomic_save", save)
    checkpoint = tmp_path / "interrupted/last.pt"
    main(common + ["--output", str(tmp_path / "interrupted"), "--resume", str(checkpoint)])
    a, pa = engine.load_model(tmp_path / "continuous/last.pt")
    b, pb = engine.load_model(checkpoint)
    assert pa["step"] == pb["step"] == 4
    assert pa["tokens_seen"] == pb["tokens_seen"]
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)


def test_dpo_lora_resume_preserves_fixed_reference(codec, tmp_path, monkeypatch):
    cfg = ModelConfig(vocab_size=codec.vocab_size, dim=16, layers=1, heads=2,
                      hidden_dim=40, max_seq_len=256)
    base = DecoderLM(cfg)
    init = tmp_path / "init.pt"
    engine.atomic_save({"model_config": asdict(cfg), "model": base.state_dict(),
                        "lora": None, "tokenizer_sha256": codec.fingerprint, "stage": "sft"}, init)
    common = ["train", "--stage", "dpo", "--method", "lora", "--tokenizer", codec.path,
              "--train", str(ROOT / "data/demo/dpo_train.jsonl"),
              "--val", str(ROOT / "data/demo/dpo_val.jsonl"),
              "--steps", "4", "--warmup-steps", "1", "--seq-len", "128",
              "--batch-size", "2", "--grad-accum", "2", "--eval-every", "2",
              "--save-every", "2", "--threads", "2", "--eval-batches", "1"]
    main(common + ["--init", str(init), "--output", str(tmp_path / "continuous")])
    save = engine.atomic_save

    def interrupt(payload, path):
        save(payload, path)
        if payload.get("step") == 2 and str(path).endswith("last.pt"):
            raise InterruptedError("simulate interruption")

    monkeypatch.setattr(engine, "atomic_save", interrupt)
    with pytest.raises(InterruptedError):
        main(common + ["--init", str(init), "--output", str(tmp_path / "interrupted")])
    checkpoint = tmp_path / "interrupted/last.pt"
    middle = torch.load(checkpoint, weights_only=True)
    monkeypatch.setattr(engine, "atomic_save", save)
    main(common + ["--resume", str(checkpoint), "--output", str(tmp_path / "interrupted")])
    a, pa = engine.load_model(tmp_path / "continuous/last.pt")
    b, pb = engine.load_model(checkpoint)
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(pb["reference"][name], middle["reference"][name], rtol=0, atol=0)
    assert pa["best_loss"] == pb["best_loss"]
