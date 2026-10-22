"""Explicit chat formatting, document packing, and completion-only labels."""
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "<system>", "<user>", "<assistant>", "<end>"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def records(path):
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected a JSON object")
                yield row
            except (ValueError, TypeError) as e:
                raise ValueError(f"{path}:{line_no}: {e}") from e


def clean_text(text):
    if not isinstance(text, str):
        raise ValueError("text/content must be a string")
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").strip()
    if any(token in text for token in SPECIALS):
        raise ValueError("Raw text must not contain reserved control tokens")
    return text


def texts(paths):
    for path in paths:
        for row in records(path):
            text = clean_text(row["text"])
            if text:
                yield text


def split_for(text, val_fraction, seed):
    digest = hashlib.sha256(text.encode()).digest()
    score = int.from_bytes(hashlib.sha256(str(seed).encode() + digest).digest()[:8], "big") / 2**64
    return "val" if score < val_fraction else "train"


def train_tokenizer(paths, output, vocab_size, val_fraction=0.02, seed=42):
    if vocab_size < 256 + len(SPECIALS):
        raise ValueError("Byte-level BPE needs at least 264 tokens")
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                 special_tokens=SPECIALS,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if Path(output).exists():
        raise ValueError("Tokenizer output already exists")
    # Match prepare_corpus's document split so validation text never trains BPE.
    stream = (text for text in texts(paths) if split_for(text, val_fraction, seed) == "train")
    tok.train_from_iterator(stream, trainer=trainer)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(output))
    return tok.get_vocab_size()


class Codec:
    def __init__(self, path):
        self.path = str(path)
        self.tokenizer = Tokenizer.from_file(self.path)
        self.fingerprint = sha256(path)
        self.special = {t: self.tokenizer.token_to_id(t) for t in SPECIALS}
        if any(v is None for v in self.special.values()):
            raise ValueError("Tokenizer is missing required special tokens")
        self.pad = self.special["<pad>"]
        self.bos = self.special["<bos>"]
        self.eos = self.special["<eos>"]
        self.end = self.special["<end>"]
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text):
        return self.tokenizer.encode(clean_text(text), add_special_tokens=False).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def chat(self, messages, generation=False):
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        ids, labels = [self.bos], [-100]
        previous = None
        for i, message in enumerate(messages):
            role = message["role"]
            if role not in ("system", "user", "assistant"):
                raise ValueError(f"Unsupported role: {role}")
            if role == "system" and i != 0:
                raise ValueError("system message must be first")
            if role == "assistant" and previous != "user":
                raise ValueError("assistant message must follow user")
            if role == "user" and previous not in (None, "system", "assistant"):
                raise ValueError("user message must follow system or assistant")
            content = self.encode(message["content"])
            if not content:
                raise ValueError("Empty chat content")
            marker = self.special[f"<{role}>"]
            ending = self.eos if role == "assistant" else self.end
            ids += [marker] + content + [ending]
            labels += [-100] + (content + [ending] if role == "assistant" else [-100] * (len(content) + 1))
            previous = role
        if generation:
            if previous != "user":
                raise ValueError("Generation prompt must end with user")
            ids += [self.special["<assistant>"]]
            labels += [-100]
        elif previous != "assistant":
            raise ValueError("SFT conversation must end with assistant")
        return ids, labels


def prepare_corpus(paths, tokenizer, output, val_fraction=0.02, seed=42):
    """Hash-split normalized, exact-deduplicated documents before token packing."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    codec = Codec(tokenizer)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Corpus output must be an empty directory")
    seen, counts, docs = set(), {"train": 0, "val": 0}, {"train": 0, "val": 0}
    handles = {split: open(out / f"{split}.bin", "wb") for split in counts}
    try:
        for text in texts(paths):
            digest = hashlib.sha256(text.encode()).digest()
            if digest in seen:
                continue
            seen.add(digest)
            split = split_for(text, val_fraction, seed)
            ids = [codec.bos] + codec.encode(text) + [codec.eos]
            handles[split].write(np.asarray(ids, dtype="<u4").tobytes())
            counts[split] += len(ids)
            docs[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    meta = {"dtype": "uint32-little-endian", "tokenizer_sha256": codec.fingerprint,
            "tokens": counts, "documents": docs, "seed": seed,
            "val_fraction": val_fraction,
            "source_sha256": {str(p): sha256(p) for p in paths}}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not all(docs.values()):
        raise ValueError("Empty split: add more documents or change val_fraction/seed; use a new output directory")
    return meta


class PackedDataset:
    def __init__(self, path, codec, seq_len):
        path = Path(path)
        meta = json.loads((path.parent / "meta.json").read_text())
        if meta["tokenizer_sha256"] != codec.fingerprint:
            raise ValueError("Packed data tokenizer fingerprint mismatch")
        if path.stat().st_size % 4 or path.stat().st_size == 0:
            raise ValueError("Invalid or empty token binary")
        self.tokens = np.memmap(path, dtype="<u4", mode="r")
        self.seq_len = seq_len
        if len(self) < 1:
            raise ValueError("Not enough tokens for one training sequence")

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, index):
        start = index * self.seq_len
        ids = self.tokens[start:start + self.seq_len + 1].astype(np.int64).tolist()
        return ids, ids.copy()


class ChatDataset:
    def __init__(self, path, codec, seq_len, preference=False):
        self.items = []
        for i, row in enumerate(records(path), 1):
            try:
                if preference:
                    messages = row["prompt"]
                    if isinstance(messages, str):
                        messages = [{"role": "user", "content": messages}]
                    prompt, _ = codec.chat(messages, generation=True)
                    pair = []
                    for name in ("chosen", "rejected"):
                        response = codec.encode(row[name])
                        if not response:
                            raise ValueError("Preference response must not be empty")
                        response += [codec.eos]
                        ids = prompt + response
                        pair.append((ids, [-100] * len(prompt) + response))
                    if pair[0][0] == pair[1][0]:
                        raise ValueError("chosen and rejected must differ")
                    item = tuple(pair)
                    sequences = pair
                else:
                    item = codec.chat(row["messages"])
                    sequences = [item]
                if any(len(ids) > seq_len + 1 for ids, _ in sequences):
                    raise ValueError("Sample exceeds seq_len + 1; shorten it explicitly or increase seq_len")
                self.items.append(item)
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(f"{path}, record {i}: {e}") from e
        if not self.items:
            raise ValueError("Dataset is empty")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(items, pad, device):
    # Right padding + causal attention ensures padding cannot affect real tokens.
    length = max(len(ids) - 1 for ids, _ in items)
    x = torch.full((len(items), length), pad, dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, (ids, labels) in enumerate(items):
        n = len(ids) - 1
        x[i, :n] = torch.tensor(ids[:-1])
        y[i, :n] = torch.tensor(labels[1:])
    return x.to(device), y.to(device)


def dataset(path, codec, seq_len, stage):
    if stage in ("pretrain", "cpt"):
        return PackedDataset(path, codec, seq_len)
    return ChatDataset(path, codec, seq_len, preference=stage == "dpo")
