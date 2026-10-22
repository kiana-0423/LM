"""Configuration/schedule checks only: no model creation or training."""
from pathlib import Path
import shlex

import pytest

from hanlm.cli import parser
from hanlm.engine import lr_at


def script_args(name):
    source = (Path(__file__).resolve().parents[1] / "scripts" / name).read_text()
    command = source.split("-m hanlm.cli ", 1)[1].replace("\\\n", "")
    tokens = shlex.split(command)
    assert tokens.pop() == "$@"
    return parser().parse_args(tokens)


@pytest.mark.parametrize("name,stage,lr,steps,warmup,scheduler", [
    ("pretrain.sh", "pretrain", 3e-4, 120000, 50, "cosine"),
    ("cpt.sh", "cpt", 1e-4, 15000, 500, "linear"),
])
def test_training_script_settings(name, stage, lr, steps, warmup, scheduler):
    args = script_args(name)
    assert (args.stage, args.lr, args.steps) == (stage, lr, steps)
    assert (args.seq_len, args.batch_size, args.grad_accum) == (512, 128, 1)
    assert (args.optimizer, args.method) == ("adamw", "full")
    assert (args.warmup_steps, args.lr_scheduler) == (warmup, scheduler)


def test_cpt_warmup_and_linear_decay():
    args = script_args("cpt.sh")
    rates = [lr_at(step, args) for step in range(args.steps)]
    assert rates[0] == pytest.approx(1e-4 / 500)
    assert rates[499] == pytest.approx(1e-4)
    assert rates[500] == pytest.approx(1e-4 * (1 - 1 / 14500))
    assert rates[-1] == 0
    assert all(a < b for a, b in zip(rates[:499], rates[1:500]))
    differences = [b - a for a, b in zip(rates[499:-1], rates[500:])]
    assert all(d == pytest.approx(-1e-4 / 14500, abs=1e-18) for d in differences)


def test_pretraining_keeps_existing_cosine_schedule():
    args = script_args("pretrain.sh")
    assert lr_at(49, args) == pytest.approx(args.lr)
    assert lr_at(50, args) == pytest.approx(args.lr)
    assert lr_at(args.steps - 1, args) == pytest.approx(args.lr * 0.1)
