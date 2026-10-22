import torch
from torch.nn import functional as F


def token_loss(logits, labels):
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                           labels.reshape(-1), ignore_index=-100, reduction="sum")


def sequence_logps(logits, labels):
    mask = labels.ne(-100)
    if not mask.any(dim=1).all():
        raise ValueError("Every sequence needs at least one supervised token")
    safe = labels.masked_fill(~mask, 0)
    logps = logits.float().log_softmax(-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    # Original DPO sums completion log-probabilities; do not length-normalize.
    return (logps * mask).sum(-1)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    chosen_reward = beta * (policy_chosen - ref_chosen)
    rejected_reward = beta * (policy_rejected - ref_rejected)
    margin = chosen_reward - rejected_reward
    return -F.logsigmoid(margin), margin.detach()
