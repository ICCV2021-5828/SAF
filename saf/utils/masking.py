
import torch

from .. import cfg
from .loss_functions import ConditionalEntropyVector


def mask_logits_wrt_presenting_labels(logits, labels):
    if torch.is_tensor(logits):
        mask = torch.zeros(logits.shape[1])
        mask[labels.unique()] = 1
        mask = mask.repeat((logits.shape[0], 1)).to(cfg.DEVICE)
        masked_logits = logits * mask
    elif isinstance(logits, tuple):
        mask_t = torch.zeros(logits[0].shape[1])
        mask_t[labels.unique()] = 1
        masked_logits = []
        for lg in logits:
            mask = mask_t.repeat((lg.shape[0], 1)).to(cfg.DEVICE)
            masked_lg = lg * mask
            masked_logits.append(masked_lg)
    else:
        raise ValueError

    return masked_logits


def mask_labels_wrt_conditional_entropy(logits, labels):
    # input:
    #  logits
    #  labels: one-hot label
    assert logits.shape == labels.shape
    if (cfg.SAF_ENTROPY_MIN is None) and (cfg.SAF_ENTROPY_MAX is None):
        return labels
    cond_entropy = ConditionalEntropyVector(logits)
    mask = torch.zeros(logits.shape[0], 1)
    if cfg.SAF_ENTROPY_MIN:
        mask[cond_entropy >= cfg.SAF_ENTROPY_MIN,0] = 1
    if cfg.SAF_ENTROPY_MAX:
        mask[cond_entropy <= cfg.SAF_ENTROPY_MAX,0] = 1

    masked_labels = mask * labels

    return masked_labels

