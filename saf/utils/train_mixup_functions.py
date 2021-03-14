
import numpy as np

from .. import cfg
from .increase_function import sigmoid_increasing_weight
from .loss_functions import CrossEntropyDivergence, ConditionalEntropy_v2


def train_iteration_MIX(model, optimizer, inputs_target):
    naive_logit, naive_label, mix_logit, mix_label = model.SAF_forward(tx=inputs_target)

    entropy_loss = 0
    if cfg.args.conditional_entropy:
        entropy_loss = ConditionalEntropy_v2(naive_logit, temperature=cfg.args.ce_temperature)
    elif cfg.args.naive_entropy:
        entropy_loss = CrossEntropyDivergence(naive_logit, naive_label)
    naive_weight = sigmoid_increasing_weight(cfg.args.MIXUP_NAIVE_low, cfg.args.MIXUP_NAIVE_high, cfg.args.MIXUP_NAIVE_max)
    mixup_loss = CrossEntropyDivergence(mix_logit, mix_label)
    mixup_weight = sigmoid_increasing_weight()
    total_mixup_loss = mixup_weight * mixup_loss + entropy_loss * naive_weight

    optimizer.zero_grad()
    total_mixup_loss.backward()
    optimizer.step()

    mixup_loss = mixup_loss.item()
    total_mixup_loss = total_mixup_loss.item()

    cfg.weight_list.append(f'MIXUP={mixup_weight:.6f}')
    cfg.weight_list.append(f'NAIVE={naive_weight:.6f}')
    cfg.loss_list.append(f'MIX={mixup_loss:.6f}')
    cfg.loss_list.append(f'ENT={entropy_loss:.6f}')
    cfg.loss_list.append(f'TTL={total_mixup_loss:.6f}')

    # sanity check for loss
    assert np.isfinite(total_mixup_loss)

