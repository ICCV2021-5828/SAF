
import numpy as np

from .. import cfg


def sigmoid_increasing_weight(
    min_value=cfg.args.MIXUP_low,
    max_value=cfg.args.MIXUP_high,
    max_iter=cfg.args.MIXUP_max
):
    '''base: 2/(1+e^(-x)) - 1'''
    '''base: 1 - 2/(e^(x)+1)'''
    '''base: (e^(x)-1) / (e^(x)+1)'''
    if min_value == max_value:
        return min_value

    def sigmoid_increase(x):
        return 2 / (1 + np.exp(-x)) - 1

    delta = max_value - min_value
    x = cfg.iter_count / max_iter
    return min_value + (delta) * sigmoid_increase(x)


def tanh_increasing_weight(
    min_value=cfg.args.MIXUP_low,
    max_value=cfg.args.MIXUP_high,
    max_iter=cfg.args.MIXUP_max,
    N=2,
):
    '''base: tanh(x/2)'''
    if min_value == max_value:
        return min_value

    delta = max_value - min_value
    x = cfg.iter_count / max_iter
    return min_value + (delta) * np.tanh(x / N)


def arctan_increasing_weight(
    min_value=cfg.args.MIXUP_low,
    max_value=cfg.args.MIXUP_high,
    max_iter=cfg.args.MIXUP_max,
    N=2,
):
    '''base: tanh(x/2)'''
    if min_value == max_value:
        return min_value

    delta = max_value - min_value
    x = cfg.iter_count / max_iter
    return min_value + (delta) * np.arctan(x / N) / np.pi


def linear_increasing_weight(
    min_value=cfg.args.MIXUP_low,
    max_value=cfg.args.MIXUP_high,
    max_iter=cfg.args.MIXUP_max
):
    '''base: x'''
    if min_value == max_value:
        return min_value

    if cfg.iter_count >= max_iter:
        return max_value

    delta = max_value - min_value
    x = cfg.iter_count / max_iter
    return min_value + (delta) * (x)


def inverse_increasing_weight(
    min_value=cfg.args.MIXUP_low,
    max_value=cfg.args.MIXUP_high,
    max_iter=cfg.args.MIXUP_max,
    N=1,
):
    '''base: (x)/(x+1)'''
    if min_value == max_value:
        return min_value

    delta = max_value - min_value
    x = cfg.iter_count / max_iter
    return min_value + (delta) * (1 - 1 / (x + N))
