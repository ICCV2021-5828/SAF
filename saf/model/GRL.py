
import numpy as np
import torch

from .. import cfg


class GradientReversalLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        coeff = GradientReversalLayer.get_coeff()
        # if cfg.iter_count % 100 == 0:
        #     cfg.logger.debug(f'GRL: iter={cfg.iter_count}, coeff={coeff:.6f}')
        return -coeff * grad_output

    @staticmethod
    def get_coeff(
        alpha=cfg.args.GRL_alpha, beta=cfg.args.GRL_beta,
        low_value=cfg.args.GRL_low, high_value=cfg.args.GRL_high,
        max_iter=cfg.args.GRL_max,
    ):
        iter_val = cfg.iter_count + cfg.epoch_count * beta
        return np.float(low_value + (high_value - low_value) * \
            (2.0 / (1.0 + np.exp(-alpha * iter_val / max_iter)) - 1)
        )
