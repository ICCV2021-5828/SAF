
import numpy as np
import torch
import torch.nn.functional as F

from .. import cfg
from .masking import mask_logits_wrt_presenting_labels


def train_iteration_MDD(model, optimizer, inputs_source, inputs_target, labels_source):
    inputs = torch.cat((inputs_source, inputs_target), dim=0)
    sbs, tbs = inputs_source.size(0), inputs_target.size(0)
    _, outputs, _, outputs_adv = model(inputs)

    if cfg.args.MDD_mask_all or (cfg.args.MDD_mask_classifier and cfg.args.MDD_mask_divergence):
        outputs, outputs_adv = mask_logits_wrt_presenting_labels((outputs, outputs_adv), labels_source)
    elif cfg.args.MDD_mask_classifier:
        outputs = mask_logits_wrt_presenting_labels(outputs, labels_source)
    elif cfg.args.MDD_mask_divergence:
        outputs_adv = mask_logits_wrt_presenting_labels(outputs_adv, labels_source)

    classifier_loss = cfg.class_criterion(outputs.narrow(0, 0, sbs), labels_source)

    target_adv = outputs.max(1)[1]
    target_adv_src = target_adv.narrow(0, 0, sbs)
    target_adv_tgt = target_adv.narrow(0, sbs, tbs)

    classifier_loss_adv_src = cfg.class_criterion(outputs_adv.narrow(0, 0, sbs), target_adv_src)

    logloss_tgt = torch.log((1. - F.softmax(outputs_adv.narrow(0, sbs, tbs), dim=1)).clamp(min=1e-7))
    classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)

    transfer_loss = cfg.args.srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt

    total_MDD_loss = classifier_loss + transfer_loss

    optimizer.zero_grad()
    total_MDD_loss.backward()
    optimizer.step()

    # clean up to save memory
    del outputs
    del outputs_adv
    del target_adv
    del target_adv_src
    del target_adv_tgt
    logloss_tgt = logloss_tgt.detach()  # del logloss_tgt
    classifier_loss = classifier_loss.item()
    classifier_loss_adv_src = classifier_loss_adv_src.item()
    classifier_loss_adv_tgt = classifier_loss_adv_tgt.item()
    transfer_loss = transfer_loss.item()
    total_MDD_loss = total_MDD_loss.item()

    cfg.loss_list.append(f'MDD={total_MDD_loss:.6f}')
    cfg.loss_list.append(f'CLS={classifier_loss:.6f}')
    cfg.loss_list.append(f'TRF={transfer_loss:.6f}')

    # sanity check for loss
    assert np.isfinite(total_MDD_loss)

