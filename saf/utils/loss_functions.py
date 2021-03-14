
import torch
import torch.nn.functional as F


def ConditionalEntropy(prediction):
    h = F.softmax(prediction, dim=1) * F.log_softmax(prediction, dim=1)
    return -h.sum(dim=1).mean(dim=0)


def ConditionalEntropyVector(prediction):
    h = F.softmax(prediction, dim=1) * F.log_softmax(prediction, dim=1)
    return -h.sum(dim=1)


def JSDivergence(p_output, q_output, get_softmax=True, dim=1):
    """
    Function that measures JS divergence between target and output logits.
    Source: https://blog.csdn.net/BierOne/article/details/104997938
    """
    if get_softmax:
        p_output = F.softmax(p_output, dim=dim)
        # q_output = F.softmax(q_output, dim=dim)
    log_mean_output = ((p_output + q_output )/2).log()
    return (
        F.kl_div(log_mean_output, p_output, reduction='batchmean') + 
        F.kl_div(log_mean_output, q_output, reduction='batchmean')
    ) / 2


def CrossEntropyDivergence(logits: torch.Tensor, labels: torch.Tensor):
    if len(labels.shape) == 1:
        return F.cross_entropy(logits, labels.long())
    assert logits.shape == labels.shape
    logits = logits.float(); labels = labels.float()
    log_probs = F.log_softmax(logits, dim=1)
    loss = -log_probs * labels
    return loss.sum(dim=1).mean(dim=0)


def ConditionalEntropy_v2(logits: torch.Tensor, temperature: float=1.0):
    h = F.softmax(logits/temperature, dim=1) * F.log_softmax(logits*temperature, dim=1)
    return -h.sum(dim=1).mean(dim=0)


# test
if __name__ == '__main__':
    logit = torch.randint(10, (10, 5)).float().softmax(dim=1)
    l1 = F.one_hot(torch.randint(5, (10,)), num_classes=5)
    l2 = F.one_hot(torch.randint(5, (10,)), num_classes=5)
    label = 0.5 * l1 + 0.5 * l2
