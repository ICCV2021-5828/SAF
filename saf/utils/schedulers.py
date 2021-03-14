


class INVScheduler(object):
    '''
    Source: MDD
    '''
    def __init__(self, gamma, decay_rate, init_lr):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def next_optimizer(self, group_ratios, optimizer, num_iter):
        lr = self.get_lr(num_iter)
        i = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * group_ratios[i]
            i += 1
        return optimizer

    def get_lr(self, num_iter, N=5):
        return self.init_lr * (1 + self.gamma * num_iter / N) ** (-self.decay_rate)


class InverseDecayScheduler():
    def __init__(self, param_ratio_dict, gamma, decay_rate, init_lr) -> None:
        self.param_ratio_dict = param_ratio_dict
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def get_lr(self, num_iter, N=5):
        return self.init_lr * (1 + self.gamma * num_iter / N) ** (-self.decay_rate)

    def update_optimizer(self, optimizer, num_iter):
        lr = self.get_lr(num_iter)
        for group in optimizer.param_groups:
            group['lr'] = lr * self.param_ratio_dict[group['name']]
