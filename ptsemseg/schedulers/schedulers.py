import torch

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import MultiStepLR 
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau


class StepLR(_LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    decayed by gamma every step_size epochs. When last_epoch=-1, sets
    initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, decay_iter=1, 
                 gamma=0.9, last_epoch=-1):
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.gamma = gamma
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch % self.decay_iter or self.last_epoch % self.max_iter:
            return [base_lr for base_lr in self.base_lrs]
        else:
            factor = (1 - self.last_epoch / float(self.max_iter)) ** self.gamma
            return [base_lr * factor for base_lr in self.base_lrs] 


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, scheduler, mode='linear', 
                 warmup_iters=100, gamma=0.2, last_epoch=-1):
        self.mode = mode
        self.scheduler = scheduler
        self.warmup_iters = warmup_iters
        self.gamma = gamma
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cold_lrs = self.scheduler.get_lr()

        if self.last_epoch < self.warmup_iters:
            if self.mode == 'linear':
                alpha = self.last_epoch / float(self.warmup_iters)
                factor = self.gamma * (1 - alpha) + alpha

            elif self.mode == 'constant': 
                factor = self.gamma
            else:
                raise KeyError('WarmUp type {} not implemented'.format(self.mode))

            return [factor * base_lr for base_lr in cold_lrs]

        return cold_lrs
