__author__ = 'max'

from collections import defaultdict
from torch.optim.optimizer import Optimizer


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_epoch = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def reset_state(self):
        self.optimizer.state.clear()


class InverseSquareRootScheduler(_LRScheduler):
    """
    Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from zero until the configured learning rate (``--lr``).
    Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.
    During warmup::
      lrs = torch.linspace(0, args.lr, args.warmup_updates)
      lr = lrs[update_num]
    After warmup::
      decay_factor = args.lr * sqrt(args.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """
    def __init__(self, optimizer, warmup_steps, init_lr, last_epoch=-1):
        assert warmup_steps > 0, 'warmup steps should be larger than 0.'
        super(InverseSquareRootScheduler, self).__init__(optimizer, last_epoch)
        self.warmup_steps = float(warmup_steps)
        self.init_lr = init_lr
        self.lr_steps = [(base_lr - init_lr) / warmup_steps for base_lr in self.base_lrs]
        self.decay_factor = self.warmup_steps ** 0.5
        if last_epoch == -1:
            last_epoch = 0
        self.step(last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.init_lr + lr_step * self.last_epoch for lr_step in self.lr_steps]
        else:
            lr_factor = self.decay_factor * self.last_epoch**-0.5
            return [base_lr * lr_factor for base_lr in self.base_lrs]


class ExponentialScheduler(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.
    We also support a warmup phase where we linearly increase the learning rate
    from zero until the configured learning rate (``--lr``).
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        warmup_steps (int): Warmup steps..
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, gamma, warmup_steps, init_lr, last_epoch=-1):
        super(ExponentialScheduler, self).__init__(optimizer, last_epoch)
        self.gamma = gamma
        # handle warmup <= 0
        self.warmup_steps = max(1, warmup_steps)
        self.init_lr = init_lr
        self.lr_steps = [(base_lr - init_lr) / self.warmup_steps for base_lr in self.base_lrs]
        if last_epoch == -1:
            last_epoch = 0
        self.step(last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.init_lr + lr_step * self.last_epoch for lr_step in self.lr_steps]
        else:
            lr_factor = self.gamma ** (self.last_epoch - self.warmup_steps)
            return [base_lr * lr_factor for base_lr in self.base_lrs]
