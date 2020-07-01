import torch
from torch.optim.optimizer import Optimizer


class AtomW(Optimizer):
    r"""Implements Atom algorithm.
        Arguments:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float, optional): learning rate (default: 0.5)
            beta (float, optional): coefficient used for computing
                running averages of gradient (default: 0.9)
            m (float, optional): bound for convexity (default: 1.0)
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-8)
            warmups (int, optional): number of warmup steps (default: 0)
            init_lr (float, optional): initial learning rate for warmup (default: 0.01)
            weight_decay (float, optional): weight decay coefficient (default: 0)
        """

    def __init__(self, params, lr=0.5, beta=0.9, m=1.0, eps=1e-8, warmups=0, init_lr=0.01, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= warmups:
            raise ValueError("Invalid warmup steps: {}".format(warmups))
        if not 0.0 <= init_lr <= lr:
            raise ValueError("Invalid initial learning rate: {}".format(init_lr))
        defaults = dict(lr=lr, beta=beta, m=m, eps=eps, warmups=warmups,
                        init_lr=init_lr, base_lr=lr, weight_decay=weight_decay)
        super(AtomW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AtomW, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['approx_hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Previous update direction
                    state['update'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Calculate current lr
                if state['step'] < group['warmups']:
                    curr_lr = (group['base_lr'] - group['init_lr']) * state['step'] / group['warmups'] + group['init_lr']
                else:
                    curr_lr = group['lr']

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Atom does not support sparse gradients.')

                beta = group['beta']
                m = group['m']
                eps = group['eps']
                exp_avg_grad = state['exp_avg_grad']
                B = state['approx_hessian']
                d_p = state['update']

                bias_prev = 1 - beta ** state['step']
                state['step'] += 1
                bias_curr = 1 - beta ** state['step']
                alpha = (1 - beta) / bias_curr

                denom = d_p.norm(p=4).add(eps)
                d_p.div_(denom)
                v_sq = d_p.mul(d_p)
                delta = (grad - exp_avg_grad).div_(denom).mul_(d_p).sum().mul(-alpha) - B.mul(v_sq).sum()

                # Update B
                B.addcmul_(v_sq, delta)

                # Decay the running average coefficient
                exp_avg_grad.mul_(beta * bias_prev / bias_curr).add_(grad, alpha=alpha)

                denom = B.abs().clamp_(min=m)
                d_p.copy_(exp_avg_grad.div(denom))

                # Perform stepweight decay
                if group['weight_decay'] != 0:
                    d_p.add_(p, alpha=group['weight_decay'])

                p.add_(d_p, alpha=-curr_lr)

        return loss
