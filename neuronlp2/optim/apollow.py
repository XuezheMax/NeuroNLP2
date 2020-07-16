import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class ApolloW(Optimizer):
    r"""Implements ApolloW algorithm.
        Arguments:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            rho (float, optional): ratio of learning rate over convexity (default: 0.1)
            beta (float, optional): coefficient used for computing
                running averages of gradient (default: 0.9)
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-8)
            warmups (int, optional): number of warmup steps (default: 0)
            init_lr (float, optional): initial learning rate for warmup (default: 0.01)
            weight_decay (float, optional): weight decay coefficient (default: 0)
        """

    def __init__(self, params, rho=0.1, beta=0.9, eps=1e-8, warmups=100, init_lr=0.01, weight_decay=0):
        if not 0.0 < rho:
            raise ValueError("Invalid rho value: {}".format(rho))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= warmups:
            raise ValueError("Invalid warmup steps: {}".format(warmups))
        if not 0.0 <= init_lr <= 1.0:
            raise ValueError("Invalid initial learning rate: {}".format(init_lr))

        lr = 1.0
        rho = lr / rho
        defaults = dict(lr=lr, base_lr=lr, beta=beta, rho=rho, eps=eps,
                        warmups=warmups, init_lr=init_lr, weight_decay=weight_decay)
        super(ApolloW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ApolloW, self).__setstate__(state)

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

        grad_norms = []
        norm_type = 2
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

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Atom does not support sparse gradients.')

                beta = group['beta']
                exp_avg_grad = state['exp_avg_grad']
                B = state['approx_hessian']
                d_p = state['update']

                state['step'] += 1
                bias_correction = 1 - beta ** state['step']
                alpha = (1 - beta) / bias_correction

                # Update the running average grad
                delta_grad = grad - exp_avg_grad
                exp_avg_grad.add_(delta_grad, alpha=alpha)
                grad_norms.append(exp_avg_grad.norm(p=norm_type))

                denom = d_p.norm(p=4).add(group['eps'])
                d_p.div_(denom)
                v_sq = d_p.mul(d_p)
                delta = delta_grad.div_(denom).mul_(d_p).sum().mul(-alpha) - B.mul(v_sq).sum()

                # Update B
                B.addcmul_(v_sq, delta)

        total_norm = torch.norm(torch.stack(grad_norms), p=norm_type)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Calculate current lr
                if state['step'] < group['warmups']:
                    curr_lr = (group['base_lr'] - group['init_lr']) * state['step'] / group['warmups'] + group['init_lr']
                else:
                    curr_lr = group['lr']

                rho = group['rho']
                exp_avg_grad = state['exp_avg_grad']
                B = state['approx_hessian']
                d_p = state['update']

                # calc direction of parameter updates
                denom = torch.max(B.abs(), total_norm.mul(rho))
                d_p.copy_(exp_avg_grad.div(denom))

                # Perform step weight decay
                if group['weight_decay'] != 0:
                    d_p.add_(p, alpha=group['weight_decay'])

                # Update parameters
                p.add_(d_p, alpha=-curr_lr)

        return loss
