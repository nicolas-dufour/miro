"""Lamb optimizer."""

import math

import torch
from torch.optim import Optimizer


class Lamb(Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.
        decouple (bool, optional): the decoupled weight decay (L2 penalty) (default: True)
            If set as True, then "weight_decay" is performed as decoupled weight decay.
            If set as False, then "weight_decay" is performed as L2 regularization.
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        adam=False,
        decouple=True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.adam = adam
        self.decouple = decouple
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Lamb does not support sparse gradients, consider SparseAdam instad."
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2
                # Apply bias to lr to avoid broadcast.
                step_size = group["lr"]

                do_layer_adaptation = (
                    group["layer_adaptation"]
                    if "layer_adaptation" in group
                    else group["weight_decay"] > 0
                )

                adam_step = exp_avg_hat / exp_avg_sq_hat.sqrt().add(group["eps"])
                if group["weight_decay"] != 0 and not self.decouple:
                    # L2 regularization (coupled with learning rate)
                    adam_step.add_(p.data, alpha=group["weight_decay"])
                if do_layer_adaptation:
                    weight_norm = p.data.norm(p=2)
                    adam_norm = adam_step.norm(p=2)
                    trust_ratio = torch.where(
                        weight_norm.ne(0),
                        torch.where(adam_norm.ne(0), weight_norm / adam_norm, 1),
                        1,
                    )
                if self.adam or not do_layer_adaptation:
                    trust_ratio = 1

                # Apply the update
                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

                # Apply decoupled weight decay if enabled
                if group["weight_decay"] != 0 and self.decouple:
                    # Decoupled weight decay (not scaled by learning rate)
                    p.data.mul_(1 - group["weight_decay"])
        return loss
