import torch
import math

class AdamLH(torch.optim.Optimizer):
    """ WD and LR decoupled as per Fabian Schaipp https://fabian-sp.github.io/posts/2024/02/decoupling/"""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), weight_decay=0., eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        self._init_lr = lr
        super(AdamLH, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamLH, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Parameters
        ----------
        closure : LossClosure, optional
            A callable that evaluates the model (possibly with backprop) and returns the loss,
            by default None.

        loss : torch.tensor, optional
            The loss tensor. Use this when the backward step has already been performed.
            By default None.

        Returns
        -------
        (Stochastic) Loss function value.
        """
        for group in self.param_groups:
            lr = group['lr']
            lmbda = group['weight_decay']
            eps = group['eps']
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                # decay 
                p.mul_(1 - lmbda*lr/self._init_lr)

                grad = p.grad
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                step_size = lr / bias_correction1
                update = -step_size * exp_avg / denom
                p.add_(update)
                

def get_param_groups(model, weight_decay):
    """Create param groups with and withou weight_decay."""
    
    # filter out parameters that do not require grad
    named_param_dict = {n: p for n,p in model.named_parameters() if p.requires_grad}

    # filter out parameters with names containing 'bias', 'norm', etc
    decay_params_names = [n for n, p in model.named_parameters() if not getattr(p, '_no_weight_decay', False)] # exclude mamba 'A_log', 'D'
    decay_params_names = [n for n in decay_params_names if "bias" not in n] # exclude bias
    decay_params_names = [n for n in decay_params_names if "norm" not in n] # exclude normalization layers

    decay_params = [p for n, p in named_param_dict.items() if n in decay_params_names]
    no_decay_params = [p for n, p in named_param_dict.items() if n not in decay_params_names]
    
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    return param_groups
