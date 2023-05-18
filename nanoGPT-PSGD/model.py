"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
# import preconditioned_stochastic_gradient_descent as psgd
import torch
from torch.optim.optimizer import Optimizer

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

from torch import Tensor

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
State = Dict[str, Any]
OptFloat = Optional[float]
Nus2 = Tuple[float, float]


import math
import torch
from torch.optim.optimizer import Optimizer, required
class PNM(Optimizer):
    r"""Implements Positive-Negative Momentum (PNM).
    It has be proposed in 
    `Positive-Negative Momentum: Manipulating Stochastic Gradient Noise to Improve 
    Generalization`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        betas (Tuple[float, float], optional): inertia coefficients used for computing
            pn momentum(default: (0.9, 1.))
        weight_decay (float, optional): weight decay (default: 0)
        decoupled (bool, optional): decoupled weight decay or L2 regularization (default: True)
    """

    def __init__(self, params, lr=required, betas=(0.9, 1.), weight_decay=0, decoupled=True):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not  0. <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, decoupled=decoupled)
        super(PNM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PNM, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('decoupled', True)

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
            beta1, beta2 = group['betas']
            

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                # Perform decoupled weight decay or L2 Regularization
                if group['decoupled']:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                else:
                    d_p.add_(p.data, alpha=group['weight_decay'])

                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    
                state['step'] += 1

                param_state = self.state[p]
                if state['step'] == 1:
                    pos_momentum = param_state['pos_momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    neg_momentum = param_state['neg_momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                elif state['step'] % 2 == 1:
                    pos_momentum = param_state['pos_momentum']
                    neg_momentum = param_state['neg_momentum']
                else:
                    neg_momentum = param_state['pos_momentum']
                    pos_momentum = param_state['neg_momentum']
                    
                pos_momentum.mul_(beta1**2).add_(d_p, alpha=1-beta1**2)
                noise_norm = math.sqrt((1+beta2) ** 2 + beta2 ** 2)
                delta_p = pos_momentum.mul(1+beta2).add(neg_momentum, alpha=-beta2).mul(1/noise_norm)
                
                p.add_(delta_p, alpha=-group['lr'])
        return loss

class AdaPNM(Optimizer):
    r"""Implements Adaptive Positive-Negative Momentum.
    It has be proposed in 
    `Positive-Negative Momentum: Manipulating Stochastic Gradient Noise to Improve 
    Generalization`__.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float, float], optional): coefficients used for computing
            running averages of gradient, its square, and pn momentum (default: (0.9,0.999, 1.))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay (default: 0.)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: True)
        decoupled (bool, optional): decoupled weight decay or L2 regularization (default: True)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 1.), eps=1e-8,
                 weight_decay=0., amsgrad=True, decoupled=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0. <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, decoupled=decoupled)
        super(AdaPNM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaPNM, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', True)
            group.setdefault('decoupled', True)

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
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('This optimizer does not support sparse gradients.')

                # Perform decoupled weight decay or L2 Regularization
                if group['decoupled']:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                else:
                    grad.add_(p.data, alpha=group['weight_decay'])

                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of gradient values
                    state['neg_exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2, beta3 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg_sq = state['exp_avg_sq']
                if state['step'] % 2 == 1:
                    exp_avg, neg_exp_avg = state['exp_avg'], state['neg_exp_avg']
                else:
                    exp_avg, neg_exp_avg = state['neg_exp_avg'], state['exp_avg']
                
                exp_avg.mul_(beta1**2).add_(grad, alpha=1 - beta1**2)
                noise_norm = math.sqrt((1+beta3) ** 2 + beta3 ** 2)
                
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                
                pnmomentum = exp_avg.mul(1+beta3).add(neg_exp_avg,alpha=-beta3).mul(1/noise_norm)
                p.addcdiv_(pnmomentum, denom, value=-step_size)
        return loss
    
class SGDW(Optimizer):
    r"""Implements SGDW algorithm.

    It has been proposed in `Decoupled Weight Decay Regularization`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0)
        weight_decay: weight decay (L2 penalty) (default: 0)
        dampening: dampening for momentum (default: 0)
        nesterov: enables Nesterov momentum (default: False)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.SGDW(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/1711.05101

    Note:
        Reference code: https://github.com/pytorch/pytorch/pull/22466
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if momentum < 0.0:
            raise ValueError('Invalid momentum value: {}'.format(momentum))
        if dampening < 0.0:
            raise ValueError('Invalid dampening value: {}'.format(dampening))
        if weight_decay < 0.0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                'Nesterov momentum requires a momentum and zero dampening'
            )
        super(SGDW, self).__init__(params, defaults)

    def __setstate__(self, state: State) -> None:
        super(SGDW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        """Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if p.grad.is_sparse:
                    msg = (
                        'SGDW does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p
                        ).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # Apply momentum
                p.data.add_(d_p, alpha=-group['lr'])

                # Apply weight decay
                if weight_decay != 0:
                    p.data.add_(weight_decay, alpha=-group['lr'])
        return loss

import math
import torch
from torch.optim.optimizer import Optimizer


class Adahessian(Optimizer):
    """Implements Adahessian algorithm.
    It has been proposed in `ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning`.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.15)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hessian_power (float, optional): Hessian power (default: 1). You can also try 0.5. For some tasks we found this to result in better performance.
        single_gpu (Bool, optional): Do you use distributed training or not "torch.nn.parallel.DistributedDataParallel" (default: True)
    """

    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), eps=1e-4,
                 weight_decay=0, hessian_power=1, single_gpu=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(
                    betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(
                    betas[1]))
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError("Invalid Hessian power value: {}".format(hessian_power))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, hessian_power=hessian_power)
        self.single_gpu = single_gpu 
        super(Adahessian, self).__init__(params, defaults)

    def get_trace(self, params, grads):
        """
        compute the Hessian vector product with a random vector v, at the current gradient point,
        i.e., compute the gradient of <gradsH,v>.
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """

        # Check backward was called with create_graph set to True
        for i, grad in enumerate(grads):
            if grad.grad_fn is None:
                raise RuntimeError('Gradient tensor {:} does not have grad_fn. When calling\n'.format(i) +
                           '\t\t\t  loss.backward(), make sure the option create_graph is\n' +
                           '\t\t\t  set to True.')

        v = [2 * torch.randint_like(p, high=2) - 1 for p in params]

        # this is for distributed setting with single node and multi-gpus, 
        # for multi nodes setting, we have not support it yet.
        if not self.single_gpu:
            for v1 in v:
                dist.all_reduce(v1)
        if not self.single_gpu:
            for v_i in v:
                v_i[v_i < 0.] = -1.
                v_i[v_i >= 0.] = 1.

        hvs = torch.autograd.grad(
            grads,
            params,
            grad_outputs=v,
            only_inputs=True,
            retain_graph=True)

        hutchinson_trace = []
        for hv in hvs:
            param_size = hv.size()
            if len(param_size) <= 2:  # for 0/1/2D tensor
                # Hessian diagonal block size is 1 here.
                # We use that torch.abs(hv * vi) = hv.abs()
                tmp_output = hv.abs()

            elif len(param_size) == 4:  # Conv kernel
                # Hessian diagonal block size is 9 here: torch.sum() reduces the dim 2/3.
                # We use that torch.abs(hv * vi) = hv.abs()
                tmp_output = torch.mean(hv.abs(), dim=[2, 3], keepdim=True)
            hutchinson_trace.append(tmp_output)

        # this is for distributed setting with single node and multi-gpus, 
        # for multi nodes setting, we have not support it yet.
        if not self.single_gpu:
            for output1 in hutchinson_trace:
                dist.all_reduce(output1 / torch.cuda.device_count())
        
        return hutchinson_trace

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            gradsH: The gradient used to compute Hessian vector product.
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        params = []
        groups = []
        grads = []

        # Flatten groups into lists, so that
        #  hut_traces can be called with lists of parameters
        #  and grads 
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    groups.append(group)
                    grads.append(p.grad)

        # get the Hessian diagonal

        hut_traces = self.get_trace(params, grads)

        for (p, group, grad, hut_trace) in zip(params, groups, grads, hut_traces):

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of Hessian diagonal square values
                state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)

            exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']

            beta1, beta2 = group['betas']

            state['step'] += 1

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad.detach_(), alpha=1 - beta1)
            exp_hessian_diag_sq.mul_(beta2).addcmul_(hut_trace, hut_trace, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            # make the square root, and the Hessian power
            k = group['hessian_power']
            denom = (
                (exp_hessian_diag_sq.sqrt() ** k) /
                math.sqrt(bias_correction2) ** k).add_(
                group['eps'])

            # make update
            p.data = p.data - \
                group['lr'] * (exp_avg / bias_correction1 / denom + group['weight_decay'] * p.data)

        return loss


@torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    # return x * torch.sigmoid(x)
    # return x * (torch.tanh(F.softplus(x)))

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
# def new_gelu(x):    
#     # Reducing to ReLU when a=0.5 and e=0
#     # Here, we set a-->0.5 from left and e-->0 from right,
#     # where adding eps is to make the derivatives have better rounding behavior around 0. 
#     a = 0.49 
#     e = torch.finfo(torch.float32).eps
#     return (1-a)*x + a*torch.sqrt(x*x + e*e) - a*e
# class Swish(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return x * F.sigmoid(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        # we can override the dropout rate
        if 'dropout' in override_args:
            config_args['dropout'] = override_args['dropout']
        # block_size is always 1024 for GPT model checkpoints
        # if one wants a lower block_size it has to be done through model surgery
        # later, by calling crop_block_size()

        # create a from-scratch initialized minGPT model
        config = GPTConfig(block_size=1024, **config_args)
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers_back(self, weight_decay, learning_rate, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # from adabelief_pytorch import AdaBelief
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        # optimizer = AdaBelief(optim_groups, lr=learning_rate, betas=betas,eps=1e-14)
        # optimizer = SGDW(optim_groups, lr=learning_rate,momentum=0.9,nesterov=True)
        # optimizer = torch.optim.SGD(optim_groups, lr=learning_rate,momentum=0.8,nesterov=False)
        # optimizer = AdaPNM(optim_groups,lr=learning_rate,betas=(0.9,.95,1))
        # optimizer = PNM(optim_groups,lr=learning_rate)
        # optimizer = psgd.UVd(
        #     preconditioner_update_probability=0.1
        #     grad_clip_max_norm=100,
        #     momentum=0.9,
        #     rank_of_approximation=100,
        #     lr_params=0.01,
            
        # )
        return optimizer
    
    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        # optimizer = psgd.UVd(
        #     preconditioner_update_probability=0.1
        #     grad_clip_max_norm=100,
        #     momentum=0.9,
        #     rank_of_approximation=100,
        #     lr_params=0.01,
            
        # )
        return optim_groups

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
