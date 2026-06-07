import torch
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import graph_lib
from model import utils as mutils


def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False, loss_type: str="pretrain") -> function:

    def loss_fn(model, batch, cond=None, t=None, perturbed_batch=None):
        """
        Batch shape: [B, L] int. D given from graph
        """

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
            
        sigma, dsigma = noise(t)
        
        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma)
        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

        loss = (dsigma[:, None] * loss).sum(dim=-1)

        return loss
    
    #-------------------------- new --------------------------
    def loss_fn_sft(model, batch, response_mask, cond=None, t=None, perturbed_batch=None) -> Tensor:
            """
            Batch shape: [B, L] int. D given from graph
            """
    
            if t is None:
                if lv:
                    raise NotImplementedError("Yeah I gotta do this later")
                else:
                    t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
                
            sigma, dsigma = noise(t)
            
            if perturbed_batch is None:
                perturbed_batch = graph.sample_transition(batch, sigma[:, None])
                # 只对 response 位置加噪，prompt 位置保持原始 token
                if response_mask is not None:
                    perturbed_batch = torch.where(response_mask.bool(), perturbed_batch, batch)

            log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
            log_score = log_score_fn(perturbed_batch, sigma)
            loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
            if response_mask is not None:
                loss = loss * response_mask  # mask prompt 位置的 loss
    
            loss = (dsigma[:, None] * loss).sum(dim=-1)
    
            return loss
    #-------------------------- new --------------------------

    if loss_type=="pretrain":
        return loss_fn
    elif loss_type=="sft":
        return loss_fn_sft


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, optimize_fn, accum, loss_type: str) -> function:
    loss_fn = get_loss_fn(noise, graph, train, loss_type)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, response_mask, cond=None) ->Tensor:
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            if loss_type == "pretrain":
                loss = loss_fn(model, batch, cond=cond).mean() / accum
            elif loss_type == "sft":
                loss = loss_fn(model, batch, response_mask, cond=cond).mean() / accum
            
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                if loss_type == "pretrain":
                    loss = loss_fn(model, batch, cond=cond).mean() / accum
                elif loss_type == "sft":
                    loss = loss_fn(model, batch, response_mask, cond=cond).mean() / accum
                ema.restore(model.parameters())

        return loss

    return step_fn