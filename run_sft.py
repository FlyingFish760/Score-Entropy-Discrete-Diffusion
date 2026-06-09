import datetime
import os
import os.path
import gc
from itertools import chain

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf

try:
    from grader import r1_zero_reward_fn
    HAS_GRADER = True
except Exception as _grader_err:   # heavy deps (sympy/math_verify/...) may be absent
    r1_zero_reward_fn = None
    HAS_GRADER = False
    _GRADER_IMPORT_ERR = _grader_err

import data
import losses
import sampling
import graph_lib
import noise_lib
import utils
from data import SFTDataset, cycle_loader
from model import SEDD
from model.ema import ExponentialMovingAverage
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2TokenizerFast, GPT2LMHeadModel


torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
    )


def cleanup():
    dist.destroy_process_group()


def run_multiprocess(rank, world_size, cfg, port):
    try:
        setup(rank, world_size, port)
        _run(rank, world_size, cfg)
    finally:
        cleanup()


def _run(rank, world_size, cfg):
    torch.cuda.set_device(rank)
    work_dir = cfg.work_dir

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    if rank == 0:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))

    # logging
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "logs"))
    def mprint(msg):
        if rank == 0:
            logger.info(msg)

    mprint(work_dir)
    mprint(cfg)

    # wandb logging (rank 0 only). Never let a wandb failure crash training /
    # break the NCCL handshake on the other ranks.
    use_wandb = (rank == 0) and OmegaConf.select(cfg, "wandb.enabled", default=True)
    if use_wandb:
        try:
            wandb.init(
                project=OmegaConf.select(cfg, "wandb.project", default="sedd-sft"),
                entity=OmegaConf.select(cfg, "wandb.entity", default=None),
                # explicit wandb.name wins; otherwise fall back to the run dir name
                name=OmegaConf.select(cfg, "wandb.name", default=None) or getattr(cfg, "wandb_name", None),
                mode=OmegaConf.select(cfg, "wandb.mode", default="online"),
                dir=work_dir,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception as e:
            mprint(f"wandb.init failed ({e}); continuing without wandb logging.")
            use_wandb = False
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.")

    # build token graph
    graph = graph_lib.get_graph(cfg, device)
    
    # build score model
    score_model = SEDD(cfg).to(device)
    score_model = DDP(score_model, device_ids=[rank], static_graph=True, find_unused_parameters=True)

    num_parameters = sum(p.numel() for p in score_model.parameters())
    mprint(f"Number of parameters in the model: {num_parameters}")

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=cfg.training.ema)
    mprint(score_model)
    mprint(f"EMA: {ema}")

    # build noise
    noise = noise_lib.get_noise(cfg).to(device)
    noise = DDP(noise, device_ids=[rank], static_graph=True)
    sampling_eps = 1e-5


    # build optimization state
    optimizer = losses.get_optimizer(cfg, chain(score_model.parameters(), noise.parameters()))
    mprint(f"Optimizer: {optimizer}")
    scaler = torch.cuda.amp.GradScaler()
    mprint(f"Scaler: {scaler}")
    state = dict(optimizer=optimizer, scaler=scaler, model=score_model, noise=noise, ema=ema, step=0) 


    # load in state
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state['step'])

    
    # load in tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Build data iterators
    pad_in_loss = cfg.data.get("pad_in_loss", False)
    train_set = SFTDataset(cfg.data.train, pad_in_loss=pad_in_loss)
    eval_set = SFTDataset(cfg.data.valid, pad_in_loss=pad_in_loss)

    if world_size > 1:
        train_sampler = DistributedSampler(train_set)
        eval_sampler = DistributedSampler(eval_set)
    else:
        train_sampler = None
        eval_sampler = None

    batch_size = cfg.training.batch_size // (cfg.ngpus * cfg.training.accum)
    train_iter = cycle_loader(DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ), train_sampler)
    eval_iter = cycle_loader(DataLoader(
        eval_set,
        batch_size=cfg.eval.batch_size // (cfg.ngpus * cfg.training.accum),
        sampler=eval_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(eval_sampler is None),
    ), eval_sampler)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, cfg.training.accum, loss_type="sft")
    eval_step_fn = losses.get_step_fn(noise, graph, False, optimize_fn, cfg.training.accum, loss_type="sft")


    # NOTE: conditional (SFT) snapshot sampling builds its sampler per-batch
    # inside the loop below, since proj_fun depends on the specific prompts.

    num_train_steps = cfg.training.n_iters
    mprint(f"Starting training loop at step {initial_step}.")


    while state['step'] < num_train_steps + 1:
        step = state['step']


        token_ids, response_mask, _ = next(train_iter)
        token_ids = token_ids.to(device)
        response_mask = response_mask.to(device)
        loss = train_step_fn(state, token_ids, response_mask)

        # flag to see if there was movement ie a full batch got computed
        if step != state['step']:
            if step % cfg.training.log_freq == 0:
                dist.all_reduce(loss)
                loss /= world_size

                mprint("step: %d, training_loss: %.5e" % (step, loss.item()))
                if use_wandb:
                    wandb.log({"train/loss": loss.item(),
                               "train/lr": state['optimizer'].param_groups[0]['lr']},
                              step=step)
            
            if step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                utils.save_checkpoint(checkpoint_meta_dir, state)

            if step % cfg.training.eval_freq == 0:
                eval_token_ids, eval_response_mask, _ = next(eval_iter)
                eval_token_ids = eval_token_ids.to(device)
                eval_response_mask = eval_response_mask.to(device)
                eval_loss = eval_step_fn(state, eval_token_ids, eval_response_mask)

                dist.all_reduce(eval_loss)
                eval_loss /= world_size

                mprint("step: %d, evaluation_loss: %.5e" % (step, eval_loss.item()))
                if use_wandb:
                    wandb.log({"eval/loss": eval_loss.item()}, step=step)

            if step > 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
                # Save the checkpoint.
                save_step = step // cfg.training.snapshot_freq
                if rank == 0:
                    utils.save_checkpoint(os.path.join(
                        checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

                # Generate CONDITIONAL (SFT) samples: pin prompts taken from the
                # eval set and let the model fill only the response positions.
                if cfg.training.snapshot_sampling:
                    mprint(f"Generating conditional samples at step: {step}")

                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                    utils.makedirs(this_sample_dir)

                    # a batch of prompts/responses/answers from the eval set
                    cond_token_ids, cond_response_mask, cond_answers = next(eval_iter)
                    cond_token_ids = cond_token_ids.to(device)
                    cond_response_mask = cond_response_mask.to(device)

                    # pin everything that is NOT a response token (prompt + padding)
                    prompt_mask = (cond_response_mask == 0)

                    def proj_fun(x):
                        x[prompt_mask] = cond_token_ids[prompt_mask]
                        return x

                    sampling_fn = sampling.get_pc_sampler(
                        graph, noise, cond_token_ids.shape,
                        predictor=cfg.sampling.predictor,
                        steps=cfg.sampling.steps,
                        denoise=cfg.sampling.noise_removal,
                        eps=sampling_eps, device=device,
                        proj_fun=proj_fun,
                    )

                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    sample = proj_fun(sampling_fn(score_model))
                    ema.restore(score_model.parameters())

                    gen_sentences = tokenizer.batch_decode(sample)
                    gt_sentences = tokenizer.batch_decode(cond_token_ids)

                    # grade generated responses with r1_zero_reward_fn (format/answer)
                    fmt_sum = 0.0
                    ans_sum = 0.0
                    n_graded = 0

                    file_name = os.path.join(this_sample_dir, f"sample_{rank}.txt")
                    with open(file_name, 'w', encoding="utf-8") as file:
                        for idx, (gen, gt) in enumerate(zip(gen_sentences, gt_sentences)):
                            resp_ids = sample[idx][cond_response_mask[idx].bool()]
                            resp_text = tokenizer.decode(resp_ids)
                            gt_answer = cond_answers[idx]

                            fmt_r = ans_r = 0.0
                            if HAS_GRADER and gt_answer:
                                try:
                                    rew = r1_zero_reward_fn(resp_text, gt_answer)
                                    fmt_r = float(rew["format_reward"])
                                    ans_r = float(rew["answer_reward"])
                                except Exception:
                                    fmt_r = ans_r = 0.0
                                fmt_sum += fmt_r
                                ans_sum += ans_r
                                n_graded += 1

                            file.write(f"[{idx}] GENERATED RESPONSE:\n{resp_text}\n")
                            file.write(f"[{idx}] GT ANSWER: {gt_answer} | "
                                       f"format={fmt_r} answer={ans_r}\n")
                            file.write(f"[{idx}] GENERATED (full):\n{gen}\n")
                            file.write(f"[{idx}] GROUND TRUTH (full):\n{gt}\n")
                            file.write("=" * 92 + "\n")

                    # aggregate format/answer accuracy across ranks (HAS_GRADER is
                    # identical on every rank, so this collective never deadlocks)
                    if HAS_GRADER:
                        agg = torch.tensor([fmt_sum, ans_sum, float(n_graded)], device=device)
                        dist.all_reduce(agg)
                        if agg[2] > 0:
                            format_acc = (agg[0] / agg[2]).item()
                            answer_acc = (agg[1] / agg[2]).item()
                            mprint("step: %d, format_acc: %.4f, answer_acc: %.4f (n=%d)"
                                   % (step, format_acc, answer_acc, int(agg[2].item())))
                            if use_wandb:
                                wandb.log({"eval/format_acc": format_acc,
                                           "eval/answer_acc": answer_acc}, step=step)
                    else:
                        mprint(f"grader unavailable ({_GRADER_IMPORT_ERR}); "
                               f"skipping format/answer accuracy.")

                    if cfg.eval.perplexity:
                        with torch.no_grad():
                            eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
                            _, logits = eval_model(sample, labels=sample)[:2]
                            logits = logits.transpose(-1, -2)                    # [B, V, L]
                            token_loss = F.cross_entropy(
                                logits[..., :-1], sample[..., 1:], reduction="none")  # [B, L-1]
                            # only score response positions (shift by 1 for next-token)
                            rmask = cond_response_mask[:, 1:].float()
                            seq_loss = (token_loss * rmask).sum(-1) / rmask.sum(-1).clamp(min=1)
                            total_perplexity = seq_loss.exp().mean()
                            dist.all_reduce(total_perplexity)
                            total_perplexity /= world_size
                            mprint(f"Generative Perplexity at step: {step}. Perplexity: {total_perplexity:.3f}.")
                            if use_wandb:
                                wandb.log({"eval/gen_ppl": total_perplexity.item()}, step=step)

                            del eval_model, logits

                    dist.barrier()

    if use_wandb:
        wandb.finish()
