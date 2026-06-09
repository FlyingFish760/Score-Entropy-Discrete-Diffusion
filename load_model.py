import os
import torch
from model import SEDD
import utils
from model.ema import ExponentialMovingAverage
import graph_lib
import noise_lib

from omegaconf import OmegaConf

def load_model_hf(dir, device):
    score_model = SEDD.from_pretrained(dir).to(device)
    graph = graph_lib.get_graph(score_model.config, device)
    noise = noise_lib.get_noise(score_model.config).to(device)
    return score_model, graph, noise


def load_model_local(root_dir, device):
    # Accept either a run directory (use checkpoints-meta/checkpoint.pth) OR a
    # specific checkpoint .pth file (e.g. checkpoints/checkpoint_3.pth). In the
    # latter case derive the run dir as the checkpoint's grandparent directory:
    #   <run_dir>/checkpoints[-meta]/<file>.pth  ->  run_dir = dirname(dirname)
    if os.path.isfile(root_dir) or root_dir.endswith(".pth"):
        ckpt_path = root_dir
        run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    else:
        run_dir = root_dir
        ckpt_path = os.path.join(run_dir, "checkpoints-meta", "checkpoint.pth")

    cfg = utils.load_hydra_config_from_run(run_dir)
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    score_model = SEDD(cfg).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)

    loaded_state = torch.load(ckpt_path, map_location=device)

    score_model.load_state_dict(loaded_state['model'])
    ema.load_state_dict(loaded_state['ema'])

    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    return score_model, graph, noise


def load_model(root_dir, device):
    try:
        return load_model_hf(root_dir, device)
    except:
        return load_model_local(root_dir, device)