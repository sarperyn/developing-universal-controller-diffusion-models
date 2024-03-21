from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import torch
import os

from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.models.autoencoder import AutoencoderKL
from ldm.modules.encoders.modules import FrozenOpenCLIPEmbedder, FrozenOpenCLIPImageEmbedder
from ldm.modules.midas.api import MiDaSInference

def load_model_config(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    print(f"Loading model from {ckpt_path}")
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    
    model = model.to(device)
    model.eval()
    return config, model

def load_config(config_path, ckpt_path, is_depth=False):
    config = OmegaConf.load(config_path)
    configs = {}
    configs['diffusion_config'] = config['model']['params']['unet_config']['params']
    configs['first_stage_config'] = config['model']['params']['first_stage_config']['params']
    if is_depth:
        configs['depth_config'] = config['model']['params']['depth_stage_config']['params']
    configs['cond_stage_config'] = config['model']['params']['cond_stage_config']['params']
    configs['ddim_config'] = config['model']['params']
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    ckpt = pl_sd["state_dict"]
    
    return configs, ckpt

def load_depth_model(config, device):
    depth_model = MiDaSInference(**config)
    depth_model = depth_model.to(device)
    return depth_model

def load_diffusion_model(config, ckpt, device):
    diffusion_model = UNetModel(**config)
    ckpt = {k.replace("model.diffusion_model.", ""): v for k, v in ckpt.items()}
    _, u = diffusion_model.load_state_dict(ckpt, strict=False)
    diffusion_model = diffusion_model.to(device)
    return diffusion_model

def load_first_stage_model(config, ckpt, device):
    first_stage_model = AutoencoderKL(**config)
    ckpt = {k.replace("first_stage_model.", ""): v for k, v in ckpt.items()}
    _, _ = first_stage_model.load_state_dict(ckpt, strict=False)
    first_stage_model = first_stage_model.to(device)
    return first_stage_model

def load_cond_stage_model(config, ckpt, device):
    cond_stage_model = FrozenOpenCLIPEmbedder(**config)
    ckpt = {k.replace("cond_stage_model.", ""): v for k, v in ckpt.items()}
    _, _ = cond_stage_model.load_state_dict(ckpt, strict=False)
    cond_stage_model = cond_stage_model.to(device)
    return cond_stage_model

def load_img_cond_stage_model(config, ckpt, device):
    cond_stage_model = FrozenOpenCLIPImageEmbedder(**config)
    ckpt = {k.replace("cond_stage_model.", ""): v for k, v in ckpt.items()}
    _, _ = cond_stage_model.load_state_dict(ckpt, strict=False)
    cond_stage_model = cond_stage_model.to(device)
    return cond_stage_model


def load_saved_depth(path, device):
    cc_path_list = path.rsplit('/', 1)
    cc_path = os.path.join(cc_path_list[0], f'depth-{cc_path_list[1]}')
    cc = torch.load(cc_path, map_location=device)
    return cc
