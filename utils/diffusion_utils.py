import torch
import os

import utils.save_file_utils as sfu
import utils.img_utils as iu

from tqdm import tqdm
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

def get_first_stage_encoding(encoder_posterior, scale_factor):
    if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        z = encoder_posterior.sample()
    elif isinstance(encoder_posterior, torch.Tensor):
        z = encoder_posterior
    return scale_factor * z


def get_learned_conditioning(cond_stage_model, c, img_token=None):
    if img_token is not None:
        c = cond_stage_model(c, img_token=img_token)
    else:
        c = cond_stage_model(c)
    if isinstance(c, DiagonalGaussianDistribution):
        c = c.mode()
    return c


def decode_first_stage(z, scale_factor, first_stage_model):
    z = 1. / scale_factor * z
    return first_stage_model.decode(z)

def prepare_depth_condition(load_path, device, midas_img=None, depth_model=None, latent_shape=None, save_path=None):
    if midas_img is not None:
        depth_image, cc = iu.img_to_depth(midas_img, depth_model, latent_shape)
        if save_path is not None:
            depth_image.save(save_path)
    if load_path is not None:
        cc = sfu.load_saved_depth(load_path, device)
        
    c_cat = torch.cat([cc], dim=1)
    return c_cat


def prepare_conditions(opt, cond_stage_model, depth_model, prompt, midas_img=None):
    uncond = get_learned_conditioning(cond_stage_model, opt.batch_size * [""])
    cond = get_learned_conditioning(cond_stage_model, opt.batch_size * [prompt]) if not type(prompt) is list else get_learned_conditioning(cond_stage_model, prompt)
    if opt.is_depth:
        ### DEPTH START ###
        c_cat = prepare_depth_condition(load_path=None, device=opt.device, midas_img=midas_img, depth_model=depth_model, latent_shape=(64,64))
        cond = {"c_concat": opt.batch_size * [c_cat], "c_crossattn": [cond]}
        uncond = {"c_concat": opt.batch_size * [c_cat], "c_crossattn": [uncond]}
        ### DEPTH END ###

    return [uncond, cond]


def inversion(opt, sampler, models):
    x_next, midas_img = sampling(opt, sampler, models, is_generation=False)
    return x_next, midas_img


def generate(opt, sampler, models, latents=None, midas_img=None):
    x_next = sampling(opt, sampler, models, is_generation=True, x_next=latents, midas_img=midas_img)
    return x_next

def sampling(opt, sampler, models, is_generation=True, x_next=None, midas_img=None):
    # initialize variables #
    cond_stage_model, depth_model, first_stage_model = models['cond_stage_model'], models['depth_model'], models['first_stage_model']
    (in_img, midas_img) = (None, midas_img) if is_generation else iu.load_batch_sd(opt.init_img_path, opt.device, opt.batch_size, opt.is_depth)
    prompt = opt.g_prompt if is_generation else opt.i_prompt
    ddim_num_steps = opt.t_dec if is_generation else opt.t_enc
    x_next = x_next.clone() if is_generation else get_first_stage_encoding(first_stage_model.encode(in_img), opt.scale_factor) 
    guidance = opt.g_guidance if is_generation else opt.i_guidance
    desc = f'{"Inversion" if not is_generation else "Generation"}, prompt: "{prompt}", scale: {guidance}, timesteps: {ddim_num_steps}'
    
    # prepare conditions #
    conds = prepare_conditions(opt, cond_stage_model, depth_model, prompt, midas_img=midas_img)
    
    # prepare timestep values and alphas #
    sampler.make_schedule(ddim_num_steps=ddim_num_steps, ddim_eta=opt.eta, verbose=False)
    
    # run diffusion process #
    for i in tqdm(range(ddim_num_steps), desc=desc):
        index = opt.t_dec - i - 1 if is_generation else i            
        x_next = sampler.ddim_step(x_next.detach(), index, conds, guidance, inverse=not is_generation)
    
    # decode the latent to pixel space #
    if is_generation: x_next = decode_first_stage(x_next, opt.scale_factor, first_stage_model)
    
    if is_generation:
        return x_next
    else:
        return x_next, midas_img


def process_condition(x_noisy, cond, conditioning_key, sequential_cross_attn):

    c_concat = cond['c_concat'] if isinstance(cond, dict) else None
    c_crossattn = cond['c_crossattn'] if isinstance(cond, dict) else [cond] 

    if conditioning_key == 'crossattn':
        cc = c_crossattn if sequential_cross_attn else torch.cat(c_crossattn, 1)
        
    elif conditioning_key == 'hybrid':
        x_noisy = torch.cat([x_noisy] + c_concat, dim=1)
        cc = torch.cat(c_crossattn, 1)
    return x_noisy, cc

def concat_conditions(uncond, cond):
    if isinstance(cond, dict):
        assert isinstance(uncond, dict)
        c_in = dict()
        for k in cond:
            if isinstance(cond[k], list):
                c_in[k] = [torch.cat([uncond[k][i], cond[k][i]]) for i in range(len(cond[k]))]
            else:
                c_in[k] = torch.cat([uncond[k], cond[k]])
    elif isinstance(cond, list):
        c_in = list()
        assert isinstance(uncond, list)
        for i in range(len(cond)):
            c_in.append(torch.cat([uncond[i], cond[i]]))
    else:
        c_in = torch.cat([uncond, cond])
    return c_in

def set_require_grads(model, val=False):
    for p in model.parameters():
        p.requires_grad = val