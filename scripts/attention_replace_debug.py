
import argparse, os
import torch


from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from tqdm import tqdm
from collections import defaultdict
import sys
sys.path.append('/coc/flash6/okara7/codes/stable-diff-asyrp')

from ldm.models.diffusion.ddim import DDIMSampler


from utils.constants import CONFIG_SD_PATH, PRETRAINED_PATH, DEPTH_SD_PATH, DEPTH_CONFIG_SD_PATH
import utils.img_utils as iu
import utils.plot_utils as pu
import utils.save_file_utils as sfu
import utils.load_utils as lu
import utils.diffusion_utils_debug as difu
import utils.attention_utils as au

def arg_parser():

    parser = argparse.ArgumentParser()

    # Arguments related to stable diffusion
    
    parser.add_argument("--i_prompt", type=str, default="A man with beard is standing next to the sea")
    parser.add_argument("--g_prompt", type=str, default="A man with beard is standing next to the sea")
    parser.add_argument("--replace_prompt", type=str, default="A man with mustache is standing next to the forest")
    
    parser.add_argument("--init_img_path", type=str, default=None, help="path to the input image")
    
    parser.add_argument("--i_guidance", type=float, default=1.0)
    parser.add_argument("--g_guidance", type=float, default=7.5, help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")

    parser.add_argument("--t_inv", type=int, default=50, help="number of ddim sampling steps in inversion")
    parser.add_argument("--t_gen", type=int, default=50, help="number of ddim sampling steps in generation")

    parser.add_argument("--eta", type=float, default=0.0, help="ddim eta (eta=0.0 corresponds to deterministic sampling")
    parser.add_argument("--batch_size", type=int, default=1, help="how many samples to produce for each given prompt. A.k.a batch size")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--outdir", type=str, help="dir to write results to", default="outputs-replace")

    parser.add_argument("--depth", action='store_true')

    args = parser.parse_args()
    return args

def init_models(configs, ckpt, opt, DEVICE):
    diffusion_model = lu.load_diffusion_model(configs['diffusion_config'], ckpt, DEVICE).eval()
    depth_model = lu.load_depth_model(configs['depth_config'], DEVICE).eval() if opt.is_depth else None
    first_stage_model = lu.load_first_stage_model(configs['first_stage_config'], ckpt, DEVICE).eval()
    first_stage_model.freeze()
    cond_stage_model = lu.load_cond_stage_model(configs['cond_stage_config'], ckpt, DEVICE).eval()

    difu.set_require_grads(diffusion_model)
    
    if opt.is_depth: difu.set_require_grads(depth_model) 
    difu.set_require_grads(cond_stage_model)    
    return diffusion_model, {'depth_model': depth_model, 'first_stage_model': first_stage_model, 'cond_stage_model': cond_stage_model}

def prepare_paths(outpath, g_prompt):
    sample_path = os.path.join(outpath, g_prompt)
    
    os.makedirs(sample_path, exist_ok=True)
    lst = os.listdir(sample_path)
    lst_temp = []
    for d in lst:
        if '.png' in d:
            lst_temp.append(int(d.replace('.png','')))
    base_count = sorted(lst_temp)[-1]+1 if len(lst) !=0 else 0
    
    return sample_path, base_count

def init_latent(opt, x_next=None):
    if opt.is_img2img:
        return x_next  
    else:
        return torch.randn([opt.batch_size, 4, 64, 64], device=opt.device)
def main():
    opt = arg_parser()
    seed = opt.seed
    
    seed_everything(seed)
    sfu.print_args(opt)
    
    opt.is_img2img = True if opt.init_img_path is not None else False
    opt.is_depth = opt.depth if opt.init_img_path is not None else False
    ckpt_path = PRETRAINED_PATH  if not opt.is_depth else DEPTH_SD_PATH
    config_path = CONFIG_SD_PATH if not opt.is_depth else DEPTH_CONFIG_SD_PATH
    midas_img = None
    latents = None
    
    opt.device = torch.device(opt.device_id) if torch.cuda.is_available() else torch.device("cpu")

    configs, ckpt = lu.load_config(config_path, ckpt_path, is_depth=opt.is_depth)  
    
    diffusion_model, models = init_models(configs, ckpt, opt, opt.device)
    
    ddim_ns = argparse.Namespace(**configs['ddim_config'])
    opt.scale_factor = ddim_ns.scale_factor
    sampler = DDIMSampler(ddim_ns, diffusion_model, device=opt.device)
    sample_path, base_count = prepare_paths(opt.outdir, opt.g_prompt)
    
    # controller = au.AttentionStore()


    prompts = [opt.g_prompt, opt.replace_prompt]
    
    
    with torch.no_grad(), autocast("cuda"):
        
        if opt.is_img2img:
            latents, midas_img = difu.inversion(opt, opt.i_prompt, sampler, models)
        latents = init_latent(opt, latents)
        
        controller = None
        au.register_attention_control(diffusion_model, controller)  
        x_generated = difu.generate(opt, opt.g_prompt, sampler, models, latents=latents, midas_img=midas_img, controller=controller)
        x_samples = pu.save_sd_samples(x_generated, sample_path, base_count)
        
        opt.batch_size = len(prompts)
        controller = au.AttentionReplace(prompts, opt, cross_replace_steps={"default_": (0., 1.), "mustache": (0., .1), "forest": (0., 0.6)}, self_replace_steps=0.6)
        au.register_attention_control(diffusion_model, controller)  
        x_generated_2 = difu.generate(opt, prompts, sampler, models, latents=latents, midas_img=midas_img, controller=controller)
  
    
    x_samples = pu.save_sd_samples(x_generated_2, sample_path, base_count+1)
    pu.show_cross_attention(controller, res=16, from_where=["up"], prompts=[opt.g_prompt], save_path=os.path.join(sample_path, f'cross_attn_{str(base_count).zfill(5)}.jpg'))
    pu.show_self_attention(controller, res=16, from_where=["up"], prompts=[opt.g_prompt], save_path=os.path.join(sample_path, f'self_attn_{str(base_count).zfill(5)}.jpg'))
    # pu.save_as_grid(x_samples, n_rows, outpath, grid_count)


if __name__ == "__main__":
    main()
