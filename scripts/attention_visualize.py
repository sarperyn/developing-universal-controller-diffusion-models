
import argparse, os
import torch

import numpy as np
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
import utils.diffusion_utils as difu
import utils.attention_utils as au

def arg_parser():

    parser = argparse.ArgumentParser()

    # Arguments related to stable diffusion
    
    parser.add_argument("--inversion_prompt", type=str, default="")
    parser.add_argument("--generation_prompt", type=str, default="A painting of a squirrel eating a burger")
    
    parser.add_argument("--init_img", type=str, default=None, help="path to the input image")
    
    parser.add_argument("--inversion_guidance", type=float, default=0, help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")
    parser.add_argument("--generation_guidance", type=float, default=7.5, help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")

    parser.add_argument("--ddim_steps_encode", type=int, default=50, help="number of ddim sampling steps in inversion")
    parser.add_argument("--ddim_steps_decode", type=int, default=50, help="number of ddim sampling steps in generation")

    parser.add_argument("--eta", type=float, default=0.0, help="ddim eta (eta=0.0 corresponds to deterministic sampling")
    parser.add_argument("--batch_size", type=int, default=1, help="how many samples to produce for each given prompt. A.k.a batch size")


    parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")

    parser.add_argument("--seed", nargs="+", default=[0])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--outdir", type=str, help="dir to write results to", default="outputs")

    parser.add_argument("--depth", action='store_true')

    args = parser.parse_args()
    return args


def main():
    opt = arg_parser()
    opt.seed = list(np.arange(20))
    prompts = ['A painting of a squirrel eating a burger', 'A painting of a lion eating a burger', 'a man walking on the road', 'a man is surfing on the sea', 'a car is being driven on the snow', 'iron man is skiing', 'a cat and a dog playing']
    for seed in opt.seed:
        for prompt in prompts:
            opt.generation_prompt = prompt
            seed_everything(seed)
            sfu.print_args(opt)
            outpath = opt.outdir
            batch_size = opt.batch_size
            inversion_prompt = opt.inversion_prompt
            generation_prompt = opt.generation_prompt
            replace_prompt = opt.replace_prompt
            t_enc = opt.ddim_steps_encode
            t_dec = opt.ddim_steps_decode
            init_img = opt.init_img
            ddim_eta = opt.eta
            inversion_guidance = opt.inversion_guidance
            generation_guidance = opt.generation_guidance
            is_img2img = True if init_img is not None else False
            is_depth = opt.depth
            ckpt_path = PRETRAINED_PATH  if not is_depth else DEPTH_SD_PATH
            config_path = CONFIG_SD_PATH if not is_depth else DEPTH_CONFIG_SD_PATH

            
            DEVICE = torch.device(opt.device_id) if torch.cuda.is_available() else torch.device("cpu")
            configs, ckpt = lu.load_config(config_path, ckpt_path, is_depth=is_depth)   
            
            diffusion_model = lu.load_diffusion_model(configs['diffusion_config'], ckpt, DEVICE).eval()
            depth_model = lu.load_depth_model(configs['depth_config'], DEVICE).eval() if is_depth else None
            first_stage_model = lu.load_first_stage_model(configs['first_stage_config'], ckpt, DEVICE).eval()
            first_stage_model.freeze()
            cond_stage_model = lu.load_cond_stage_model(configs['cond_stage_config'], ckpt, DEVICE).eval()

            difu.set_require_grads(diffusion_model)
            
            if is_depth: difu.set_require_grads(depth_model) 
            difu.set_require_grads(cond_stage_model)

            ddim_ns = argparse.Namespace(**configs['ddim_config'])
            sampler = DDIMSampler(ddim_ns, diffusion_model, device=DEVICE)
            sample_path = os.path.join(outpath, generation_prompt)

            os.makedirs(sample_path, exist_ok=True)
            lst = os.listdir(sample_path)

            lst_temp = []
            for d in lst:

                if '.png' in d:

                    lst_temp.append(int(d.replace('.png','')))
            base_count = sorted(lst_temp)[-1]+1 if len(lst) !=0 else 0
            
            grid_count = len(os.listdir(outpath)) - 1

            
            controller = au.AttentionStore()
            
            au.register_attention_control(diffusion_model, controller)
            print(controller.step_store.keys())

            if is_img2img:
                batch = iu.load_batch_sd(init_img, DEVICE, batch_size, is_depth)
                init_latent = difu.get_first_stage_encoding(first_stage_model.encode(batch['jpg']), ddim_ns.scale_factor)  # move to latent space

                
            precision_scope = autocast if opt.precision == "autocast" else nullcontext
            
            with torch.no_grad(), precision_scope("cuda"):
                uncond = difu.get_learned_conditioning(cond_stage_model, batch_size * [""])
                cond_inversion = difu.get_learned_conditioning(cond_stage_model, batch_size * [inversion_prompt])
                cond_generation = difu.get_learned_conditioning(cond_stage_model, batch_size * [generation_prompt])

                if is_depth:
                    ### DEPTH START ###
                    c_cat = difu.prepare_depth_condition(load_path=None, device=DEVICE, batch=batch, depth_model=depth_model, latent_shape=init_latent.shape[2:])
                    cond_inversion = {"c_concat": [c_cat], "c_crossattn": [cond_inversion]}
                    cond_generation = {"c_concat": [c_cat], "c_crossattn": [cond_generation]}
                    uncond = {"c_concat": [c_cat], "c_crossattn": [uncond]}
                    ### DEPTH END ###

                conds_inversion = [uncond, cond_inversion]
                conds_generation = [uncond, cond_generation]
                
                if is_img2img:
                    ### ENCODE START ###
                    sampler.make_schedule(ddim_num_steps=t_enc, ddim_eta=ddim_eta, verbose=False)
                    x_next = init_latent

                    for i in tqdm(range(t_enc), desc='Encoding Image'):
                        x_next = sampler.ddim_step(x_next.detach(), i, conds_inversion, inversion_guidance, inverse=True)
                    
                    ### ENCODE FINISH ###



                z_enc = x_next if is_img2img else torch.randn([batch_size, 4, 64, 64], device=DEVICE)
                ### DECODE START ###
                sampler.make_schedule(ddim_num_steps=t_dec, ddim_eta=ddim_eta, verbose=False)

                print(f"Running DDIM Sampling with {t_dec} timesteps")
                x_next = z_enc.clone()

                for i in tqdm(range(t_dec), desc='Decoding image'):
                    index = t_dec - i - 1            
                    x_next = sampler.ddim_step(x_next.detach(), index, conds_generation, generation_guidance, inverse=False)
 
                x_samples = difu.decode_first_stage(x_next, ddim_ns.scale_factor, first_stage_model)
                for key,val in (controller.step_store.items()):
                    if len(val) != 0:
                        print(len(val), val[0].size())
            x_samples = pu.save_sd_samples(x_samples, sample_path, base_count)
            pu.show_cross_attention(controller, res=16, from_where=["up"], prompts=[generation_prompt], save_path=os.path.join(sample_path, f'cross_attn_{str(base_count).zfill(5)}.jpg'))
            pu.show_self_attention(controller, res=16, from_where=["up"], prompts=[generation_prompt], save_path=os.path.join(sample_path, f'self_attn_{str(base_count).zfill(5)}.jpg'))
            # pu.save_as_grid(x_samples, n_rows, outpath, grid_count)


if __name__ == "__main__":
    main()
