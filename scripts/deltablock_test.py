
import argparse, os
import torch

from glob import glob
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from tqdm import tqdm
from collections import defaultdict
import sys
import einops
sys.path.append('/home/okara7/codes2/sd-asyrp')

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
import numpy as np

from utils.constants import CONFIG_SD_PATH, PRETRAINED_PATH, DEPTH_SD_PATH, DEPTH_CONFIG_SD_PATH
import utils.img_utils as iu
import utils.plot_utils as pu
import utils.save_file_utils as sfu
import utils.load_utils as lu



def arg_parser():

    parser = argparse.ArgumentParser()

    # Arguments related to stable diffusion
    parser.add_argument("--prompt", type=str, default="", help="the prompt to render")
    parser.add_argument("--init_img", type=str, default=None, help="path to the input image")
    parser.add_argument("--get_all",  action='store_true')

    
    parser.add_argument("--decoder_scale", type=float, default=7.5, help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")

    parser.add_argument("--ddim_steps_encode", type=int, default=50, help="number of ddim sampling steps in inversion")
    parser.add_argument("--ddim_steps_decode", type=int, default=50, help="number of ddim sampling steps in generation")

    parser.add_argument("--eta", type=float, default=0.0, help="ddim eta (eta=0.0 corresponds to deterministic sampling")
    parser.add_argument("--batch_size", type=int, default=1, help="how many samples to produce for each given prompt. A.k.a batch size")

    parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--outdir", type=str, help="dir to write results to", default="outputs/training_results")

    parser.add_argument("--depth", action='store_true')
    parser.add_argument("--save_depth", action='store_true')
    parser.add_argument("--init_depth", type=str, default=None, help="path to the depth cc")

    parser.add_argument("--asyrp", action='store_true')
    parser.add_argument('--delta_block', action='store_true')
    parser.add_argument('--asyrp_idx', type=int, default=20)
    parser.add_argument('--save_h_space', action='store_true')
    parser.add_argument('--delta_gama', type=float, default=1.0)

    parser.add_argument('--class1_path', type=str, default=None)
    parser.add_argument('--class2_path', type=str, default=None)

    args = parser.parse_args()
    if args.decoder_scale == 0.0:
        args.prompt = ""
    return args


def get_first_stage_encoding(encoder_posterior, scale_factor):
    if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        z = encoder_posterior.sample()
    elif isinstance(encoder_posterior, torch.Tensor):
        z = encoder_posterior
    return scale_factor * z


def get_learned_conditioning(cond_stage_model, c):
    if hasattr(cond_stage_model, 'encode') and callable(cond_stage_model.encode):
        c = cond_stage_model.encode(c)
        if isinstance(c, DiagonalGaussianDistribution):
            c = c.mode()
    else:
        c = cond_stage_model(c)
    return c


def decode_first_stage(z, scale_factor, first_stage_model):
    z = 1. / scale_factor * z
    return first_stage_model.decode(z)


def calculate_delta_h_extra(current_class_path, target_class_path, iter, device):
        current_class = torch.load(os.path.join(current_class_path, 'h_space', f'{str(iter).zfill(5)}.pt'), map_location='cpu').to(device) #current class
        target_class = torch.load(os.path.join(target_class_path, 'h_space', f'{str(iter).zfill(5)}.pt'), map_location='cpu').to(device) #target class

        return target_class - current_class

def save_h_sp(middle_h, outpath, i):
    h_space_mean = torch.mean(middle_h, 0) # Calculates batch mean
    os.makedirs(os.path.join(outpath, 'h_space'), exist_ok=True)
    h_space_path = os.path.join(outpath, 'h_space', f'{str(i).zfill(5)}.pt')
    torch.save(h_space_mean, h_space_path)


def main():
    opt = arg_parser()
    sfu.print_args(opt)
    outpath = opt.outdir
    batch_size = opt.batch_size
    n_rows = opt.batch_size
    prompt = opt.prompt
    t_enc = opt.ddim_steps_encode
    t_dec = opt.ddim_steps_decode
    ddim_eta = opt.eta
    decoder_scale = opt.decoder_scale

    init_img = opt.init_img
    is_img2img = True if init_img is not None else False
    get_all = opt.get_all

    is_depth = opt.depth
    init_depth = opt.init_depth
    save_depth = opt.save_depth

    asyrp_flag = opt.asyrp
    delta_block = opt.delta_block
    asyrp_idx = opt.asyrp_idx
    delta_gama = opt.delta_gama
    save_h_space = opt.save_h_space

    class1_path = opt.class1_path
    class2_path = opt.class2_path

    ckpt_path = PRETRAINED_PATH  if not is_depth else DEPTH_SD_PATH
    config_path = CONFIG_SD_PATH if not is_depth else DEPTH_CONFIG_SD_PATH

    gama_list = [1000, 100, 30, 20, 10, 5, 1, -1, -5, -10, -20, -30, -100, -1000]
    

    seed_everything(opt.seed)
    DEVICE = torch.device(opt.device_id) if torch.cuda.is_available() else torch.device("cpu")
    configs, ckpt = lu.load_config(config_path, ckpt_path, is_depth=is_depth)    

    diffusion_model = lu.load_diffusion_model(configs['diffusion_config'], ckpt, DEVICE).eval()
    depth_model = lu.load_depth_model(configs['depth_config'], DEVICE).eval() if is_depth else None
    first_stage_model = lu.load_first_stage_model(configs['first_stage_config'], ckpt, DEVICE).eval()
    first_stage_model.freeze()
    cond_stage_model = lu.load_cond_stage_model(configs['cond_stage_config'], ckpt, DEVICE).eval()
    
    diffusion_model.set_deltablock_layer()
    diffusion_model.deltablock.to(DEVICE)

    ddim_ns = argparse.Namespace(**configs['ddim_config'])
    sampler = DDIMSampler(ddim_ns, diffusion_model, device=DEVICE)
    sample_path = os.path.join(outpath, "samples")
    
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    
    if is_img2img:
        if get_all:
            img_paths = glob(os.path.join(init_img.rsplit('/', 1)[0], '*.png'))
            for idx, path in enumerate(img_paths):
                try:
                    batch['jpg'] = torch.cat((batch['jpg'], iu.load_batch_sd(path, DEVICE, batch_size, is_depth)['jpg']))
                except Exception as e:
                    batch = iu.load_batch_sd(path, DEVICE, batch_size, is_depth)
                
            batch_size = batch['jpg'].size(0)
        else:
            batch = iu.load_batch_sd(init_img, DEVICE, batch_size, is_depth)
        init_latent = get_first_stage_encoding(first_stage_model.encode(batch['jpg']), ddim_ns.scale_factor)  # move to latent space


    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad(), precision_scope("cuda"):
        uc = get_learned_conditioning(cond_stage_model, batch_size * [""])
        c = get_learned_conditioning(cond_stage_model, batch_size * [prompt])

        if is_depth:
            ### DEPTH START ###
            if init_depth is None:
                cc = batch['midas_in']
                depth_image, cc = iu.img_to_depth(cc, depth_model, init_latent.shape[2:])
                if save_depth:
                    depth_path = os.path.join(outpath, 'depth')
                    os.makedirs(depth_path, exist_ok=True)
                    torch.save(cc, os.path.join(depth_path, f"{base_count:05}.pt"))
                    depth_image.save(os.path.join(depth_path, f"{base_count:05}.png"))
            else:
                cc = torch.load(init_depth, map_location='cpu').to(DEVICE)
                cc = torch.cat(batch_size*[cc])
            

            c_cat = torch.cat([cc], dim=1)
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc]}
            ### DEPTH END ###

        if is_img2img:
            ### ENCODE START ###
            sampler.make_schedule(ddim_num_steps=t_enc, ddim_eta=ddim_eta, verbose=False)
            timesteps = sampler.ddim_timesteps
            alphas_next = sampler.ddim_alphas
            alphas = torch.tensor(sampler.ddim_alphas_prev)
            x_next = init_latent

            for i in tqdm(range(t_enc), desc='Encoding Image'):
                t = torch.full((init_latent.shape[0],), timesteps[i], device=DEVICE, dtype=torch.long)

                noise_pred, _, _, _  = sampler.apply_model(x_next.detach(), t, 
                                                        cond if is_depth else c, 
                                                        model_ns=None)

                xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
                weighted_noise_pred = alphas_next[i].sqrt() * ((1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
                x_next = xt_weighted + weighted_noise_pred
            z_enc = x_next
            ### ENCODE FINISH ###

        ### DECODE START ##
        if not is_img2img:
            shape = [batch_size, 4, 64, 64]
            z_enc = torch.randn(shape, device=DEVICE)

        sampler.make_schedule(ddim_num_steps=t_dec, ddim_eta=ddim_eta, verbose=False)
        timesteps_decoder = np.flip(sampler.ddim_timesteps)

        print(f"Running DDIM Sampling with {t_dec} timesteps")
        iterator = tqdm(timesteps_decoder, desc='Decoding image', total=t_dec)

        for idx, gama in enumerate(gama_list):
            x_dec = z_enc

            res_dict = defaultdict(list)

            for i, step in enumerate(iterator):
                index = t_dec - i - 1
                ts = torch.full((x_dec.shape[0],), step, device=DEVICE, dtype=torch.long)
                
                if asyrp_flag:
                    if delta_block:
                        model_ns = {
                            'delta_h_extra': None, 
                            'gamma': delta_gama,
                            'delta_block_flag': True,
                        }
                    else:
                        delta_h_extra = calculate_delta_h_extra(class1_path, class2_path, i, DEVICE)
                        model_ns = {
                            'delta_h_extra': delta_h_extra.to(DEVICE), 
                            'gamma': gama, #delta_gama,
                            'delta_block_flag': False,
                        }
                else:
                    model_ns = None
            

                x_dec, pred_x0, middle_h, delta_h = sampler.p_sample_ddim(x_dec.detach(),
                                            cond if is_depth else c,
                                            ts, 
                                            index=index,
                                            unconditional_guidance_scale=decoder_scale,
                                            unconditional_conditioning= uc_full if is_depth else uc, 
                                            asyrp_flag=True if asyrp_flag and i < asyrp_idx else False, 
                                            modified_flag=False, model_ns=model_ns)
                

                if i == len(iterator):
                    res_dict['x_pred'].append(x_dec)
                else:
                    res_dict['x_pred'].append(x_dec.detach().cpu().numpy())

                res_dict['x0_pred'].append(pred_x0.detach())
                res_dict['middle_h'].append(middle_h.detach().cpu().numpy())
                if delta_h is not None:
                    res_dict['delta_h'].append(delta_h.detach().cpu().numpy())
                
                if save_h_space:
                    save_h_sp(middle_h.clone().detach(), outpath, i)

                
            res_dict['x_pred'][-1] = torch.from_numpy(res_dict['x_pred'][-1]).to(x_dec.device)
            ### DECODE END ###
            samples = res_dict['x_pred'][-1]
            x_samples = decode_first_stage(samples, ddim_ns.scale_factor, first_stage_model)
    
            x_samples = pu.save_sd_samples(x_samples, sample_path, base_count)
            pu.save_as_grid(x_samples, n_rows, outpath, grid_count)
            base_count +=1
            grid_count +=1



if __name__ == "__main__":
    main()
