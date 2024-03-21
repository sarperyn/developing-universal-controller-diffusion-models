
import argparse, os
import torch


from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from tqdm import tqdm
from collections import defaultdict
import sys
sys.path.append('/home/okara7/codes2/sd-asyrp')

from ldm.models.diffusion.ddim import DDIMSampler

import numpy as np

from utils.constants import CONFIG_SD_PATH, PRETRAINED_PATH
import utils.img_utils as iu
import utils.plot_utils as pu
import utils.save_file_utils as sfu
import utils.load_utils as lu


def arg_parser():
    parser = argparse.ArgumentParser()

    # Arguments related to stable diffusion
    parser.add_argument("--prompt", type=str, default="a painting of a virus monster playing guitar", help="the prompt to render")
    parser.add_argument("--init_img", type=str, default=None, help="path to the input image")
    
    parser.add_argument("--decoder_scale", type=float, default=7.5, help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")

    parser.add_argument("--ddim_steps_encode", type=int, default=50, help="number of ddim sampling steps in inversion")
    parser.add_argument("--ddim_steps_decode", type=int, default=50, help="number of ddim sampling steps in generation")

    parser.add_argument("--eta", type=float, default=0.0, help="ddim eta (eta=0.0 corresponds to deterministic sampling")
    parser.add_argument("--batch_size", type=int, default=4, help="how many samples to produce for each given prompt. A.k.a batch size")


    parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--outdir", type=str, help="dir to write results to", default="outputs/training_results")

    args = parser.parse_args()
    if args.decoder_scale == 0.0:
        args.prompt = ""
    return args

def main():
    opt = arg_parser()
    sfu.print_args(opt)
    outpath = opt.outdir
    batch_size = opt.batch_size
    n_rows = opt.batch_size
    prompt = opt.prompt
    t_enc = opt.ddim_steps_encode
    t_dec = opt.ddim_steps_decode
    init_img = opt.init_img
    ddim_eta = opt.eta
    decoder_scale = opt.decoder_scale


    seed_everything(opt.seed)
    DEVICE = torch.device(opt.device_id) if torch.cuda.is_available() else torch.device("cpu")
    _, model = lu.load_model_config(CONFIG_SD_PATH, PRETRAINED_PATH, DEVICE)
    sampler = DDIMSampler(model)
    sample_path = os.path.join(outpath, "samples")
    
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1
    video_frames = [] # --> list of torch video frames
    for init_image in video_frames:
        # init_image = iu.load_img_sd(init_img, batch_size).to(DEVICE)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        
        with torch.no_grad(), precision_scope("cuda"), model.ema_scope():

            uc = None
            if decoder_scale != 1.0:
                uc = model.get_learned_conditioning(batch_size * [""])

            c = model.get_learned_conditioning(batch_size * [prompt])

            ### ENCODE START ###
            sampler.make_schedule(ddim_num_steps=t_enc, ddim_eta=ddim_eta, verbose=False)
            timesteps = sampler.ddim_timesteps
            alphas_next = sampler.ddim_alphas
            alphas = torch.tensor(sampler.ddim_alphas_prev)
            x_next = init_latent

            for i in tqdm(range(t_enc), desc='Encoding Image'):
                t = torch.full((init_latent.shape[0],), timesteps[i], device=DEVICE, dtype=torch.long)

                noise_pred, _, _, _  = sampler.model.apply_model(x_next, t, c, model_ns=None)

                xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
                weighted_noise_pred = alphas_next[i].sqrt() * ((1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
                x_next = xt_weighted + weighted_noise_pred
            z_enc = x_next
            ### ENCODE FINISH ###

    x_dec_t0 = video_frames_inverses[0]
    for i, step in enumerate(iterator):
        index = t_dec - i - 1
        ts = torch.full((x_dec_t0.shape[0],), step, device=DEVICE, dtype=torch.long)

        x_dec_t0, pred_x0_t0, middle_h_t0, delta_h_t0 = sampler.p_sample_ddim(x_dec_t0, c, ts, index=index,
                                    unconditional_guidance_scale=decoder_scale,
                                    unconditional_conditioning=uc, asyrp_flag=False, 
                                    modified_flag=False, model_ns=None)
        # LIST

    video_frames_inverses = [] # --> list of torch video frames inverse
    for ii in range(1,len(video_frames_inverses)-1):
        ### DECODE START ###
        sampler.make_schedule(ddim_num_steps=t_dec, ddim_eta=ddim_eta, verbose=False)
        timesteps_decoder = np.flip(sampler.ddim_timesteps)

        print(f"Running DDIM Sampling with {t_dec} timesteps")
        iterator = tqdm(timesteps_decoder, desc='Decoding image', total=t_dec)
        x_dec_t1 = video_frames_inverses[ii]
        res_dict = defaultdict(list)


        for i, step in enumerate(iterator):
            index = t_dec - i - 1
            ts = torch.full((x_dec_t0.shape[0],), step, device=DEVICE, dtype=torch.long)


            x_dec_t1, pred_x0_t1, middle_h_t1, delta_h_t1 = sampler.p_sample_ddim(x_dec_t1, c, ts, index=index,
                                        unconditional_guidance_scale=decoder_scale,
                                        unconditional_conditioning=uc, asyrp_flag=False, 
                                        modified_flag=False, model_ns=None)

            loss = loss_fn(pred_x0_t0, pred_x0_t1)
            loss.backward()
            # x_dec_0_list = x_dec_t1_list #.....

        ### DECODE END ###
        samples = res_dict['x_pred'][-1]
        x_samples = model.decode_first_stage(samples)

    # x_samples = pu.save_sd_samples(x_samples, sample_path, base_count)
    # pu.save_as_grid(x_samples, n_rows, outpath, grid_count)


if __name__ == "__main__":
    main()
