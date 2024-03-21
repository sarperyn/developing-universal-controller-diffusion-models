"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict


import utils.diffusion_utils as difu

from ldm.modules.diffusionmodules.util import make_beta_schedule, make_ddim_sampling_parameters, make_ddim_timesteps


class DDIMSampler(object):
    def __init__(self, diffusion_params, diffusion_model, device=None):
        super().__init__()
        # self.model = model
        self.params = diffusion_params
        self.ddpm_num_timesteps = 1000
        self.schedule = diffusion_params.schedule
        self.device = device
        self.diffusion_model = diffusion_model #u net model
        self.conditioning_key = diffusion_params.conditioning_key

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        
        betas = make_beta_schedule(self.schedule, self.ddpm_num_timesteps, linear_start=self.params.linear_start, linear_end=self.params.linear_end,
                                    cosine_s=8e-3)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        

        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: torch.from_numpy(x).to(torch.float32).to(self.device)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod,
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', to_torch(ddim_sigmas))
        self.register_buffer('ddim_alphas', to_torch(ddim_alphas))
        self.register_buffer('ddim_alphas_prev', to_torch(ddim_alphas_prev))
        self.register_buffer('ddim_sqrt_one_minus_alphas', to_torch(np.sqrt(1. - ddim_alphas)))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def apply_model(self, x_noisy, t, conds, model_ns=None):
        uncond, cond = conds[0], conds[1]
        params_dict = vars(self.params)
        sequential_cross_attn = params_dict.pop("sequential_crossattn", False)

        c_concat = difu.concat_conditions(uncond, cond)
        # for key,val in c_concat.items():
        #     print(type(val), len(val), val[0].size())
        # input()
        x_in = torch.cat([x_noisy] * 2)
        t_in = torch.cat([t] * 2)
        x_in, c_in = difu.process_condition(x_in, c_concat, self.conditioning_key, sequential_cross_attn)  
        
        out = self.diffusion_model(x_in, t_in,  model_ns=model_ns, context=c_in)        
        return out


    # def p_sample_ddim(self, x, c, t, index, repeat_noise=False, 
    #                   temperature=1.,
    #                   unconditional_guidance_scale=1., unconditional_conditioning=None,
    #                   asyrp_flag=False, 
    #                   modified_flag=False, model_ns=None):
    #     b, *_, device = *x.shape, x.device

    #     if asyrp_flag and model_ns is None:
    #         model_ns = {
    #             'delta_h_extra': None,
    #             'gamma': 1.0,
    #             'delta_block_flag': True,
    #         }
        
    #     if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
    #         model_output = self.apply_model(x, t, c, model_ns)
    #         e_t, e_t_modified, delta_h, middle_h = model_output
    #     else:
    #         x_in = torch.cat([x] * 2)
    #         t_in = torch.cat([t] * 2)
    #         if isinstance(c, dict):
    #             assert isinstance(unconditional_conditioning, dict)
    #             c_in = dict()
    #             for k in c:
    #                 if isinstance(c[k], list):
    #                     c_in[k] = [torch.cat([unconditional_conditioning[k][i], c[k][i]]) for i in range(len(c[k]))]
    #                 else:
    #                     c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
    #         elif isinstance(c, list):
    #             c_in = list()
    #             assert isinstance(unconditional_conditioning, list)
    #             for i in range(len(c)):
    #                 c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
    #         else:
    #             c_in = torch.cat([unconditional_conditioning, c])
    #         e_t, e_t_modified, delta_h, middle_h = self.apply_model(x_in, t_in, c_in, model_ns)
    #         e_t_uncond, e_t = e_t.chunk(2)
    #         if delta_h is not None:
    #             delta_h_uc, delta_h_c = delta_h.chunk(2)
    #             delta_h = delta_h_uc + unconditional_guidance_scale * (delta_h_c - delta_h_uc)
    #         if middle_h is not None:
    #             middle_h_uc, middle_h_c = middle_h.chunk(2)
    #             middle_h = middle_h_uc + unconditional_guidance_scale * (middle_h_c - middle_h_uc)
    #         if e_t_modified is not None:
    #             e_t_modified_uncond, e_t_modified = e_t_modified.chunk(2)
    #             e_t_modified = e_t_modified_uncond + unconditional_guidance_scale * (e_t_modified - e_t_modified_uncond)

    #         # e_t_cond = e_t
    #         # if epsilon_guidance is not None:
    #         #     coef = 0.01
    #         #     e_t = e_t + coef * epsilon_guidance 
    #         # e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)




    #     alphas = self.ddim_alphas
    #     alphas_prev = self.ddim_alphas_prev
    #     sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
    #     sigmas = self.ddim_sigmas
    #     # select parameters corresponding to the currently considered timestep
    #     a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
    #     a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
    #     sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
    #     sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

    #     # return the asyrp sample
    #     if asyrp_flag == True:
    #         modified_flag = False


    #     # use et if don't want to modify else et_modified
    #     noise_et = e_t_modified if modified_flag else e_t

    #     # current prediction for x_0
    #     if asyrp_flag == True:
    #         pred_x0 = (x - sqrt_one_minus_at * e_t_modified) / a_t.sqrt()
    #     else:
    #         pred_x0 = (x - sqrt_one_minus_at * noise_et) / a_t.sqrt()

    #     # direction pointing to x_t
    #     dir_xt = (1. - a_prev - sigma_t**2).sqrt() * noise_et
    #     noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
    #     x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

    #     return x_prev, pred_x0, middle_h, delta_h

    def ddim_guidance(self, out, guidance_scale):
        e_t, e_t_modified, delta_h, middle_h = out
        e_t_uncond, e_t = e_t.chunk(2)
        # if delta_h is not None:
        #     delta_h_uc, delta_h_c = delta_h.chunk(2)
        #     delta_h = delta_h_uc + guidance_scale * (delta_h_c - delta_h_uc)
        # if middle_h is not None:
        #     middle_h_uc, middle_h_c = middle_h.chunk(2)
        #     middle_h = middle_h_uc + guidance_scale * (middle_h_c - middle_h_uc)
        # if e_t_modified is not None:
        #     e_t_modified_uncond, e_t_modified = e_t_modified.chunk(2)
        #     e_t_modified = e_t_modified_uncond + guidance_scale * (e_t_modified - e_t_modified_uncond)
        e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
        return e_t, e_t_modified, delta_h, middle_h

    def ddim_step(self, x_next, index, conds, guidance_scale, inverse=False):

        t = torch.full((x_next.shape[0],), self.ddim_timesteps[index], device=x_next.device, dtype=torch.long)
        out  = self.apply_model(x_next.detach(), t, conds, model_ns=None)
        e_t, e_t_modified, delta_h, middle_h = self.ddim_guidance(out, guidance_scale)
        
        a = self.ddim_alphas[index] if inverse else self.ddim_alphas_prev[index]
        a_prev = self.ddim_alphas_prev[index] if inverse else self.ddim_alphas[index]
        sigma_t = self.ddim_sigmas[index]

        Pt = (x_next - (1-a_prev).sqrt() * e_t) / a_prev.sqrt() # predicted x0
        Dt = (1-a).sqrt() * e_t # direction pointing xt
        noise = sigma_t * torch.randn_like(x_next)
        
        x_next = a.sqrt() * Pt + Dt + noise
        
        return x_next



    # def encode(self, x0, c, t_enc,
    #            unconditional_guidance_scale=1.0, unconditional_conditioning=None, model_ns=None):
    #     num_reference_steps = self.ddim_timesteps.shape[0]
    #     # print(t_enc, num_reference_steps)
    #     # input()
    #     timesteps =  self.ddim_timesteps
    #     num_reference_steps = timesteps.shape[0]
    #     assert t_enc <= num_reference_steps
    #     num_steps = t_enc


    #     alphas_next = self.ddim_alphas[:num_steps]
    #     alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

    #     x_next = x0
    #     inter_steps = []

    #     for i in tqdm(range(num_steps), desc='Encoding Image'):
    #         t = torch.full((x0.shape[0],), timesteps[i], device=self.device, dtype=torch.long)
    #         if unconditional_guidance_scale == 1.:
    #             noise_pred, e_t_modified, delta_h, middle_h  = self.apply_model(x_next, t, c, model_ns)
    #         else:
    #             assert unconditional_conditioning is not None

    #             e_t, _, _, middle_h = self.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)), torch.cat((unconditional_conditioning, c)))
    #             e_t_uncond, e_t = e_t.chunk(2)

    #             if middle_h is not None:
    #                 middle_h_uc, middle_h_c = middle_h.chunk(2)
    #                 middle_h = middle_h_uc + unconditional_guidance_scale * (middle_h_c - middle_h_uc)
                    
    #             noise_pred = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            
    #         xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
    #         weighted_noise_pred = alphas_next[i].sqrt() * ((1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
    #         x_next = xt_weighted + weighted_noise_pred


    #     out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}

    #     return x_next, out


    # def stochastic_encode(self, x0, t, noise=None):
    #     # fast, but does not allow for exact reconstruction
    #     # t serves as an index to gather the correct alphas


    #     sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
    #     sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

    #     if noise is None:
    #         noise = torch.randn_like(x0)
    #     return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
    #             extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)


    # def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
    #             asyrp_flag=False, modified_flag=False, model_ns=None):

    #     timesteps = self.ddim_timesteps[:t_start]

    #     time_range = np.flip(timesteps)
    #     total_steps = timesteps.shape[0]
    #     print(f"Running DDIM Sampling with {total_steps} timesteps")
    #     iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
    #     x_dec = x_latent
    #     res_dict = defaultdict(list)
    #     x_decs = [x_dec.detach().cpu().numpy()]
    #     for i, step in enumerate(iterator):
    #         index = total_steps - i - 1
    #         ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)

    #         x_dec, pred_x0, middle_h, delta_h = self.p_sample_ddim(x_dec, cond, ts, index=index,
    #                                       unconditional_guidance_scale=unconditional_guidance_scale,
    #                                       unconditional_conditioning=unconditional_conditioning, asyrp_flag=asyrp_flag, 
    #                                       modified_flag=modified_flag, model_ns=model_ns)
    #         res_dict['x_pred'].append(x_dec.detach().cpu().numpy())
    #         res_dict['x0_pred'].append(pred_x0.detach())
    #         res_dict['middle_h'].append(middle_h.detach().cpu().numpy())
    #         if delta_h is not None:
    #             res_dict['delta_h'].append(delta_h.detach().cpu().numpy())
        
    #     res_dict['x_pred'][-1] = torch.from_numpy(res_dict['x_pred'][-1]).to(x_dec.device)
    #     return res_dict
