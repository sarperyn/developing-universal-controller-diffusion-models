import PIL
import torch
from torchvision import transforms
import numpy as np
import einops
from ldm.data.util import AddMiDaS


def load_batch_sd(img_path,
                device='cuda',
                num_samples=1,
                is_depth=False,
                depth_model_type='dpt_hybrid'
                ):
    image = load_img_pil(img_path)
    w, h = 512,512 # TODO take sizes as an argument
    print(f"loaded input image of size ({w}, {h}) from {img_path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
    image = torch.from_numpy(np.array(image)).to(dtype=torch.float32) / 127.5 - 1.0
    batch = {
        'jpg': image,
        'midas_in': None
    }
    if is_depth:
        midas_trafo = AddMiDaS(model_type=depth_model_type)
        batch = midas_trafo(batch)
        batch["midas_in"] = einops.repeat(torch.from_numpy(batch["midas_in"][None, ...]).to(
            device=device), "1 ... -> n ...", n=num_samples)

    batch["jpg"] = einops.rearrange(batch["jpg"], 'h w c -> 1 c h w')
    batch["jpg"] = einops.repeat(batch["jpg"].to(device=device), "1 ... -> n ...", n=num_samples)
    return batch["jpg"], batch["midas_in"]

def load_batch_video_sd(image,
                device='cuda',
                num_samples=1,
                is_depth=False,
                depth_model_type='dpt_hybrid'
                ):
    image = image.to(dtype=torch.float32)
    image = einops.rearrange(image, 'h w c -> 1 c h w')
    image = torch.nn.functional.interpolate(image, (512, 512)).squeeze(0)
    image = einops.rearrange(image, 'c h w -> h w c')
    batch = {
        'jpg': image,
    }
    if is_depth:
        midas_trafo = AddMiDaS(model_type=depth_model_type)
        batch = midas_trafo(batch)
        batch["midas_in"] = einops.repeat(torch.from_numpy(batch["midas_in"][None, ...]).to(
            device=device), "1 ... -> n ...", n=num_samples)

    batch["jpg"] = einops.rearrange(batch["jpg"], 'h w c -> 1 c h w')
    batch["jpg"] = einops.repeat(batch["jpg"].to(device=device), "1 ... -> n ...", n=num_samples)

    return batch

def img_to_depth(cc, depth_model, z_shape):
    cc = depth_model(cc)
    depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3], keepdim=True)
    display_depth = (cc - depth_min) / (depth_max - depth_min)
    depth_image = PIL.Image.fromarray((display_depth[0, 0, ...].cpu().numpy() * 255.).astype(np.uint8))
    cc = torch.nn.functional.interpolate(cc,size=z_shape,mode="bicubic",align_corners=False,)
    depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3], keepdim=True)
    cc = 2. * (cc - depth_min) / (depth_max - depth_min) - 1.
    return depth_image, cc

def load_img_pil(img_path):
    pil_img = PIL.Image.open(img_path).convert('RGB')
    return pil_img

def pil_to_tensor(pil_img):
    return transforms.ToTensor()(pil_img)

def load_img_tensor(img_path):
    pil_img = load_img_pil(img_path)
    return pil_to_tensor(pil_img)