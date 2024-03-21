import os
import torch
import einops
import cv2


from PIL import Image
from itertools import product
from torchvision.utils import make_grid, save_image

import numpy as np
import matplotlib.pyplot as plt

def save_diffusion_output_torch(torch_array, save_path):
    save_image(torch_array, save_path, normalize=True)

def save_as_grid(samples, n_rows, outpath, grid_count):
        # additionally, save as grid
        # grid = torch.stack(samples, 0)
        # grid = einops.rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(samples, nrow=n_rows)

        # to image
        grid = 255. * einops.rearrange(grid, 'c h w -> h w c').cpu().numpy()
        grid = Image.fromarray(grid.astype(np.uint8))
        grid.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))

def save_sd_samples(x_samples, sample_path, base_count):
    '''
    x_samples -> list of torchs [-1, 1]
    '''
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    for x_sample in x_samples:
        save_sd_sample(x_sample, sample_path, base_count)
        base_count += 1
    return x_samples

def save_sd_x0_samples(x_samples, sample_path, base_count, step):
    '''
    x_samples -> list of torchs [-1, 1]
    '''
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    for x_sample in x_samples:
        save_sd_x0_sample(x_sample, sample_path, base_count, step)
        base_count += 1
    return x_samples

def save_sd_x0_sample(x_sample, sample_path, base_count, step):
    x_sample = 255. * einops.rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(x_sample.astype(np.uint8))
    img.save(os.path.join(sample_path, f"x0-{step:05}-{base_count:05}.png"))

def save_sd_sample(x_sample, sample_path, base_count):
    x_sample = 255. * einops.rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(x_sample.astype(np.uint8))
    img.save(os.path.join(sample_path, f"{base_count:05}.png"))


def plot_results(imgs, recons_target, modified, current_deg, target_deg, save_path, epoch, idx, args):

    bs = imgs.shape[0]
    fig, axes = plt.subplots(nrows=3,ncols=bs,figsize=(bs*4,20))

    for i, (row,col) in enumerate(product(range(3),range(bs))):

        if row == 0:
            axes[row][col].imshow(np.transpose(imgs[col].detach().cpu().numpy(),(1,2,0)))
            axes[row][col].set_xlabel(f'Current Degree: {int(current_deg[col])}',fontsize=15,fontweight='bold')
            if col == 0:
                axes[row][col].set_ylabel('ORIGINAL',fontsize=15,fontweight='bold')
        
        elif row == 1:
            axes[row][col].imshow(np.transpose(modified[col].detach().cpu().numpy(),(1,2,0)))

            if col == 0:
                axes[row][col].set_ylabel('MODIFIED',fontsize=15,fontweight='bold')

        elif row == 2:
            axes[row][col].imshow(np.transpose(recons_target[col].detach().cpu().numpy(),(1,2,0)))
            axes[row][col].set_xlabel(f'Target Degree: {int(target_deg[col])}',fontsize=15,fontweight='bold')

            if col == 0:
                axes[row][col].set_ylabel('T_RECONSTRUCTED',fontsize=15,fontweight='bold')
            
        axes[row][col].set_yticks([])
        axes[row][col].set_xticks([])

    plt.subplots_adjust(wspace=0,hspace=0)
    suptitle = plt.suptitle(f'lr:{args.lr} \n  t_eta:{args.t_eta} \n  t_x0_loss:{args.t_x0_loss}  \n  lower-upper degree:{args.lower_degree}-{args.upper_degree}  \n view_per_instance:{args.view_per_instance}  \n object_amount:{args.object_amount}', fontsize=20,fontweight='bold')#plt.suptitle("My Awesome, World-Changing Plot", y=1.02)
    plt.savefig(os.path.join(save_path,f'fig_{epoch}_{idx}.jpg'),format='jpg', bbox_inches='tight', pad_inches=0, dpi=100, bbox_extra_artists=(suptitle,))
    plt.show() 
    plt.close()

def plot_validation(imgs, recons_target, modified, current_deg, target_deg, save_path, idx, metric_dict):

    bs = 1
    fig, axes = plt.subplots(nrows=3,ncols=bs,figsize=(4.2,12))

    for i, (row,col) in enumerate(product(range(3),range(bs))):

        if row == 0:
            axes[row].imshow(np.transpose(imgs[0].detach().cpu().numpy(),(1,2,0)))
            axes[row].set_xlabel(f'Current Degree: {int(current_deg[col])}',fontsize=15,fontweight='bold')
            axes[row].set_ylabel('ORIGINAL',fontsize=15,fontweight='bold')
        
        elif row == 1:
            axes[row].imshow(np.transpose(modified[0].detach().cpu().numpy(),(1,2,0)))
            axes[row].set_ylabel('MODIFIED',fontsize=15,fontweight='bold')

        elif row == 2:
            axes[row].imshow(np.transpose(recons_target[0].detach().cpu().numpy(),(1,2,0)))
            axes[row].set_xlabel(f'Target Degree: {int(target_deg[col])}',fontsize=15,fontweight='bold')
            axes[row].set_ylabel('T_RECONSTRUCTED',fontsize=15,fontweight='bold')
            
        axes[row].set_yticks([])
        axes[row].set_xticks([])

    plt.subplots_adjust(wspace=0,hspace=0)
    plt.savefig(os.path.join(save_path,f'fig_val_{idx}.jpg'),format='jpg', bbox_inches='tight', pad_inches=0, dpi=100,)
    plt.close()

def plot_viewpoint_validation(inputs, gt_angle, pred_angle, save_path, epoch):

    bs = 6
    fig, axes = plt.subplots(nrows=1,ncols=bs,figsize=(12,6))

    for i, (row,col) in enumerate(product(range(1),range(bs))):

        axes[col].imshow(np.transpose(inputs[col].detach().cpu().numpy(),(1,2,0)))
        axes[col].set_xlabel(f'GT Degree: {int(gt_angle[col])}  Pred Degree: {int(pred_angle[col])}',fontsize=6)
        axes[col].set_yticks([])
        axes[col].set_xticks([])

    plt.subplots_adjust(wspace=0,hspace=0)
    plt.savefig(os.path.join(save_path,f'viewpow_val_{epoch}.jpg'), format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def text_under_image(image, text, text_color=(0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02, save_path=None):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    pil_img.save(save_path)

def aggregate_attention(attention_store, res, from_where, is_cross, prompts):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[0]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

def show_cross_attention(attention_store, res, from_where, prompts, save_path):
    tokens = prompts[0].split()
    st = ['start_of_token']
    st.extend(tokens)
    st.extend(['end_of_token'])
    tokens = st.copy()
    attention_maps = aggregate_attention(attention_store, res, from_where, True, prompts)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = text_under_image(image, tokens[i], (0, 0, 0))
        images.append(image)
    view_images(np.stack(images, axis=0), save_path=save_path)
    
    
    
def show_self_attention(attention_store, res, from_where, max_com=10, save_path=None, prompts=None):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, prompts=prompts).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    view_images(np.concatenate(images, axis=1), save_path=save_path)