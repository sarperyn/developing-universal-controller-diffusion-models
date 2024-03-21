from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np

def get_explanation(image, device, model):
    image = image.squeeze(0).cpu().numpy()
    image = (image + 1) / 2
    image = (image * 255).astype(np.uint8)
    if image.shape[0] == 1 or image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    image = Image.fromarray(image)

    image_size = 384
    image = load_demo_image(image, image_size, device)
    # model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    
    # model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    # model.eval()
    # model = model.to(device)

    with torch.no_grad():
        caption = model.generate(image, sample=True, num_beams=3, max_length=20, min_length=5) 

    return caption[0]


def load_demo_image(img, image_size,device):    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(img).unsqueeze(0).to(device)
    return image