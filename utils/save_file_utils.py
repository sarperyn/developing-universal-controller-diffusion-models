import torchvision.utils as tvu
import yaml



def save_diffusion_output_torch(torch_array, save_path):
    tvu.save_image(torch_array, save_path, normalize=True)
    
    
def save_dict_as_yaml(dic, path):
    with open(path, 'w') as outfile:
        yaml.dump(dic, outfile, default_flow_style=False)

def args_to_text(args):
    args_dict = vars(args)
    file_name = ''
    for key,value in args_dict.items():
        file_name += f'{key}-{value}-'
    return file_name[:-1] 
        
def print_args(opt):
    opt_dict = vars(opt)
    for key,val in opt_dict.items():
        print(f'{key}: {val}')
    print()