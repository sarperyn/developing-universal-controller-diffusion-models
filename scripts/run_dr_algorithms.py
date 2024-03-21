import os
import sys
import numpy as np
import einops
import scipy
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap
sys.path.append(os.path.dirname(os.getcwd()))

from tqdm import tqdm
import glob
import torch
import torchvision.transforms as transforms

import utils.datasets as d
import utils.functions as f
import utils.img_utils as iu
import utils.save_file_utils as sfu
import utils.schedule_utils as su
from utils.constants import *



def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

def get_hvecs(chair_360_dirs,index):
    hvec_paths = []
    for chair in chair_360_dirs[index:index+1]:

        hvec_path = sorted(glob.glob(chair + '/hvecs_model378000_gd_128/*.npy'))

        for i in hvec_path:

            hvec = np.load(i)
            hvec_paths.append(hvec)
    
    hvecs = np.array(hvec_paths)

    return hvecs

def get_comps(hvecs, algorithm):

    hvecs_flatten = einops.rearrange(hvecs,'a b c d e -> a (b c d e)')
    #hvecs_flatten = StandardScaler().fit_transform(hvecs_flatten)
    print(hvecs_flatten.shape)
    np.save('hvecs_4_1inc_flatten_manchester_reds.npy',hvecs_flatten)

    return None

    if algorithm == 'PCA':
        pca = PCA(n_components=2)
        comps = pca.fit_transform(hvecs_flatten)

    elif algorithm == 'UMAP':
        comps = umap.UMAP().fit(hvecs_flatten)
    
    elif algorithm == 'TSNE':
        comps = TSNE(n_components=2).fit_transform(hvecs_flatten)

    elif algorithm == "INC_PCA":
        ipca = IncrementalPCA(n_components=2, batch_size=10)
        comps = ipca.fit_transform(hvecs_flatten)

    return comps


def plot_comps(pca_comps, index, num_points, num_chairs):

    color1 = "#D4CC47"
    color2 = "#7C4D8B"
    plt.figure(figsize=(15,15))

    plt.scatter(pca_comps[:,0],pca_comps[:,1],
        color=get_color_gradient(color1, color2, num_points))
    plt.colorbar()
    plt.title("Gradient Scatter")
    plt.savefig(f'chair_{index}.png',format='png',dpi=100)
    plt.show()
    plt.close()
    print(f"saved_chair{index}")

def plot_comps2(comps, index, num_points, num_chairs):


    plt.figure(figsize=(15,15))

    increment = num_points//num_chairs
    colors = cm.rainbow(np.linspace(0, 1, num_points))
    print(len(colors))
    
    for idx in range(num_chairs):
        plt.scatter(comps[increment*idx:increment*(idx+1),0],comps[increment*idx:increment*(idx+1),1],
            color=colors[increment*idx:increment*(idx+1)])
        
    plt.colorbar()
    plt.title("Gradient Scatter")
    plt.savefig(f'chair_{index}.png',format='png',dpi=100)
    plt.show()
    plt.close()
    print(f"saved_chair{index}")

def plot_sample_comps(comps, index, num_points, num_chairs):


    plt.figure(figsize=(15,15))

    increment = num_points//num_chairs

    colors = []
    for i in ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'][:num_chairs]:
        colors.extend([i]*increment)

    plt.scatter(comps[:,0],comps[:,1],color=colors)
    plt.colorbar()
    plt.title("Gradient Scatter")
    plt.savefig(f'chair_{index}.png',format='png',dpi=100)
    plt.show()
    plt.close()
    print(f"saved_chair{index}")



def plot_degree_comps(comps, index, n_points, num_chairs):

    color1 = "#D4CC47"
    color2 = "#7C4D8B"
    
    degree_per_image = n_points // num_chairs
    gradient_color = get_color_gradient(color1, color2, degree_per_image)
    gradient_color = gradient_color * num_chairs

    plt.scatter(comps[:,0],comps[:,1],color=gradient_color, s=1)
    plt.colorbar()
    plt.title("Gradient Scatter")
    plt.savefig(f'chair_{index}.png',format='png',dpi=100)
    plt.show()
    plt.close()
    print(f"saved_chair{index}")


def dr_for_all():
    dataset = CHAIR_PATH
    chair_dirs = sorted(glob.glob(os.path.join(dataset,'*')))
    which_chairs = 480 if dataset == CHAIR_PATH2 else 360
    ALGORITHM = 'PCA'

    chair_360_dirs = []
    for chair in chair_dirs:

        if (len(sorted(glob.glob(chair + '/hvecs_stablediff_50/*.npy'))) == which_chairs):
            chair_360_dirs.append(chair)


    for i in range(len(chair_360_dirs)):

        hvecs = get_hvecs(chair_360_dirs,i)
        comps = get_comps(hvecs, ALGORITHM)
        plot_comps(comps.embedding_,i, 360)
        print(i)


def dr_for_some(n_chair, deg_increment, hue):
    dataset = CHAIR_PATH
    chair_dirs = sorted(glob.glob(os.path.join(dataset, '*')))
    chair_360_dirs = []
    for chair in chair_dirs:

        if (len(sorted(glob.glob(chair + '/hvecs_stablediff_50/*.npy'))) == 360):
            chair_360_dirs.append(chair)

    chair_dirs = chair_360_dirs[n_chair:n_chair+n_chair]
    ALGORITHM = 'INC_PCA'

    n_degree = 360//deg_increment
    
    degrees = np.linspace(0, 360, n_degree, endpoint=False, dtype=int)
    degrees = [str(x).zfill(4) + '.npy' for x  in degrees]

    
    #chair_dirs = [ '/data/datasets/CHAIR/chair_fixed_elev/seen/03001627_2a28a1658e9b557062c658925896f75e', 
                   #'/data/datasets/CHAIR/chair_fixed_elev/seen/03001627_1f1b07bf637b3a1582db9fca4b68095', '/data/datasets/CHAIR/chair_fixed_elev/seen/03001627_22b40d884de52ca3387379bbd607d69e']
    chair_dirs = ['/data/datasets/CHAIR/chair_fixed_elev/seen/03001627_1f1a9120cba6c1ce177b3ebe695b7c2f', '/data/datasets/CHAIR/chair_fixed_elev/seen/03001627_13c18609602e4ced37b2bb75885cfc44', '/data/datasets/CHAIR/chair_fixed_elev/seen/03001627_249b40a630dd751f8023b347a089645c']
    print(chair_dirs)
    chair_hvecs = []
    for chair in chair_dirs:
    
        for idx in range(len(degrees)):

            hvec_path = os.path.join(chair, 'hvecs_stablediff_50', f'{degrees[idx]}')
            hvec = np.load(hvec_path)
            chair_hvecs.append(hvec)
        print(chair)
    
    hvecs_stack = np.concatenate(np.expand_dims(chair_hvecs, axis=0))

    comps = get_comps(hvecs_stack, ALGORITHM)
    # if hue == 'degree':
    #     plot_degree_comps(comps, f'{n_chair}-{deg_increment}_{ALGORITHM}_{hue}', n_degree*len(chair_dirs), len(chair_dirs))
    # elif hue == 'sample':
    #     plot_sample_comps(comps, f'{n_chair}-{deg_increment}_{ALGORITHM}_{hue}', n_degree*len(chair_dirs), len(chair_dirs))


dr_for_some(n_chair=4, deg_increment=1, hue='degree')
