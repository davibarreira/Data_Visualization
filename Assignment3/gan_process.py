import os
import numpy as np
import time
from PIL import Image

from tqdm import tqdm

import torch

from models import MODEL_ZOO
from models import build_generator

# if you have CUDA-enabled GPU, set this to True!
is_cuda = False

# StyleGAN tower
model_name = 'stylegan_tower256'
model_config = MODEL_ZOO[model_name].copy()
url = model_config.pop('url')  # URL to download model if needed.

# generator
generator = build_generator(**model_config)

# load the weights of the generator
checkpoint_path = os.path.join('checkpoints', model_name+'.pth')
checkpoint = torch.load(checkpoint_path, map_location='cpu')
if 'generator_smooth' in checkpoint:
    generator.load_state_dict(checkpoint['generator_smooth'])
else:
    generator.load_state_dict(checkpoint['generator'])
if is_cuda:
    generator = generator.cuda()
generator.eval()
'''
Postprocess images from the generator network - suitable to write to disk via PIL.
'''
def postprocess(images):
    scaled_images = (images+1)/2
    np_images = 255*scaled_images.numpy()
    np_images = np.clip(np_images + 0.5, 0, 255).astype(np.uint8)
    np_images = np_images.transpose(0, 2, 3, 1)
    return np_images
#

'''
TODO

Compute the Intersection-over-Union score between all pairs of channel activations for the provided tensors.
The tensors should be of the same spatial resolution. Further, the tensors should be comprised of values 0 or 1, derived from quantile-based thresholding.
This should return a tensor of shape (channel x channel)

NOTE: this can be done with a few lines of code using broadcasting! (no loops necessary)
'''
def iou(a_i,a_j):
    iou = torch.zeros(a_i.shape[0],512,512)
    for s in range(0,a_i.shape[0]):
        for i in range(0,a_i.shape[1]):
            for j in range(i,a_j.shape[1]):
                tsum = a_i[s][i] + a_j[s][j]
                iou[s,i,j] = torch.sum(tsum>1)/torch.sum(tsum>0)
                iou[s,j,i] = iou[s,i,j]
                
    return iou
    
#

'''
TODO

Given a tensor of activations (n_samples x channels x x-resolution x y-resolution), compute the per-channel top quantile (defined by perc), and then threshold activations
based on the quantile (perform this per channel)
'''
def threshold(acts,k=4):
    tensor_list=[]
    qt =[]
    for i in range(0,acts.shape[1]):
        q = torch.quantile(acts[:,i,:,:],1-1/4)
        tensor_list.append((acts[:,i,:,:] > q).type(torch.uint8))
        qt.append(q)
    t = torch.stack(tensor_list)
    qt=  torch.stack(qt)
    return t.permute(1,0,2,3),qt

'''
TODO

Preprocessing:
    1. Generate a set of samples from the generator network (see sample_generator above).
    2. Threshold channel activations at each layer.
    3. Compute IoU scores, for each sample, between all pairs of channels from layer 2 to layer 3, and layer 3 to layer 4 -> should produce 2 tensors of shape (n_samples x channels x channels).
    4. Postprocess images and write the images to disk.
    5. Write out the threhsolded activations, and IoU score tensors, to disk.

Write everything to the 'static' directory.
'''
if __name__=='__main__':
    n_samples = 20
#     samples = []
    act2    = []
    act3    = []
    act3up  = []
    act4    = []
    images  = []
    for i in range(0,n_samples):
        sample = sample_generator()
        act2.append(sample[0])
        act3.append(sample[1])
        act3up.append(sample[2])
        act4.append(sample[3])
        images.append(sample[4])
        
    act2   = torch.stack(act2)
    act3   = torch.stack(act3)
    act3up = torch.stack(act3up)
    act4   = torch.stack(act4)
    images = torch.stack(images)
    
    tact2,q2   = threshold(act2)
    tact3,q3   = threshold(act3)
    tact3up,q3up = threshold(act3up)
    tact4,q4   = threshold(act4)
    
    qt = torch.dstack((q2,q3,q3up,q4))[0]
    torch.save(qt,'./static/qt.pt')
    
    iou2_3  = iou(tact2,tact3)
    torch.save(iou2_3,'./static/tensor2_3.pt')
    
    iou3_4  = iou(tact3up,tact4)
    torch.save(iou3_4,'./static/tensor3_4.pt')
    
    images = postprocess(images)
    for i in range(0,n_samples):
        im = Image.fromarray(images[i],"RGB")
        im.save('./static/image'+str(i)+'.jpeg')
