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
This draws a sample from the StyleGAN generator.
First we sample a random latent code.
Then we feed this through a generator neural network, producing a 3-channel RGB image, as well as activations at early layers.
In particular:
* `act2` are activations at layer 2 (0-indexing used here), giving us an 512x8x8 tensor of activations, e.g. 512 channels, 8x8 spatial resolution
* `act3` are activations at layer 3, giving us another 512x8x8 tensor of activations.
* `act3_up` is the result of bilinear upsampling of `act3` to a 512x16x16 tensor.
* `act4` are activations at layer 4, giving us a 512x16x16 tensor of activations.
'''
def sample_generator():
    code = torch.randn(1,generator.z_space_dim)
    if is_cuda:
        code = code.cuda()

    with torch.no_grad():
        # truncated normal distribution, no random noise in style layers!
        gen_out =  generator(code, trunc_psi=0.7,trunc_layers=8,randomize_noise=False)

        act2 = gen_out['act2'][0].detach()
        act3 = gen_out['act3'][0].detach()
        act3_up = torch.nn.functional.interpolate(act3.unsqueeze(0),scale_factor=2,mode='bilinear',align_corners=True)[0]
        act4 = gen_out['act4'][0].detach()

        image = gen_out['image'][0].detach()
    #

    return act2,act3,act3_up,act4,image
#

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
    pass
#

'''
TODO

Given a tensor of activations (n_samples x channels x x-resolution x y-resolution), compute the per-channel top quantile (defined by perc), and then threshold activations
based on the quantile (perform this per channel)
'''
def threshold(acts,k=4):
    pass
#

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
#