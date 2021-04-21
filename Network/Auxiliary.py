#Свистелки и мигалки

import numpy as np 
from PIL import Image
from torch import Tensor


def tensor_to_pillow(t):
    """
    Transforms a PyTorch tensor with dimensions suitable for network
    to a Pillow image to show it
    """
    mode = None
    sh = t.shape
    if sh[1] == 1:
        mode = 'L'
        t1 = t.reshape(sh[2],sh[3])
    else:    
        t1 = t.reshape(sh[1],sh[2],sh[3])
        t1 = t1.permute(1,2,0)
    t1 = t1.detach().numpy()
    t1 = t1*255
    img = Image.fromarray(np.uint8(t1), mode)
    return img 