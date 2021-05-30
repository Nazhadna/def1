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
    
    
def flip(arr):
    l = len(arr)
    h = l//2
    for i in range(h):
        arr[i],arr[l-i-1] = arr[l-i-1].copy(),arr[i].copy()
        
        
def two_for_one(IMG_PATH,MASK_PATH):
    for idx in df['id']:
        name = 'ex' + str(int(idx[2:])+45)
        img = Image.open(IMG_PATH+idx+'.jpg')
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(IMG_PATH+'flip\\'+name+'.jpg')
        mask = np.load(MASK_PATH+idx+'.npy')
        flip(mask)
        np.save(MASK_PATH+'flip\\'+name+'.npy',mask)