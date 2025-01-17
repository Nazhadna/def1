#Загрузка и предобработка информации

import numpy as np
import pandas as pd
import os
import io

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from PIL import Image


def get_names(IMG_PATH):
    names = []
    for _,_,files in os.walk(IMG_PATH):
        for file in files:
            names.append(file.split('.')[0])
    return pd.DataFrame({'id':names}, index = np.arange(len(names)))
    
    
def load_img(bytes):
    img = Image.open(io.BytesIO(bytes))
    img.save('static/uploads/image.jpg')
    img = img.convert('RGB')
    mean=[0.485, 0.456, 0.406] #Эти числа всё время встречаются в документации PyTorch
    std=[0.229, 0.224, 0.225] #Поэтому использованы именно они
    t = T.Compose([T.ToTensor(),T.Normalize(mean,std)])
    img = t(img)
    sh = img.shape
    img = img.reshape(1,sh[0],sh[1],sh[2])
    return img
    
    
def save_img(img,path):
    nc = img.shape[1]
    img = torch.argmax(F.softmax(img, dim=1), dim=1)
    sh = img.shape
    img = img.detach().cpu().numpy().reshape((sh[1],sh[2]))
    img = np.uint8(img)
    r = img.copy()
    r[r==1] = 255
    r[r==4] = 255
    r[r!=255] = 0
    r = Image.fromarray(r)
    g = img.copy()
    g[g==2] = 255
    g[g!=255] = 0
    g = Image.fromarray(g)
    b = img.copy()
    b[b==3] = 255
    b[b==4] = 255
    b[b!=255] = 0
    b = Image.fromarray(b)
    img = Image.merge('RGB',(r,g,b))
    img.save(path)    
        
        
class PipeDataset(Dataset):
    def __init__(self,img_path,mask_path, sample_ids,transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.sample_ids = sample_ids
        self.transform = transform
        
    def __len__(self):
        return len(self.sample_ids)
        
    def __getitem__(self,idx):
        img = Image.open(self.img_path+self.sample_ids[idx]+'.png')
        mask = np.load(self.mask_path+self.sample_ids[idx]+'.npy')
        if self.transform is not None:
            img = np.array(img)
            aug = self.transform(image=img,mask=mask)
            img = aug['image']
            mask = aug['mask']
        mean=[0.485, 0.456, 0.406] #Эти числа всё время встречаются в документации PyTorch
        std=[0.229, 0.224, 0.225] #Поэтому использованы именно они
        t = T.Compose([T.ToTensor(),T.Normalize(mean,std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()
        
        return img,mask


    
