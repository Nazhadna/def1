#Загрузка и предобработка информации

import numpy as np
import pandas as pd
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from PIL import Image
from PIL.ImageOps import scale


def get_names(path):
    names = []
    for _,_,files in os.walk(path):
        for file in files:
            names.append(file.split('.')[0])
    return pd.DataFrame({'id':names}, index = np.arange(len(names)))
    
    
def load_img(path):
    img = Image.open(path+'.jpg')
    img = scale(img,0.25,Image.NEAREST)
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
    img = img/nc*255
    img = np.uint8(img)
    img = Image.fromarray(img)
    img.save(path+'.png')    
    

class DroneDataset(Dataset):
    """
    Загрузчик для данных, на которых тестиуется сеть,
    пока своего датасета нет.
    """
    def __init__(self,img_path,mask_path, sample_ids):
        self.img_path = img_path
        self.mask_path = mask_path
        self.sample_ids = sample_ids
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self,Id):
        img = Image.open(self.img_path+self.sample_ids[Id])
        img = scale(img,0.25,Image.NEAREST)
        mean=[0.485, 0.456, 0.406] #Эти числа всё время встречаются в документации PyTorch
        std=[0.229, 0.224, 0.225] #Поэтому использованы именно они
        t = T.Compose([T.ToTensor(),T.Normalize(mean,std)])
        img = t(img)
        
        mask = Image.open(self.mask_path+self.sample_ids[Id])
        mask = scale(mask,0.25,Image.NEAREST)
        mask = np.array(mask)
        mask = torch.from_numpy(mask).long()
        
        return img,mask
        
        
class PipeDataset(Dataset):
    def __init__(self,img_path,mask_path, sample_ids):
        self.img_path = img_path
        self.mask_path = mask_path
        self.sample_ids = sample_ids
        
    def __getitem__(self,idx):
        img = Image.open(self.img_path+self.sample_ids[idx])
        mean=[0.485, 0.456, 0.406] #Эти числа всё время встречаются в документации PyTorch
        std=[0.229, 0.224, 0.225] #Поэтому использованы именно они
        t = T.Compose([T.ToTensor(),T.Normalize(mean,std)])
        img = t(img)
        
        mask = np.load(mask_path+self.sample_ids[idx]
        mask = torch.from_numpy(mask).long()
        
        return img,mask
    
