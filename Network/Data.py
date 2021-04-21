#Загрузка и предобработка информации

import numpy as np
import pandas as pd
import os

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
        img = Image.open(self.img_path+self.sample_ids[Id]+'.jpg')
        img = scale(img,0.25,Image.NEAREST)
        img = T.functional.to_tensor(img)
        
        mask = Image.open(self.mask_path+self.sample_ids[Id]+'.png')
        mask = scale(mask,0.25,Image.NEAREST)
        mask = np.array(mask)
        mask = torch.from_numpy(mask).long()
        
        return img,mask