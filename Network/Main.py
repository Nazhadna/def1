from Data import *
from Training import *
from Simple_Unet import *


class Segmenter:
    def __init__(self,img_path = '',net_path = 'Model_Weights.pth'):
        self.img_path = img_path
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.network = torch.load(net_path).to(self.device)
        self.network.eval()
        
    def __call__(self,img_name):
        torch.cuda.empty_cache()
        img = load_img(self.img_path+img_name).to(self.device)
        with torch.no_grad():
            res = self.network(img)
        save_img(res,self.img_path+img_name)