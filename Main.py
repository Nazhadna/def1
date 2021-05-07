from Data import *
from Training import *
from Simple_Unet import *
# from Network.Data import *
# from Network.Training import *
# from Network.Simple_Unet import *


class Segmenter:
    def __init__(self,img_path = '',net_path = 'dict.pth',**kwargs):
        self.img_path = img_path
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.network = UNet(**kwargs)
        self.network.load_state_dict(torch.load(net_path))
        self.network = self.network.to(self.device)
        self.network.eval()
        
    def __call__(self,img_name):
        torch.cuda.empty_cache()
        img = load_img(self.img_path+img_name).to(self.device)
        with torch.no_grad():
            res = self.network(img)
        save_img(res,self.img_path+img_name)
        
        
if __name__ == '__main__':
    seg = Segmenter(enc_chs=(3,16,32,64,128,256), dec_chs=(256, 128, 64, 32, 16), num_class=23)
    seg('000')