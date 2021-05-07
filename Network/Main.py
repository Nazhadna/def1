from Data import *
from Training import *
from Simple_Unet import *


img = load_img(bytes)
h,w = img.shape[2],img.shape[3]
net = UNet(num_class=23,retain_dim=True,out_sz=(h,w))
net.load_state_dict(torch.load(net_path))
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
net = net.to(device)
net.eval()
res = net(img)
save_img(res,'res.png')
    
