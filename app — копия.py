import os
from flask import Flask, render_template, url_for, jsonify, request, flash, redirect, send_from_directory
from werkzeug.utils import secure_filename
from Main import Segmenter

from Data import *
from Training import *
from Simple_Unet import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(3,16,32,64,128,256)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(256, 128, 64, 32, 16)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = T.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(3,16,32,64,128,256), dec_chs=(256, 128, 64, 32, 16), num_class=1, retain_dim=False, out_sz=(1000,1500)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        if retain_dim:
            self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out

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

UPLOAD_FOLDER = './static/uploads/'
UPLOAD_NAME = 'image.jpg'
ALLOWED_EXTENSIONS = {'jpeg', 'jpg', 'png'}

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_NAME'] = UPLOAD_NAME
# app.config['TEMPLATES_AUTO_RELOAD'] = True

# @app.before_request
# def before_request():
#     if 'localhost/upload' in request.host_url:
#         app.jinja_env.cache = {}

@app.route('/')
def upload():
    return render_template('upload.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    # args = {"method": "GET"}
    if request.method == 'POST':
        # check if the post request has the file part
        # if 'file' not in request.files:
        #     flash('No file part')
        #     return redirect(request.url)
        f = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        # if f.filename == '':
        #     flash('No selected file')
        #     return redirect(request.url)
        f.filename = secure_filename(app.config['UPLOAD_NAME'])
        fullpath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(fullpath)
        net = Segmenter(fullpath, 'Network/Model_weights.pth', enc_chs=(3,16,32,64,128,256), dec_chs=(256, 128, 64, 32, 16), num_class=23)
        net('')
        # if f and allowed_file(f.filename):
        #     filename = secure_filename(f.filename)
        #     f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('uploaded_file',
            #                         filename=filename))
        # Вызов метода обработки с передачей имени в качестве параметра
    return render_template('result.html')

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run()