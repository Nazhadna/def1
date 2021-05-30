import os
from flask import Flask, render_template, url_for, jsonify, request, flash, redirect, send_from_directory
from werkzeug.utils import secure_filename
from Data import *
from Simple_Unet import *

UPLOAD_FOLDER = './static/uploads/'
UPLOAD_NAME = 'image.jpg'
ALLOWED_EXTENSIONS = {'jpeg', 'jpg', 'png'}

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_NAME'] = UPLOAD_NAME

def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload():
    return render_template('upload.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            img_bytes = file.read()
            img = load_img(img_bytes)
            h,w = img.shape[2],img.shape[3]
            net = UNet(num_class=23,retain_dim=True,out_sz=(h,w))
            net.load_state_dict(torch.load('dict.pth'))
            net = net.to('cpu')
            net.eval()
            res = net(img)
            save_img(res,'static/uploads/res.png')
            # tensor = transform_image(img_bytes)
            # prediction = get_prediction(tensor)
            # data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
            return render_template('result.html')
        except:
            return jsonify({'error': 'error during prediction'})

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

if __name__ == '__main__':
    app.run()