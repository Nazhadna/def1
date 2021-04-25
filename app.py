import os
from flask import Flask, render_template, url_for, jsonify, request, flash, redirect, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './static/uploads/'
UPLOAD_NAME = 'image.jpg'
ALLOWED_EXTENSIONS = {'jpeg', 'jpg', 'png'}

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_NAME'] = UPLOAD_NAME

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
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        # if f and allowed_file(f.filename):
        #     filename = secure_filename(f.filename)
        #     f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('uploaded_file',
            #                         filename=filename))
    return render_template('result.html')

# @app.route('/temp/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'],
#                                filename)

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run()