
from flask import Flask, flash, request, redirect,  jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.models import load_model
import tensorflow as tf
import keras.backend as K
import json

UPLOAD_FOLDER = '\\uploaded'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
modelpath='cat-dog-nn.dat'
target=(256,256)
#labels=['dog','cat','dog']


def prepare_image(image, target=target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255
    print(image.size)
    # return the processed image
    return image


def recognize(image):
    config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4, allow_soft_placement=True, device_count = {'CPU' : 1, 'GPU' : 0})
    session = tf.Session(config=config)
    K.set_session(session)
    model=load_model(modelpath)
    image=prepare_image(image)
    preds=model.predict(image)
    return preds

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(filename)
            image = Image.open(filename)
            
            result=list(recognize(image))
            print(result)
            result=result[0]
            #i=np.argmax(result)
            #res=labels[i]
            print(result)
            return jsonify({'cat':int(result[0]*100),'dog':int(result[1]*100)})
        
            
           
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

   
    
if __name__ == '__main__':
    app.run(debug=True)    
