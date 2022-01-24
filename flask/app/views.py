from flask import render_template, request
from flask import redirect, url_for
import os
from PIL import Image
from app.utils import predict_model

UPLOAD_FLODER = 'static/uploads'

def base():
    return render_template('base.html')

def index():
    return render_template('index.html')

def faceapp():
    return render_template('faceapp.html')


def getwidth(path):
    img = Image.open(path)
    size = img.size # width and height
    aspect = size[0]/size[1] # width / height
    w = 300 * aspect
    return int(w)

def expression():
    if request.method == "POST":
        f = request.files['image']
        filename=  f.filename
        path = os.path.join(UPLOAD_FLODER,filename)
        f.save(path)
        w = getwidth(path)
        # prediction (pass to pipeline model)

        if not predict_model(path,filename):
            predModel1 = 'Nu s-a gasit fata.'
            predModel2 = 'Nu s-a gasit fata.'
            predModel3 = 'Nu s-a gasit fata.'
            predModel4 = 'Nu s-a gasit fata.'
        else:
            text = predict_model(path,filename)
            predModel1 = text[0]
            predModel2 = text[1]
            predModel3 = text[2]
            predModel4 = text[3]
        
        return render_template('expression.html',fileupload=True,img_name=filename, w=w, text=predModel1, text2=predModel2, text3=predModel3, text4=predModel4)

    return render_template('expression.html', fileupload=False)

def realTime():
    return render_template('realTime.html')