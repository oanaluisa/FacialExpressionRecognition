import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import pickle
import cv2
from keras.models import load_model
from PIL import Image
from flask import request
from flask import jsonify
import tensorflow as tf
from keras.preprocessing import image

#loading models
# load all the models
haar = cv2.CascadeClassifier('../model/haarcascade_frontalface_default.xml')

#model
model1 = load_model('./model/MODEL_1_FER2013_7.h5')
model2 = load_model('./model/MODEL_2_FER2013_7.h5')
model3 = load_model('./model/MODEL_1_ALL_7.h5')
model4 = load_model('./model/MODEL_2_ALL_7.h5')

models = [model1, model2, model3, model4]

print('Model loaded sucessfully', models)

# settins
font = cv2.FONT_HERSHEY_SIMPLEX

label_dict = {0:'Furie',1:'Dezgust',2:'Frica',3:'Fericire',4:'Neutru',5:'Suparare',6:'Surprindere'}

def detect_faces(img):
    rez = []
    faces = haar.detectMultiScale(img, 1.3, 5)

    if faces == ():
        return False

    for (x, y, w, h) in faces:
        fc = img[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48))
        for model in models:
            result = model.predict(roi[np.newaxis, :, :, np.newaxis])
            result = list(result[0])
            img_index = result.index(max(result))
            pred = label_dict[img_index]
            rez.append(pred)

    return rez    


def predict_model(path, filename):
    img_name = './static/uploads/' + filename     
    detect_img = cv2.imread(img_name)
    gray_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)

    if not detect_faces(gray_img):
        return False

    pred = detect_faces(gray_img)
    return pred 