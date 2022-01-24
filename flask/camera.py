import cv2
import numpy as np
from keras.models import load_model
import time

facec = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
model = load_model('./model/MODEL_2_ALL_7.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

label_dict = {0:'Furie',1:'Dezgust',2:'Frica',3:'Fericire',4:'Neutru',5:'Suparare',6:'Surprindere'}

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        if faces is ():
            text = 'Nu s-a gasit fata.'
            cv2.putText(fr, text, (200, 200), font, 1, (255, 255, 0), 2)

        for (x, y, w, h) in faces:
            print('in for')
            fc = gray_fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            result = model.predict(roi[np.newaxis, :, :, np.newaxis])
            result = list(result[0])
            img_index = result.index(max(result))
            pred = label_dict[img_index]
            print('label dict', label_dict[img_index])
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()