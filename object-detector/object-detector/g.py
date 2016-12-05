import numpy as np
import imutils
import cv2
import sys
from skimage.feature import hog
from sklearn.externals import joblib
from config import *
import sys,os,re
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import skimage.data
import selectivesearch
import json
from PIL import Image

def background_subtract(filename):
    cap = cv2.VideoCapture(filename)
    while True:
        print "loaded"
        ret, frame = cap.read()
        if ret == False:
            break
        frame = imutils.resize(frame, width=(len(frame)))
        img_lbl, regions = selectivesearch.selective_search(frame, scale=500, sigma=0.9, min_size=10)
        print regions
        a = json.loads(regions)
        print a
        if(len(a) > 10):
            a = a[:10]
        for c in a:
            if c['size'] > 2000:
                continue
            (x, y, w, h) = c['rect']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            image = frame[y:y+h,x:x+w]
            cv2.imwrite('image.png', image)
            image = Image.open('image.png').convert('L')
            image = image.resize((100,100),Image.ANTIALIAS)
            data = np.asarray(image.getdata()).reshape((100,100))
            fd = hog(data, orientations=9, pixels_per_cell=(8,8),
            cells_per_block=(3, 3), visualise=False)
            prediction=clf.predict(fd)
            image = frame[y:y + h/2, x : x + w]
            image = Image.open('image.png').convert('L')
            image = image.resize((100,100),Image.ANTIALIAS)
            data = np.asarray(image.getdata()).reshape((100,100))
            fd = hog(data, orientations=9, pixels_per_cell=(8,8),
            cells_per_block=(3, 3), visualise=False)
            prediction2=clf.predict(fd)
            if(prediction2[0] == 4 or prediction2[0] == 3):
                prediction = prediction2
            if (prediction[0]==4):
                print "Bicycle"
                cv2.putText(frame,'Bicycle',(x,y),cv2.FONT_HERSHEY_PLAIN , 1,(255,255,255),2,cv2.LINE_AA)
            elif (prediction[0]==1):
                print "Car"
                cv2.putText(frame,'Car',(x,y), cv2.FONT_HERSHEY_PLAIN , 1,(255,255,255),2,cv2.LINE_AA)
            elif (prediction[0]==2):
                print "Person"
                cv2.putText(frame,'Person',(x,y),cv2.FONT_HERSHEY_PLAIN , 1,(255,255,255),2,cv2.LINE_AA)
            elif (prediction[0]==3):
                print "Motorcycle"
                cv2.putText(frame,'MOtorcycle',(x,y),cv2.FONT_HERSHEY_PLAIN , 1,(255,255,255),2,cv2.LINE_AA)
            else:
                cv2.putText(frame,'Rickshaw',(x,y),cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255),2,cv2.LINE_AA)
                print "rickshaw"
        j += 1
        text = "Occupied"
        cv2.imshow("Security Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        clf = joblib.load(model_path)
        background_subtract(filename)
