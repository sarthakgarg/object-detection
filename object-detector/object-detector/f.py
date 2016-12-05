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
from PIL import Image
def background_subtract(filename):
    cap = cv2.VideoCapture(filename)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold = 30, detectShadows = True)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5),(3,3))
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=(len(frame)))
    gray = cv2.blur(frame,(3,3))
    fgmask = fgbg.apply(gray)
    i = 1
    j = 1
    k = 1
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = imutils.resize(frame, width=(len(frame)))
        gray = cv2.blur(frame,(3,3))
        fgmask = fgbg.apply(gray)
        thresh = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.threshold(thresh, 100, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow('thresh without erosion', thresh)
        (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 2000:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
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
                cv2.putText(frame,'Motorcycle',(x,y),cv2.FONT_HERSHEY_PLAIN , 1,(255,255,255),2,cv2.LINE_AA)
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
