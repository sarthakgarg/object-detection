
import json
import numpy as np
import cv2
import sys
import imutils
import pickle
import os
import numpy as np
# datafiles = ['datasample1.json','input_video_sample1.json','input_video_sample2.json','input_video_sample3.json','nov92015-1.json','nov92015-2.json']
# videofiles = ['datasample1.mov','input_video_sample1.mov','input_video_sample2.mov','input_video_sample3.mov','nov92015-1.dav','nov92015-2.dav']
directories = ['Car','Bicycle','Motorcycle','Number-plate','Autorickshaw','Rickshaw','Person']
# directories = ['Number-plate']

label_list = []
desc = []
for d in directories:
    for f in os.listdir(d):
        image = cv2.imread(d + '/' + f,0)
        sift = cv2.SIFT()
        key1, des1 = sift.detectAndCompute(image, None)
        if des1 != None:
            no_of_features = min(len(des1), 200)


            desc.append(des1[:no_of_features])
            label_list.append(d)
a = np.array(desc)
pickle.dump(a,open("sift_features_datasample1.p","wb"))
pickle.dump(label_list, open("sift_label_datasample1.p", "wb"))
