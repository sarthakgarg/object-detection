import json
import numpy as np
import cv2
import sys
import imutils

# datafiles = ['datasample1.json','input_video_sample1.json','input_video_sample2.json','input_video_sample3.json','nov92015-1.json','nov92015-2.json']
# videofiles = ['datasample1.mov','input_video_sample1.mov','input_video_sample2.mov','input_video_sample3.mov','nov92015-1.dav','nov92015-2.dav']
datafiles = ['datasample1.json']
videofiles = ['datasample1.mov']
red = 2
j = 1

for datafile,videofile in zip(datafiles,videofiles):
        frame_dict = {}

        with open(datafile) as json_data:
                d = json.load(json_data)
                for item in d:
                    label = d[item]["label"]
                    for frame in d[item]["boxes"]:
                        if not frame in frame_dict:
                            frame_dict[frame] = []
                        # print frame, label
                        temp_list = [d[item]["boxes"][frame]["xtl"], d[item]["boxes"][frame]["ytl"], d[item]["boxes"][frame]["xbr"], d[item]["boxes"][frame]["ybr"], label, d[item]["boxes"][frame]["occluded"], d[item]["boxes"][frame]["outside"]]
                        frame_dict[frame].append(temp_list)


        cap = cv2.VideoCapture(videofile)
        i = 1
        k = 1
        while True:
            i+=1
            ret, frame = cap.read()
            if ret == False:
                break
            if(i%30 != 1):
                continue

            cp = np.copy(frame)

            if str(i) in frame_dict:
                for temp_list in frame_dict[str(i)]:
                    xtl = temp_list[0]
                    ytl = temp_list[1]
                    xbr = temp_list[2]
                    ybr = temp_list[3]
                    label = temp_list[4]

                    if (xbr - xtl) < 50 or (ybr - ytl) < 50:
                        continue
                    if temp_list[5]==1 or temp_list[6]==1:
                        continue
                    obj = cp[ytl:ybr, xtl:xbr]
                    width = len(obj[0])
                    height = len(obj)
                    obj = cv2.resize(obj, (int(width/red),int(height/red)))
                    cv2.imwrite('./' + label + '/' + str(j) + '.png', obj)
                    cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
                    j += 1

            frame = imutils.resize(frame, width=500)
#            cv2.namedWindow('detected objects', cv2.WINDOW_OPENGL)
#            cv2.imshow('detected objects', frame)
            if cv2.waitKey(2) & 0xFF == ord('q') :
                    break

        cap.release()
        cv2.destroyAllWindows()
        print j
