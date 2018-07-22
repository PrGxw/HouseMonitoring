import cv2
from server.align_custom import AlignCustom
from server.face_feature import FaceFeature
from server.mtcnn_detect import MTCNNDetect
from server.tf_graph import FaceRecGraph
import argparse
import sys
import json
import numpy as np
import time
import matplotlib.pyplot as plt
def findPeople(features_arr, positions, thres = 0.6, percent_thres = 70):
    '''
    :param features_arr: a list of 128d Features of all faces on screen
    :param positions: a list of face position types of all faces on screen
    :param thres: distance threshold
    :return: person name and percentage
    '''
    f = open('./facerec_128D.txt','r')
    data_set = json.loads(f.read());
    returnRes = [];
    for (i,features_128D) in enumerate(features_arr):
        result = "Unknown";
        smallest = sys.maxsize
        for person in data_set.keys():
            person_data = data_set[person][positions[i]];
            for data in person_data:
                distance = np.sqrt(np.sum(np.square(data-features_128D)))
                if(distance < smallest):
                    smallest = distance;
                    result = person;
        percentage =  min(100, 100 * thres / smallest)
        if percentage <= percent_thres :
            result = "Unknown"
        returnRes.append((result,percentage))
    return returnRes

FRGraph = FaceRecGraph();
aligner = AlignCustom();
extract_feature = FaceFeature(FRGraph)
face_detect = MTCNNDetect(FRGraph, scale_factor=2); #scale_factor, rescales image for faster detection

print("[INFO] camera sensor warming up...")
vs = cv2.VideoCapture(0); #get input from webcam

_,frame = vs.read();
rects, landmarks = face_detect.detect_face(frame,80);#min face size is set to 80x80
aligns = []
positions = []
for (i, rect) in enumerate(rects):
    aligned_face, face_pos = aligner.align(160, frame, landmarks[i])
    if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
        aligns.append(aligned_face)
        positions.append(face_pos)
    else:
        print("Align face failed")  # log
if(len(aligns) > 0):
    features_arr = extract_feature.get_features(aligns)
    recog_data = findPeople(features_arr,positions);
    for (i,rect) in enumerate(rects):
        cv2.rectangle(frame,(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(255,0,0)) #draw bounding box for the face
        cv2.putText(frame,recog_data[i][0]+" - "+str(recog_data[i][1])+"%",(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)




#
# cv2.imshow("frame", frame)
# key = cv2.waitKey(1) & 0xFF
# if key == ord("q"):
#     print('Ha!')
# time.sleep(10)
# vs.release()
# cv2.destroyAllWindows()
