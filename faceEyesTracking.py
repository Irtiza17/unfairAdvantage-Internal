from statistics import mean
import cv2
import mediapipe as mp
from globalFunctions import *


def faceDetect(frame,landmarks):
    frame_h,frame_w,_ = frame.shape
    for id,landmark in enumerate(landmarks):
    # for id,landmark in enumerate(landmarks[474:478]):
        x = int( landmark.x * frame_w)
        y = int( landmark.y * frame_h)
        z = landmark.z
        cv2.putText(frame, f'{id}',(x,y), cv2.FONT_HERSHEY_DUPLEX, 0.25, (0, 255, 200), 1)
    # cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
        # cv2.circle(frame,(x,y),2,(0,0,255),2)


def eyeDetection(frame,landmarks):

    #Function Detects eyes and returns center of eye for focus tracking

    frame_h,frame_w,_ = frame.shape
    lix=[]
    liy=[]
    rix=[]
    riy=[]
    #For Left Eye
    for i,landmark in enumerate(landmarks[474:478]): 
        lx = int(landmark.x * frame_w)
        lix.append(lx)
        ly = int(landmark.y * frame_h)
        liy.append(ly)
        z = landmark.z
    cv2.circle(frame,(int(mean(lix)),int(mean(liy))),2,(0,0,255),2)
    #For Right Eye
    for i,landmark in enumerate(landmarks[469:473]):
        rx = int(landmark.x * frame_w)
        rix.append(rx)
        ry = int(landmark.y * frame_h)
        riy.append(ry)
        z = landmark.z
    left_eye_center = {'x':int(mean(lix)),'y':int(mean(liy))}
    # right_eye_center = {'xright':int(mean(rix)),'yright':int(mean(riy))}
    state['left_eye_center'] = left_eye_center
    stateMaintain(state)
    cv2.circle(frame,(int(mean(rix)),int(mean(riy))),2,(0,0,255),2)
    return left_eye_center
