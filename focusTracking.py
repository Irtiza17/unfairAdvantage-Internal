from statistics import mean
import cv2
import mediapipe as mp
from globalFunctions import *
import numpy as np


def eyesOpenDetection(frame,landmarks):
    left = [landmarks[145],landmarks[159]]
    right = [landmarks[373],landmarks[386]]

    if ((left[0].y - left[1].y) < 0.015) and ((right[0].y - right[1].y) < 0.015):
        state['open']=0
        cv2.putText(frame,'EYES CLOSED',(200,100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        stateMaintain(state)
    else:
        state['open']=1      
        stateMaintain(state)

def headTurned(frame,eyes_center):
    if eyes_center['xleft'] < 290:
        cv2.putText(frame,'Looking left. Not focused',(200,100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
    elif eyes_center['xleft'] > 410:
        cv2.putText(frame,'Looking right. Not focused',(200,100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
    elif eyes_center['yleft'] > 320:
        cv2.putText(frame,'Looking down. Not focused',(200,100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
    elif eyes_center['yleft'] < 135:

        cv2.putText(frame,'Looking up. Not focused',(200,100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
    else:
        cv2.putText(frame,'Focused',(200,100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)  
def eyeIrisMismatch(frame,landmarkPoints):
    height,width,_ = frame.shape
    if landmarkPoints:
        for i in range(len(landmarkPoints)):
            mesh_points=np.array([np.multiply([p.x, p.y], [width, height]).astype(int) for p in landmarkPoints[i].landmark])

            # Mulitply [p.x, p.y] and [width, height]
            # and make the data type to be defined in astype
            # p.x and p.y will extract the corresponding x y coordinates

            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            center_left_iris = np.array([l_cx, l_cy], dtype=np.int32)
            cv2.circle(frame, center_left_iris, int(l_radius), (0,0,255), 1, cv2.LINE_AA)

            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_EYE])
            center_left_eye = np.array([l_cx, l_cy], dtype=np.int32)
            cv2.circle(frame, center_left_eye, int(l_radius), (0,255,0), 1, cv2.LINE_AA)


            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_right_iris = np.array([r_cx, r_cy], dtype=np.int32)
            cv2.circle(frame, center_right_iris, int(r_radius), (0,0,255), 1, cv2.LINE_AA)


            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_EYE])
            center_right_eye = np.array([r_cx, r_cy], dtype=np.int32)
            cv2.circle(frame, center_right_eye, int(r_radius), (0,255,0), 1, cv2.LINE_AA)
        
            threshold = 1.5

            if (distance((center_left_iris),(center_left_eye))>threshold and distance((center_right_eye),center_right_iris)>threshold):
                cv2.putText(frame, 'NOT FOCUSED', (10,30), cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
                print('NOT FOCUSED')
            else:
                cv2.putText(frame, 'FOCUSED', (10,30), cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)