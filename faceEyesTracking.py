from statistics import mean
import cv2
import mediapipe as mp


def faceDetect(frame,landmarks):
    frame_h,frame_w,_ = frame.shape
    forehead = landmarks[10]
    lcheek = landmarks[234]
    chin = landmarks[152]
    rcheek = landmarks[352]
    x1 = int( lcheek.x * frame_w)
    y1 = int( forehead.y * frame_h)
    x2 = int( rcheek.x * frame_w)
    y2 = int( chin.y * frame_h)
    # for id,landmark in enumerate(landmarks):
    # for id,landmark in enumerate(landmarks[474:478]):
        # x = int( landmark.x * frame_w)
        # y = int( landmark.y * frame_h)
        # z = landmark.z
        # cv2.putText(frame, f'{id}',(x,y), cv2.FONT_HERSHEY_DUPLEX, 0.25, (0, 255, 200), 1)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)


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
    eye_center = {'xleft':int(mean(lix)),'yleft':int(mean(liy)),'xright':int(mean(rix)),'yright':int(mean(riy))}
    print(eye_center)
    cv2.circle(frame,(int(mean(rix)),int(mean(riy))),2,(0,0,255),2)
    return eye_center
