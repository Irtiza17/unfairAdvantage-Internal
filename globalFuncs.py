import cv2 as cv
import copy
import subprocess
import sys
import itertools
import numpy as np
import csv
import math
import time
import datetime

ROI = [246, 161, 160, 159, 158, 157, 173, 33, 7, 163, 144, 145, 153, 154, 155, 133, 473, 474, 475, 476, 
        477, 466, 388, 387, 386, 385, 384, 398, 263, 249, 390, 373, 374, 380, 381, 382, 362, 468,469, 470, 
        471, 472]

noseTip= [1]
noseBottom= [2]
noseRightCorner= [98]
noseLeftCorner= [327]

rightCheek= [205]
leftCheek= [425]

silhouette= [
10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
]

ROI2 =  silhouette + noseTip + noseBottom + noseRightCorner + noseLeftCorner + rightCheek + leftCheek



def camSetup(inputSource,sourcePath=0):
    if inputSource == 'cam':
        cap_device = sourcePath
    elif inputSource == 'video':
        cap_device = sourcePath
    cap_width = 1920
    cap_height = 1080    
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    dimensions = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    return cap,dimensions

def startTimeFunc():
    startTime = datetime.datetime.now()
    frameTime = datetime.datetime.now()
    return startTime,frameTime

def imgManip(img,inputSource,dimensions):
    if inputSource == 'video':
        img= rescaleFrame(img, dimensions)
    debug_img = copy.deepcopy(cv.flip(img,1))
    img = cv.flip(img, 1)  # Mirror display
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img,debug_img

def get_OS_platform():
    platforms = {
        'linux' : 'Linux',
        'darwin' : 'OS X',
        'win32' : 'Windows',
        'nt' : 'Windows',
        'win64' : 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform

    return platforms[sys.platform]

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]
        # print("landmark Point is " , landmark_point)

        landmark_array = np.append(landmark_array, landmark_point, axis=0)
        # print("landmark array is " , landmark_array)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def draw_bounding_rect(image, brect,use_brect=True):
        if use_brect:
            # Outer rectangle
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                        (0, 255, 0), 2)

        return image

def videoDisplayFunc(video_file_name,video_display = True):
    if video_display:
        systemOS = get_OS_platform()
        if systemOS == "Windows":
            child_process = subprocess.Popen(["C:/Program Files (x86)/VideoLAN/VLC/vlc.exe","videos/" + video_file_name])
            print("Windows",type(child_process))
        elif systemOS == "OS X":
            child_process = subprocess.Popen(["/Applications/VLC.app/contents/MacOS/vlc", "videos/" + video_file_name])
            print("Mac OS")
        else: print("Other OS")
    
        return child_process

def draw_info_text(fps,image,focus_text='',emotion_text='',head_text='', pointing_text='',wave_text='',move_text=''):
    y_coordinate = 30
    y_spacing = 35
    font_size = 0.7

    cv.rectangle(image, (0, 0), (290,y_coordinate+(7*y_spacing)+5),
                (0, 0, 0), -1)

    if focus_text != "":
        info_text = 'Gaze: ' + focus_text
        cv.putText(image, info_text, (5,y_coordinate),
                cv.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1, cv.LINE_AA)
    
    if emotion_text != "":
        info_text2 = 'Emotion: ' + emotion_text
        cv.putText(image, info_text2, (5,y_coordinate+(1*y_spacing)),
                cv.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1, cv.LINE_AA)
                
    if head_text != "":
        info_text3 = 'Head: ' + head_text
        cv.putText(image, info_text3, (5,y_coordinate+(2*y_spacing)),
                cv.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1, cv.LINE_AA)

    if pointing_text != "":
        info_text4 = 'Pointing: ' + pointing_text
        cv.putText(image, info_text4, (5,y_coordinate+(3*y_spacing)),
                cv.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1, cv.LINE_AA)
    else:
        info_text4 = 'Pointing: N/A'

    if wave_text != "":
            info_text5 = 'Wave: ' + wave_text
            cv.putText(image, info_text5, (5,y_coordinate+(4*y_spacing)),
                    cv.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1, cv.LINE_AA)
    else:
        info_text5 = 'Wave: N/A'
        cv.putText(image, info_text5, (5,y_coordinate+(4*y_spacing)),
                cv.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1, cv.LINE_AA)
    if move_text != "":
            info_text6 = 'Move: ' + move_text
            cv.putText(image, info_text6, (5,y_coordinate+(5*y_spacing)),
                    cv.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1, cv.LINE_AA)
    else:
        info_text6 = 'Move: N/A'
        cv.putText(image, info_text6, (5,y_coordinate+(5*y_spacing)),
                cv.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1, cv.LINE_AA)

    val = []
    if focus_text == 'Focused':
        val.append(1)
    if emotion_text == 'Positive':
        val.append(1)
    if head_text == 'Center':
        val.append(1)
    if pointing_text == 'Pointing':
        val.append(1)
    if wave_text == 'waving':
        val.append(1)
    if move_text == 'moving':
        val.append(1)


    cv.putText(image, f"Score: {sum(val)}/6", (5,y_coordinate+(6*y_spacing)),
                cv.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(image, f"FPS: {fps}", (5,y_coordinate+(7*y_spacing)),cv.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1, cv.LINE_AA)
    return image

def calc_landmark_list(image, landmarks,model):

    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []
    # Keypoint
    
        # for i in ROI:
            # landmark = landmarks.landmark[i]
    if model == 'Focus':
        for i in ROI:
            landmark = landmarks.landmark[i]
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])
            
    elif model == 'Head':
        for i in ROI2:
            landmark = landmarks.landmark[i]
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])

    elif model == "Hands" or "Emotion":
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def labelReader(labelPath):
    with open('model/keypoint_classifier/' + labelPath,encoding='utf-8-sig') as f:
        labels = csv.reader(f)
        labels = [row[0] for row in labels]
    
    return labels

def coordinatesCalc(image,landmarks, index):
        image_width, image_height = image.shape[1], image.shape[0]
        point = landmarks.landmark[index]
        x = int(point.x * image_width)
        y = int(point.y * image_height)
        return (x,y)

def length(a,b):
    x1 = a[0] 
    y1 = a[1] 
    x2 = b[0] 
    y2 = b[1]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def fps(prev_frame_time):

    new_frame_time = datetime.datetime.now()
    fps = 1/(new_frame_time.timestamp()-prev_frame_time.timestamp())
    prev_frame_time = new_frame_time
    fps = str(int(fps))

    return fps, new_frame_time

def hi5Detection(landmarks, image):

    indexFingerTip = coordinatesCalc(image,landmarks, 8)
    indexFingerPip = coordinatesCalc(image,landmarks, 6)
    middleFingerTip = coordinatesCalc(image,landmarks, 12)
    middleFingerPip = coordinatesCalc(image,landmarks, 10)
    ringFingerTip = coordinatesCalc(image,landmarks, 16)
    ringFingerPip = coordinatesCalc(image,landmarks, 14)
    pinkyFingerTip = coordinatesCalc(image,landmarks, 20)
    pinkyFingerPip = coordinatesCalc(image,landmarks, 18)

    if indexFingerTip[1] < indexFingerPip [1]:
        if middleFingerTip[1] < middleFingerPip[1]:
            if ringFingerTip[1] < ringFingerPip[1]:
                if pinkyFingerTip[1] < pinkyFingerPip[1]:
                    cv.putText(image, 'Hi5', (25,70),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1, cv.LINE_AA)
                    return True

def outputDisplay(show,img,title):
    if show:
        cv.imshow(title,img)
    else:
        pass

def programTimingFunction(pre_time,vidDuration):
    cur_time = datetime.datetime.now()   
    timediff = (cur_time - pre_time).seconds
    if timediff <= vidDuration:
        start = True
    else:
        start = False
    return start

def rescaleFrame2(frame, scale=1.8):
    # Images, Videos and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def rescaleFrame(frame,dims):
    #600 , 480 
    #1280,720

    scale_x = 1920/dims[0]
    scale_y = 810/dims[1]
    # Images, Videos and Live Video
    width = int(frame.shape[1] * scale_x)
    height = int(frame.shape[0] * scale_y)

    dimensions = (width,height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)