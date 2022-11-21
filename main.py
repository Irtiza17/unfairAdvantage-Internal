import csv
import copy
import itertools
import depthai
import pandas as pd
import cv2 as cv
import numpy as np
import mediapipe as mp
from model import KeyPointClassifier
from scoring import score,secondScore, videoMapping
import datetime
import subprocess
import sys
from PIL import Image,ImageGrab
import time
import math



camera = "cam" #Camera Values can be "cam","othercam","video","oakD", depending on income stream source.
total_video_dur = 2100
show_live = True
video_display = False
video_file_name = "demovideo.mp4"
camera_to_use = 1
bounding_box = (0,0,800,800)


def main():

    HeadModel = True
    focusModel = True
    emotionModel = True
    pointingModel = True
    waveModel = True
    handModel = True

    

    model_path ='model/keypoint_classifier/keypoint_classifier.tflite' # Focus Model Path 
    model_path2 ='model/keypoint_classifier/keypoint_classifier2.tflite' # Emotion Model Path
    model_path3 ='model/keypoint_classifier/keypoint_classifier3.tflite' # Head Model Path
    model_path4 = 'model/keypoint_classifier/keypoint_classifier4.tflite' # HandPointing Model Path

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

    filepath = 'scorelog/SingleFrameData.csv'  #csv file for score data of every frame.
    filepath2 = 'scorelog/PerSecondData.csv' # csv file for score data of every second.
    filepath3 = 'scorelog/Report.csv' # csv file for videos information and score mapping against videos.

    df = pd.DataFrame(columns=['Date','Time','Focus','Emotion','Head','Pointing','Waving','Hand Movement'])
    
    def rescaleFrame(frame, scale=1.8):
        # Images, Videos and Live Video
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)

        dimensions = (width,height)

        return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

    def calc_landmark_list(image, landmarks,ROI=False,hand = False):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []
        # Keypoint
        if ROI == False:
            if hand == False:
                for _, landmark in enumerate(landmarks.landmark):
                    landmark_x = min(int(landmark.x * image_width), image_width - 1)
                    landmark_y = min(int(landmark.y * image_height), image_height - 1)
                    landmark_point.append([landmark_x, landmark_y])
            elif hand == True:
                for landmark in enumerate(landmarks.landmark):
                    landmark_x = min(int(landmark.x * image_width), image_width - 1)
                    landmark_y = min(int(landmark.y * image_height), image_height - 1)
                    landmark_point.append([landmark_x, landmark_y])
            # for i in ROI:
                # landmark = landmarks.landmark[i]


        elif ROI != False:
            for i in ROI:
                landmark = landmarks.landmark[i]
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


    def draw_bounding_rect(use_brect, image, brect):
        if use_brect:
            # Outer rectangle
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                        (0, 255, 0), 2)

        return image


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


    def draw_info_text(image, focus_text='',emotion_text='',head_text='',pointing_text ='', wave_text='', move_text=''):
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
            cv.putText(image, info_text4, (5,y_coordinate+(3*y_spacing)),
                    cv.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1, cv.LINE_AA)

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
            info_text6 = 'Wave: N/A'
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

        return image


    def classificationLabel(modelPath,labelPath):
            classifier = KeyPointClassifier(modelPath)
            with open('model/keypoint_classifier/' + labelPath,encoding='utf-8-sig') as f:
                labels = csv.reader(f)
                labels = [row[0] for row in labels]
            return classifier,labels


    def inputStream(camera):
        if camera == 'cam':
            cap_device = camera_to_use
            cap_width = 1920
            cap_height = 1080
            # Camera preparation
            cap = cv.VideoCapture(cap_device)
            cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
        
        elif camera == 'video':
            cap_device = 'D:/Internship/Tasks/Training/Moving hand while focus/handwhilefocus (10).mp4'
            # cap_device = 'D:/Internship/Tasks/Training/Focussed on Screen/1038007019-preview.mp4'
            cap_width = 1920
            cap_height = 1080
            # Camera preparation
            cap = cv.VideoCapture(cap_device)
            cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

        elif camera == 'oakD':
            pipeline= oakD()
            with depthai.Device(pipeline) as device:
                q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                while True:
                # Process Key (ESC: end)
                    key = cv.waitKey(10)
                    if key == 27:  # ESC
                        break
                    # Camera capture
                    in_rgb = q_rgb.get()
                    image = in_rgb.getCvFrame()
        else:
            cap = None
          
        return cap


    def oakD():
        pipeline = depthai.Pipeline()
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setPreviewSize(300, 300) 
        cam_rgb.setInterleaved(False)
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)
        return pipeline


    def length(a,b):
        x1 = a[0] 
        y1 = a[1] 
        x2 = b[0] 
        y2 = b[1]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


    def coordinatesCalc(landmarks, index):
        point = landmarks.landmark[index]
        x = int(point.x * image_width)
        y = int(point.y * image_height)
        return (x,y)


    def hi5Detection(landmarks, image):
        indexFingerTip = coordinatesCalc(landmarks, 8)
        indexFingerPip = coordinatesCalc(landmarks, 6)
        middleFingerTip = coordinatesCalc(landmarks, 12)
        middleFingerPip = coordinatesCalc(landmarks, 10)
        ringFingerTip = coordinatesCalc(landmarks, 16)
        ringFingerPip = coordinatesCalc(landmarks, 14)
        pinkyFingerTip = coordinatesCalc(landmarks, 20)
        pinkyFingerPip = coordinatesCalc(landmarks, 18)

        if indexFingerTip[1] < indexFingerPip [1]:
            if middleFingerTip[1] < middleFingerPip[1]:
                if ringFingerTip[1] < ringFingerPip[1]:
                    if pinkyFingerTip[1] < pinkyFingerPip[1]:
                        cv.putText(image, 'Hi5', (25,70),
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1, cv.LINE_AA)
                        return True
    
    # Set the font and spacing of text in black box
    font_size = 0.7
    y_coordinate = 30
    y_spacing = 35

    # Initial frame position of left and right hand (index finger tip)
    start_waveFrame_left = (0,0)
    start_waveFrame_right = (0,0)

    # Initial frame position of pose landmarks
    start_right_wrist = (0,0)
    start_right_elbow = (0,0)
    start_right_shoulder = (0,0)
    start_left_wrist = (0,0)
    start_left_elbow = (0,0)
    start_left_shoulder = (0,0)

    mode = 0

    # Face Model load
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) 

    # Hand Model Load
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    # Pose Model Load
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.8)


    # Initializing keypoint classifiers and labels list
    focus_keypoint_classifier,focus_labels = classificationLabel(model_path,'focus_labels.csv')
    emotion_keypoint_classifier,emotion_labels = classificationLabel(model_path2,'emotion_labels.csv')
    head_keypoint_classifier,head_labels = classificationLabel(model_path3,'head_labels.csv')
    pointing_keypoint_classifier,pointing_labels = classificationLabel(model_path4,'pointing_labels.csv')


    use_brect = True    

    
        
    cap = inputStream(camera)
    start = 0 # While start = 0, it will continue to provide stream, at decided time (21s), it will stop the stream by changing its value.

    if video_display:

        systemOS = get_OS_platform()

        if systemOS == "Windows":
            child_process = subprocess.Popen(["C:/Program Files (x86)/VideoLAN/VLC/vlc.exe","videos/" + video_file_name])
            print("Windows")
        elif systemOS == "OS X":
            child_process = subprocess.Popen(["/Applications/VLC.app/contents/MacOS/vlc", "videos/" + video_file_name])
            print("Mac OS")
        else: print("Other OS")

    # used to record the time when we processed last frame
    prev_frame_time = 0
    
    # used to record the time at which we processed current frame
    new_frame_time = 0
    now = datetime.datetime.now()


    while start == 0:
        next = datetime.datetime.now()

        new_frame_time = time.time()
 
        # Calculating the fps
 
        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
    
        # converting the fps into integer
        fps = str(int(fps))
    
        # converting the fps to string so that we can display it on frame
        # by using putText function


        # print('Next1',next)
        timedif = (next - now).seconds
        # Process Key (ESC: end)
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        if timedif > total_video_dur:
            start = 1
        # Camera capture
        if camera == 'othercam':
            img = ImageGrab.grab(bbox=bounding_box) #bbox specifies specific region (bbox= x,y,width,height)
            image = np.array(img)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        elif camera == 'cam' or camera == 'video':
            ret, image = cap.read()
            try:
                if camera == 'video':
                    image = rescaleFrame(image, scale=2)
            except:
                pass
            if not ret:
                break
        
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        image_width2, image_height2 = image.shape[1], image.shape[0] # for Detections use
        tab = int((((image_width2-290)/2)*0.5)+290) # for Detection use
        tab2 = int((((image_width2-290)/2)*0.5)+tab) # for Detection use

        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        face_results = face_mesh.process(image)
        hand_results = hands.process(image)
        pose_results = pose.process(image)
        image.flags.writeable = True

        if face_results.multi_face_landmarks is not None:
            for face_landmarks in face_results.multi_face_landmarks:
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, face_landmarks)

                # Landmark calculation
                if focusModel == True:
                    focus_landmark_list = calc_landmark_list(debug_image, face_landmarks,ROI)
                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list1 = pre_process_landmark(
                    focus_landmark_list)
                    facial_focus_id = focus_keypoint_classifier(pre_processed_landmark_list1)
                    eyeFocusVal = focus_labels[facial_focus_id]
                else:
                    eyeFocusVal = 'N/A'

                if emotionModel == True:
                    emotion_landmark_list = calc_landmark_list(debug_image, face_landmarks)
                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list2 = pre_process_landmark(
                    emotion_landmark_list)
                    facial_emotion_id = emotion_keypoint_classifier(pre_processed_landmark_list2)
                    emotionVal = emotion_labels[facial_emotion_id]
                else:
                    emotionVal = 'N/A'

                if HeadModel == True:
                    head_landmark_list = calc_landmark_list(debug_image, face_landmarks,ROI2)
                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list3 = pre_process_landmark(
                    head_landmark_list)
                    head_id = head_keypoint_classifier(pre_processed_landmark_list3)
                    headVal = head_labels[head_id]
                else:
                    headVal = 'N/A'

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_info_text(
                                debug_image,eyeFocusVal,emotionVal,headVal)
                # image_width2, image_height2
                cv.rectangle(debug_image, (290, 0), (tab,30),(0,255,0), -1)
                cv.putText(debug_image, "FACE DETECTED", (310,23),cv.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1, cv.LINE_AA)
        else:
            eyeFocusVal = 'N/A'
            emotionVal = 'N/A'
            headVal = 'N/A'
            cv.rectangle(debug_image, (290, 0), (tab,30),(0,0,255), -1)
            cv.putText(debug_image, "CANNOT DETECT FACE", (310,23),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        if hand_results.multi_hand_landmarks is not None:
            for hand_no, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                if pointingModel == True:
                    mpDraw.draw_landmarks(debug_image, hand_landmarks, mpHands.HAND_CONNECTIONS)
                    pointing_landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list4 = pre_process_landmark(
                    pointing_landmark_list)
                    pointing_id = pointing_keypoint_classifier(pre_processed_landmark_list4)
                    pointingVal = pointing_labels[pointing_id]
                    debug_image = draw_info_text(
                                debug_image,eyeFocusVal,emotionVal,headVal,pointingVal)

                else:
                    pointingVal = 'N/A'

                if waveModel == True:
                    image_width, image_height = image.shape[1], image.shape[0]

                    if hand_no == 0:
                        next_frame_left = coordinatesCalc(hand_landmarks,8)
                        difference = length(start_waveFrame_left,next_frame_left)
                        
                        if difference > 30 and hi5Detection(hand_landmarks, image):
                            waveVal_0 = 'waving'
                        else:
                            waveVal_0 = 'not waving'
                        start_waveFrame_left = next_frame_left
                    if hand_no == 1:
                        next_frame_right = coordinatesCalc(hand_landmarks,8)
                        difference = length(start_waveFrame_right,next_frame_right)
                        if difference > 30 and hi5Detection(hand_landmarks, image):
                            waveVal_1 = 'waving'
                        else:
                            waveVal_1 = 'not waving'
                        start_waveFrame_right = next_frame_right
                    else:
                        waveVal_1 = 'not waving'

                    if waveVal_0 == 'waving' or waveVal_1 == 'waving':
                        waveVal = 'waving'
                    else:
                        waveVal = 'not waving'
                    debug_image = draw_info_text(
                            debug_image,eyeFocusVal,emotionVal,headVal,pointingVal,waveVal)

                else:
                    waveVal = 'N/A'

                cv.rectangle(debug_image, (tab+1, 0), (tab2,30),(0,255,0), -1)
                cv.putText(debug_image, "HANDS DETECTED", (tab+20,23),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1, cv.LINE_AA)

        else:
            pointingVal = 'N/A'
            waveVal = 'N/A'
            cv.rectangle(debug_image, (tab+1, 0), (int(tab2),30),
                    (0, 0,255), -1)
            cv.putText(debug_image, "CANNOT DETECT HANDS", (tab+20,23),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)    

        if pose_results.pose_landmarks is not None:
            if handModel == True:
                image_width, image_height = image.shape[1], image.shape[0]

                # Calculating next coordinates for pose landmarks
                next_right_wrist = coordinatesCalc(pose_results.pose_landmarks,15)
                next_right_elbow = coordinatesCalc(pose_results.pose_landmarks,13)
                next_right_shoulder = coordinatesCalc(pose_results.pose_landmarks,11)
                next_left_wrist = coordinatesCalc(pose_results.pose_landmarks,16)
                next_left_elbow = coordinatesCalc(pose_results.pose_landmarks,14)
                next_left_shoulder = coordinatesCalc(pose_results.pose_landmarks,12)

                # Calculating difference of initial and next frame
                diff_right_wrist = length(start_right_wrist, next_right_wrist)
                diff_left_wrist = length(start_left_wrist, next_left_wrist)
                diff_right_elbow = length(start_right_elbow,next_right_elbow)
                diff_left_elbow = length(start_left_elbow, next_left_elbow)

                # Checking condition of movement
                if diff_right_wrist > 17 and next_right_wrist[1]/image_height< 0.98:
                    moveVal_0 = 'moving'
                else:
                    moveVal_0 = 'not moving'
                start_right_wrist = next_right_wrist

                if diff_left_wrist > 17 and next_left_wrist[1]/image_height< 0.98:
                    moveVal_1 = 'moving'
                else:
                    moveVal_1 = 'not moving'
                start_left_wrist = next_left_wrist
            
                if moveVal_0 == 'moving' or moveVal_1 == 'moving':
                    moveVal = 'moving'
                else:
                    moveVal = 'not moving'
                debug_image = draw_info_text(
                            debug_image,eyeFocusVal,emotionVal,headVal,pointingVal,waveVal,moveVal)
            else:
                moveVal = 'N/A'
                
        # Drawing part
        try:
            # Scoring part
            cv.putText(debug_image, f"FPS: {fps}", (5,y_coordinate+(7*y_spacing)),cv.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1, cv.LINE_AA)
            df = score(eyeFocusVal,emotionVal,headVal,pointingVal,waveVal,moveVal,next,df)
        except Exception as e:
            print("the error is :", e)
            

    
        # Screen reflection
        if show_live:
            cv.imshow('Facial Emotion and focus Recognition', debug_image)
        # if timedif >= 10 and ret2:


    df.to_csv(filepath,index=False)
    df2 = secondScore(df)
    df2.to_csv(filepath2,index=False)
    df3 = videoMapping(df2)
    df3.to_csv(filepath3,index=False)
    # cap.release()
    cv.destroyAllWindows()
    child_process.terminate()


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

if __name__ == "__main__":
    main()