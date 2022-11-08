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




camera = "cam" #Camera Values can be "cam","othercam","video","oakD", depending on income stream source.
total_video_dur = 210
show_live = True
video_display = False
video_file_name = "demovideo.mp4"
camera_to_use = 0
bounding_box = (0,0,800,800)

def main():

    HeadModel = True
    focusModel = True
    emotionModel = True
    pointingModel = True

    

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

    df = pd.DataFrame(columns=['Date','Time','Focus','Emotion','Head','Hand'])

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


    def draw_info_text(image, focus_text,emotion_text='',head_text='',pointing_text =''):
        cv.rectangle(image, (0, 0), (290,200),
                    (0, 0, 0), -1)

        if focus_text != "":
            info_text = 'Gaze: ' + focus_text
            cv.putText(image, info_text, (5,30),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv.LINE_AA)
        
        if emotion_text != "":
            info_text2 = 'Emotion: ' + emotion_text
            cv.putText(image, info_text2, (5,70),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv.LINE_AA)
                    
        if head_text != "":
            info_text3 = 'Head: ' + head_text
            cv.putText(image, info_text3, (5,110),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv.LINE_AA)

        if pointing_text != "":
            info_text4 = 'Pointing: ' + pointing_text
            cv.putText(image, info_text4, (5,150),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv.LINE_AA)
        else:
            info_text4 = 'Pointing: N/A'
            cv.putText(image, info_text4, (5,150),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv.LINE_AA)

        val = []
        if focus_text == 'Focused':
            val.append(1)
        if emotion_text == 'Positive':
            val.append(1)
        if head_text == 'Center':
            val.append(1)
        if pointing_text == 'Pointing':
            val.append(1)
        

        cv.putText(image, f"Score: {sum(val)}/4", (5,190),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv.LINE_AA)

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
            cap_device = 'videos/video1.mp4'
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
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils


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

    now = datetime.datetime.now()


    while start == 0:
        next = datetime.datetime.now()
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
            if not ret:
                break
        
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        face_results = face_mesh.process(image)
        hand_results = hands.process(image)
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

                if emotionModel == True:
                    emotion_landmark_list = calc_landmark_list(debug_image, face_landmarks)
                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list2 = pre_process_landmark(
                    emotion_landmark_list)
                    facial_emotion_id = emotion_keypoint_classifier(pre_processed_landmark_list2)

                if HeadModel == True:
                    head_landmark_list = calc_landmark_list(debug_image, face_landmarks,ROI2)
                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list3 = pre_process_landmark(
                    head_landmark_list)
                    head_id = head_keypoint_classifier(pre_processed_landmark_list3)

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_info_text(
                                debug_image,
                                focus_labels[facial_focus_id],emotion_labels[facial_emotion_id],head_labels[head_id])
        else:
            eyeFocusVal = 'Not Focused'
            emotionVal = 'Negative'
            headVal = 'Not Center'
            df = score(eyeFocusVal,emotionVal,headVal,next,df)
            cv.rectangle(debug_image, (0, 0), (1920,100),
                    (0, 0,255), -1)
            cv.putText(debug_image, "CANNOT DETECT FACE", (300,75),
                    cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

        if hand_results.multi_hand_landmarks is not None:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                if pointingModel == True:
                    mpDraw.draw_landmarks(debug_image, hand_landmarks, mpHands.HAND_CONNECTIONS)
                    pointing_landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list4 = pre_process_landmark(
                    pointing_landmark_list)
                    pointing_id = pointing_keypoint_classifier(pre_processed_landmark_list4)
                    debug_image = draw_info_text(
                                debug_image,
                                focus_labels[facial_focus_id],emotion_labels[facial_emotion_id],head_labels[head_id],pointing_labels[pointing_id]
                                )

        else:
            eyeFocusVal = 'Not Focused'
            emotionVal = 'Negative'
            headVal = 'Not Center'
            df = score(eyeFocusVal,emotionVal,headVal,next,df)
            cv.rectangle(debug_image, (0, 200), (1920,300),
                    (0, 0,255), -1)
            cv.putText(debug_image, "CANNOT DETECT HANDS", (300,275),
                    cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

                

        # Drawing part
        try:
            # debug_image = draw_info_text(
            #                     debug_image,
            #                     focus_labels[facial_focus_id],emotion_labels[facial_emotion_id],head_labels[head_id],pointing_labels[pointing_id]
            #                     )
            # Scoring part
            eyeFocusVal = focus_labels[facial_focus_id]
            emotionVal = emotion_labels[facial_emotion_id]
            headVal = head_labels[head_id]
            df = score(eyeFocusVal,emotionVal,headVal,next,df)
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