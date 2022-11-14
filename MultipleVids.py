import csv
import copy
import itertools
import pandas as pd
import cv2 as cv
import numpy as np
import mediapipe as mp
from model import KeyPointClassifier
import time
import os



Testing = 'Moving hand while focus'
dataset_path = os.listdir(f'Testing Videos/{Testing}')

HeadModel = True
focusModel = True
emotionModel = True
pointingModel = True


def main():
    show_live = True

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

    def rescaleFrame(frame, scale=1.8):
        # Images, Videos and Live Video
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)

        dimensions = (width,height)

        return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

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


    def draw_info_text(image, focus_text='',emotion_text='',head_text='',pointing_text =''):
        cv.rectangle(image, (0, 0), (290,230),
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


    def inputStream(vidpath):
        cap_device = vidpath
        # cap_device = 'D:/Internship/Tasks/Training/Focussed on Screen/1038007019-preview.mp4'
        cap_width = 1920
        cap_height = 1080
        # Camera preparation
        cap = cv.VideoCapture(cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    
        return cap

    
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


    # Initializing keypoint classifiers and labels list
    focus_keypoint_classifier,focus_labels = classificationLabel(model_path,'focus_labels.csv')
    emotion_keypoint_classifier,emotion_labels = classificationLabel(model_path2,'emotion_labels.csv')
    head_keypoint_classifier,head_labels = classificationLabel(model_path3,'head_labels.csv')
    pointing_keypoint_classifier,pointing_labels = classificationLabel(model_path4,'pointing_labels.csv')


    use_brect = True    

    for item in dataset_path:
        vids = (f'Testing Videos/{Testing}' + '/' +item)       
        cap = inputStream(vids)
        start = 0 # While start = 0, it will continue to provide stream, at decided time (21s), it will stop the stream by changing its value.

        # used to record the time when we processed last frame
        prev_frame_time = 0
        # used to record the time at which we processed current frame
        new_frame_time = 0
        count= 0
        while start == 0:
            #Fps calc
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))


            
            
            # Camera capture
            ret, image = cap.read()
            try:
                image = rescaleFrame(image)
            except Exception as e:
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
            if HeadModel == True or focusModel == True or emotionModel == True:
                face_results = face_mesh.process(image)
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
                    eyeFocusVal = 'Not Focused'
                    emotionVal = 'Negative'
                    headVal = 'Not Center'
                    # df = score(eyeFocusVal,emotionVal,headVal,next,df)
                    cv.rectangle(debug_image, (290, 0), (tab,30),(0,0,255), -1)
                    cv.putText(debug_image, "CANNOT DETECT FACE", (310,23),
                            cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

            if pointingModel == True:
                hand_results = hands.process(image)
                image.flags.writeable = True
                if hand_results.multi_hand_landmarks is not None:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                            mpDraw.draw_landmarks(debug_image, hand_landmarks, mpHands.HAND_CONNECTIONS)
                            pointing_landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                            # Conversion to relative coordinates / normalized coordinates
                            pre_processed_landmark_list4 = pre_process_landmark(
                            pointing_landmark_list)
                            pointing_id = pointing_keypoint_classifier(pre_processed_landmark_list4)
                            pointingVal = pointing_labels[pointing_id]
                            debug_image = draw_info_text(
                                        debug_image,eyeFocusVal,emotionVal,headVal
                                        ,pointingVal
                                        )
                            cv.rectangle(debug_image, (tab+1, 0), (tab2,30),(0,255,0), -1)
                            cv.putText(debug_image, "HANDS DETECTED", (tab+20,23),
                            cv.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1, cv.LINE_AA)
                else:
                    eyeFocusVal = 'Not Focused'
                    emotionVal = 'Negative'
                    headVal = 'Not Center'
                    # df = score(eyeFocusVal,emotionVal,headVal,next,df)
                    cv.rectangle(debug_image, (tab+1, 0), (int(tab2),30),
                            (0, 0,255), -1)
                    cv.putText(debug_image, "CANNOT DETECT HANDS", (tab+20,23),
                            cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

                        

            # Drawing part


            key = cv.waitKey(1)
            if key == 27:  # ESC
                break
            # Screen reflection
            key2 = cv.waitKey(1)
            if key2 == ord('s'):  # Save the frame to file
                print(item)
                count+=1
                cv.imwrite(f"Model Testing/{Testing}/{item}-{count}.jpg" , image)
                
            if show_live:
                cv.imshow('Facial Emotion and focus Recognition', debug_image)

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()