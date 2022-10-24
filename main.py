import csv
import copy
import itertools
import depthai

import cv2 as cv
import numpy as np
import mediapipe as mp
from model import KeyPointClassifier
from scoring import score,assignscore


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

poseModel = False
focusModel = True
emotionModel = True

camera = "cam"

model_path ='model/keypoint_classifier/keypoint_classifier.tflite'
model_path2 ='model/keypoint_classifier/keypoint_classifier2.tflite'
model_path3 ='model/keypoint_classifier/keypoint_classifier3.tflite'

ROI = [246, 161, 160, 159, 158, 157, 173, 33, 7, 163, 144, 145, 153, 154, 155, 133, 473, 474, 475, 476, 
        477, 466, 388, 387, 386, 385, 384, 398, 263, 249, 390, 373, 374, 380, 381, 382, 362, 468,469, 470, 
        471, 472]


def calc_landmark_list(image, landmarks,ROI=False,Pose=False):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []
    # Keypoint
    if ROI == False and Pose == False:
        for _, landmark in enumerate(landmarks.landmark):
        # for i in ROI:
            # landmark = landmarks.landmark[i]
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])

    elif ROI != False:
        for i in ROI:
            landmark = landmarks.landmark[i]
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])

    elif Pose != False:
        for _, landmark in enumerate(landmarks.pose_landmarks.landmark):
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
                     (0, 0, 0), 1)

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


def draw_info_text(image, facial_text1,facial_text2='',facial_text3=''):
    cv.rectangle(image, (0, 0), (250,160),
                 (0, 0, 0), -1)

    if facial_text1 != "":
        info_text = 'Gaze: ' + facial_text1
        cv.putText(image, info_text, (5,30),
                cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv.LINE_AA)
    
    if facial_text2 != "":
        info_text2 = 'Emotion: ' + facial_text2
        cv.putText(image, info_text2, (5,70),
                cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv.LINE_AA)
                
    if facial_text3 != "":
        info_text3 = 'Pose: ' + facial_text3
        cv.putText(image, info_text3, (5,110),
                cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv.LINE_AA)
    val = []
    if facial_text1 == 'Focused':
        val.append(1)
    if facial_text2 == 'Positive':
        val.append(1)
    if facial_text3 == 'Sitting':
        val.append(1)
    if poseModel == False:
        pars = 2
    else:
        pars = 3

    cv.putText(image, f"Score: {sum(val)}/{pars}", (5,150),
                cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv.LINE_AA)

    # cv.putText(image, 'SCORING: [Focus(5) Emotion(4)]', (brect[0] + 5, brect[1] + 40),
    #            cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv.LINE_AA)


    return image


def oakD():
    pipeline = depthai.Pipeline()
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(300, 300) 
    cam_rgb.setInterleaved(False)
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)
    return pipeline



with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:


    
    
    mode = 0

    # Model load
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) 

    keypoint_classifier = KeyPointClassifier(model_path)
    keypoint_classifier2 = KeyPointClassifier(model_path2)
    keypoint_classifier3 = KeyPointClassifier(model_path3)

    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    for idx, i in enumerate(keypoint_classifier_labels):
        print(idx,i)
    # Read labels
    with open('model/keypoint_classifier/keypoint_classifier_label2.csv',
                encoding='utf-8-sig') as f:
        keypoint_classifier_labels2 = csv.reader(f)
        keypoint_classifier_labels2 = [
            row[0] for row in keypoint_classifier_labels2
        ]

    for idx2, i2 in enumerate(keypoint_classifier_labels2):
        print(idx2,i2)

    with open('model/keypoint_classifier/keypoint_classifier_label3.csv',
                encoding='utf-8-sig') as f:
        keypoint_classifier_labels3 = csv.reader(f)
        keypoint_classifier_labels3 = [
            row[0] for row in keypoint_classifier_labels3
        ]

    for idx3, i3 in enumerate(keypoint_classifier_labels3):
        print(idx3,i3)
    
    use_brect = True    
    
    if camera == 'cam':
        cap_device = 1
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

    while True:
        # Process Key (ESC: end)
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        # Camera capture
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        if focusModel == True or emotionModel == True:

            image.flags.writeable = False
            results = face_mesh.process(image)
            image.flags.writeable = True

            if results.multi_face_landmarks is not None:
                for face_landmarks in results.multi_face_landmarks:
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, face_landmarks)

                    # Landmark calculation
                    # landmark_list1 = calc_landmark_list(debug_image, face_landmarks,ROI)
                    if focusModel == True:
                        focus_landmark_list = calc_landmark_list(debug_image, face_landmarks,ROI)
                        # Conversion to relative coordinates / normalized coordinates
                        pre_processed_landmark_list1 = pre_process_landmark(
                        focus_landmark_list)
                        facial_focus_id = keypoint_classifier(pre_processed_landmark_list1)



                    if emotionModel == True:
                        emotion_landmark_list = calc_landmark_list(debug_image, face_landmarks)
                        pre_processed_landmark_list2 = pre_process_landmark(
                        emotion_landmark_list)
                        facial_emotion_id = keypoint_classifier2(pre_processed_landmark_list2)
                        if cv.waitKey(5) & 0xFF == ord('t'):
                            print(facial_emotion_id)
                
                    
                    
                    # Drawing part
                    # debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    # debug_image = draw_info_text(
                    #         debug_image,
                    #         brect,
                    #         keypoint_classifier_labels[facial_focus_id],keypoint_classifier_labels2[facial_emotion_id])

            

        if poseModel == True:
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True

            if results is not None:
                # Bounding box calculation
                # brect = calc_bounding_rect(debug_image, results)

                # Landmark calculation
                pose_landmark_list = calc_landmark_list(debug_image, results,Pose=True)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list3 = pre_process_landmark(
                    pose_landmark_list)

                #focus classification
                pose_id = keypoint_classifier3(pre_processed_landmark_list3)
                if cv.waitKey(5) & 0xFF == ord('s'):
                    print(pose_id)
                    
        # Drawing part
        debug_image = draw_bounding_rect(use_brect, debug_image, brect)
        if poseModel == True:
            debug_image = draw_info_text(
                                debug_image,
                                keypoint_classifier_labels[facial_focus_id],keypoint_classifier_labels2[facial_emotion_id],keypoint_classifier_labels3[pose_id]
                                )
        else:
             debug_image = draw_info_text(
                                debug_image,
                                keypoint_classifier_labels[facial_focus_id],keypoint_classifier_labels2[facial_emotion_id],
                                )

        # Scoring part
        eyeFocusVal = keypoint_classifier_labels[facial_focus_id]
        emotionFocusVal = keypoint_classifier_labels2[facial_emotion_id]
        assignscore(eyeFocusVal,emotionFocusVal)

        # Screen reflection
        cv.imshow('Facial Emotion and focus Recognition', debug_image)

    # cap.release()
    cv.destroyAllWindows()