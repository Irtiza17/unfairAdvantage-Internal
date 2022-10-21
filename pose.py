import csv
import copy
import itertools
import depthai

import cv2 as cv
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


from model import KeyPointClassifier
model_path ='model/keypoint_classifier/keypoint_classifier.tflite'
model_path3 ='model/keypoint_classifier/keypoint_classifier3.tflite'


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.pose_landmarks.landmark):
    # print(i)
# for i in ROI:
    # landmark = landmarks.landmark[i]
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

    for _, landmark in enumerate(landmarks.pose_landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]
        # print("landmark Point is " , landmark_point)

        landmark_array = np.append(landmark_array, landmark_point, axis=0)
        # print("landmark array is " , landmark_array)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_info_text(image, brect, facial_text1):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    if facial_text1 != "":
        info_text = 'Pose :' + facial_text1
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    


    return image




# def oakD():
#     pipeline = depthai.Pipeline()
#     cam_rgb = pipeline.createColorCamera()
#     cam_rgb.setPreviewSize(300, 300) 
#     cam_rgb.setInterleaved(False)
#     xout_rgb = pipeline.createXLinkOut()
#     xout_rgb.setStreamName("rgb")
#     cam_rgb.preview.link(xout_rgb.input)
#     return pipeline

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    cap_device = 0
    cap_width = 1920
    cap_height = 1080

    use_brect = True

    # Camera preparation
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    # pipeline= oakD()

    # Model load
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) 

    # keypoint_classifier = KeyPointClassifier(model_path)
    # keypoint_classifier2 = KeyPointClassifier(model_path2)
    keypoint_classifier = KeyPointClassifier(model_path3)


    # Read labels
    with open('model/keypoint_classifier/keypoint_classifier_labelpose.csv',
                encoding='utf-8-sig') as f:
        keypoint_classifier_labelspose = csv.reader(f)
        keypoint_classifier_labelspose = [
            row[0] for row in keypoint_classifier_labelspose
        ]
    for idx, i in enumerate(keypoint_classifier_labelspose):
        print(idx,i)


    mode = 0

    # with depthai.Device(pipeline) as device:
        # q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        # Process Key (ESC: end)
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        # Camera capture
        # in_rgb = q_rgb.get()
        # image = in_rgb.getCvFrame()
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True

        if results is not None:
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, results)

                # Landmark calculation
                pose_landmark_list = calc_landmark_list(debug_image, results)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list1 = pre_process_landmark(
                    pose_landmark_list)

                #focus classification
                pose_id = keypoint_classifier(pre_processed_landmark_list1)
                if cv.waitKey(5) & 0xFF == ord('s'):
                    print(pose_id)
                
                
                
                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_info_text(
                        debug_image,
                        brect,
                        keypoint_classifier_labelspose[pose_id])

                # debug_image = draw_info_text1(
                #         debug_image,
                #         brect,
                #         keypoint_classifier_labels2[facial_emotion_id])

        # Screen reflection
        cv.imshow('Facial Emotion and focus Recognition', debug_image)

    # cap.release()
    cv.destroyAllWindows()