import csv
import copy
import itertools
import os

import cv2 as cv
import numpy as np
import mediapipe as mp
from model import KeyPointClassifier

ROI = [246, 161, 160, 159, 158, 157, 173, 33, 7, 163, 144, 145, 153, 154, 155, 133, 473, 474, 475, 476, 477, 466, 388, 387, 386, 385, 384, 398, 263, 249, 390, 373, 374, 380, 381, 382, 362, 468,
469, 470, 471, 472]

def calc_landmark_list(image, landmarks, ROI):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    # for _, landmark in enumerate(landmarks.landmark):
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
                     (0, 0, 0), 1)

    return image

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def draw_info_text(image, brect, facial_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    if facial_text != "":
        info_text = 'Emotion :' + facial_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) 

keypoint_classifier = KeyPointClassifier()

with open('model/keypoint_classifier/keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]

for idx, i in enumerate(keypoint_classifier_labels):
    print(idx,i)

mode = 0
use_brect = True

root = "G:/My Drive/JP/JP/Tasks/unfairAdvantage/modelTrainingApproach/testLab/test"
IMAGE_FILES = []
result = []
for path, subdirs, files in os.walk(root):
    for name in files:
        IMAGE_FILES.append(os.path.join(path, name))

count = 0
noLandmark = 0
for idx, file in enumerate(IMAGE_FILES):

    image = cv.imread(file)
    height,width,_ = image.shape

    # Detection implementation
    # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            for i in ROI:
                pt = face_landmarks.landmark[i]
                x = int(pt.x * width)
                y = int(pt.y * height)
                cv.circle(image, (x,y),2,(0,255,0),-1)
    cv.putText(image, file, (0,50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    cv.imshow('Facial Emotion Recognition', image)
    cv.waitKey(0)