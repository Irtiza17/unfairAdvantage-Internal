import time
import cv2 
from deepface import DeepFace

camera = cv2.VideoCapture(0)
camera.set(3,1280)
camera.set(4,960)

while camera.isOpened():

    ok, image = camera.read()
    # image = cv2.imread('ammar1.jpg')
    if not ok:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    try:
        result = DeepFace.analyze(image, actions=['emotion'],detector_backend = 'ssd')
        # time.sleep(5)
        print('='*50)
        print(result['dominant_emotion'])
        print('='*50)
    except:
        print('try again')


    cv2.imshow('Result',image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
camera.release()