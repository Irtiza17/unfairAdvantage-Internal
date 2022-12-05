import cv2 
from deepface import DeepFace

camera = cv2.VideoCapture(0)

while camera.isOpened():

    ok, image = camera.read()
    # image = cv2.imread('ammar1.jpg')
    if not ok:
        print("Ignoring empty camera frame.")
        continue

    try:
        result = DeepFace.stream(image, model_name='VGG-Face',detector_backend='opencv',enable_face_analysis=False,time_threshold=1,frame_threshold=1)
    except:
        print('try')

    cv2.imshow('Result',image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
camera.release()