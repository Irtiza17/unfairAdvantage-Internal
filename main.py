from statistics import mean
import cv2
import mediapipe as mp

from faceEyesTracking import faceDetect
from faceEyesTracking import eyeDetection
from focusTracking import headTurned
from focusTracking import eyeIrisMismatch
from focusTracking import eyesOpenDetection



cam = cv2.VideoCapture(0) # Open webcam live feed
cam.set(3,1080)
cam.set(4,960)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks = True,
max_num_faces=2,
min_detection_confidence=0.5,
min_tracking_confidence=0.5) # Creating facemesh

while True:
    ret,frame = cam.read()
    frame = cv2.flip(frame,1)        
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)

    #Extract Landmark Features
    landmark_points = output.multi_face_landmarks
    if landmark_points:
        landmarks = landmark_points[0].landmark

        faceDetect(frame,landmarks)
        eyes_center = eyeDetection(frame,landmarks)
        eyesOpenDetection(frame,landmarks)
        headTurned(frame,eyes_center)
        eyeIrisMismatch(frame,landmark_points)



    cv2.imshow("",frame)
    if cv2.waitKey(30) == ord('x'):
            break
cam.release()
cv2.destroyAllWindows()

