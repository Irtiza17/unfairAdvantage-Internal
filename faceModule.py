import mediapipe as mp
from globalFuncs import * 
from model import KeyPointClassifier


total_video_dur = 210
show_live = True

class faceDetector():
    def __init__(self,
               staticMode=False,
               maxFaces=1,
               refineLandmarks=True,
               minDetectionConfidence=0.5,
               minTrackingConfidence=0.5,
               headModel = True,
               focusModel = True,
               emotionModel = True,
               pointingModel = True,
               draw = False):
               

        self.focusModel = focusModel
        self.emotionModel = emotionModel
        self.headModel = headModel
        self.pointingModel = pointingModel
        self.draw = draw

        #Models Load
        self.mpFace = mp.solutions.face_mesh
        self.face = self.mpFace.FaceMesh(staticMode,maxFaces,refineLandmarks, minDetectionConfidence,minTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

        # Models and labels Path
        focusModelPath = 'model/keypoint_classifier/keypoint_classifier.tflite'
        emotionModelPath = 'model/keypoint_classifier/keypoint_classifier2.tflite'
        headModelPath = 'model/keypoint_classifier/keypoint_classifier3.tflite'
        focuslabelPath = 'focus_labels.csv'
        emotionlabelPath = 'emotion_labels.csv'
        headlabelPath = 'head_labels.csv'

        #Classifiers initialization
        self.focusClassifier = KeyPointClassifier(focusModelPath)
        self.emotionClassifier = KeyPointClassifier(emotionModelPath)
        self.headClassifier = KeyPointClassifier(headModelPath)

        #Labels List
        self.focus_labels = labelReader(focuslabelPath)
        self.emotion_labels = labelReader(emotionlabelPath)
        self.head_labels = labelReader(headlabelPath)
    
    def focusModelProcess(self):        
        focus_landmark_list = calc_landmark_list(self.debug_img, self.face_landmarks,"Focus")
        focus_processed_landmarks = pre_process_landmark(focus_landmark_list)
        focus_id = self.focusClassifier(focus_processed_landmarks)
        focusVal = self.focus_labels[focus_id]
        return focusVal

    def emotionModelProcess(self):        
        emotion_landmark_list = calc_landmark_list(self.debug_img, self.face_landmarks,"Emotion")
        emotion_processed_landmarks = pre_process_landmark(emotion_landmark_list)
        emotion_id = self.emotionClassifier(emotion_processed_landmarks)
        emotionVal = self.emotion_labels[emotion_id]
        return emotionVal

    def headModelProcess(self):        
        head_landmark_list = calc_landmark_list(self.debug_img, self.face_landmarks,"Head")
        processed_landmarks = pre_process_landmark(head_landmark_list)
        head_id = self.headClassifier(processed_landmarks)
        headVal = self.head_labels[head_id]
        return headVal

    def classPredictor(self,img,debug_image):
        self.debug_img = debug_image
        img.flags.writeable = False
        face_results = self.face.process(img)
        img.flags.writeable = True
        if face_results.multi_face_landmarks is not None:
            for self.face_landmarks in face_results.multi_face_landmarks:
                if self.draw:
                    self.mpDraw.draw_landmarks(self.debug_img, self.face_landmarks, self.mpFace.FACEMESH_TESSELATION)
                brect = calc_bounding_rect(self.debug_img,self.face_landmarks)
                if self.focusModel == True:
                    focusVal = self.focusModelProcess()
                else:
                    focusVal = 'N/A'
                if self.emotionModel == True:
                    emotionVal = self.emotionModelProcess()
                else:
                    emotionVal = 'N/A'
                if self.headModel == True:
                    headVal = self.headModelProcess()
                else:
                    headVal = 'N/A'

                self.debug_img = draw_bounding_rect(self.debug_img, brect)
        else:
            focusVal = 'N/A'
            emotionVal = 'N/A'
            headVal = 'N/A'

        return (self.debug_img,focusVal,emotionVal,headVal)
                

                


