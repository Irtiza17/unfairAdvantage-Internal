import mediapipe as mp
from globalFuncs import * 
from model import KeyPointClassifier


class handDetector():
    def __init__(self,
               staticMode=False,
               maxHands=2,
               modelComplexity=1,
               minDetectionConfidence=0.5,
               minTrackingConfidence=0.4,
               draw = True,
               pointingModel = True,
               wavingModel = True):

        self.draw = draw
        self.pointingModel = pointingModel
        self.wavingModel = wavingModel
        self.start_waveFrame_left = (0,0)
        self.start_waveFrame_right = (0,0)

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(staticMode,maxHands,modelComplexity, minDetectionConfidence,minTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils


        pointingModelPath = 'model/keypoint_classifier/keypoint_classifier4.tflite'
        pointingLabelPath = 'pointing_labels.csv'
        self.pointingClassifier = KeyPointClassifier(pointingModelPath)
        self.pointing_labels = labelReader(pointingLabelPath)


    def pointingModelProcess(self):        
        hands_landmark_list = calc_landmark_list(self.debug_img, self.hand_landmarks,"Hands")
        hands_processed_landmarks = pre_process_landmark(hands_landmark_list)
        pointing_id = self.pointingClassifier(hands_processed_landmarks)
        pointingVal = self.pointing_labels[pointing_id]
        return pointingVal

    def wavingModelProcess(self):        
        if self.handNo == 0:
            next_frame_left = coordinatesCalc(self.img,self.hand_landmarks,8)
            difference = length(self.start_waveFrame_left,next_frame_left)
            
            if difference > 30 and hi5Detection(self.hand_landmarks, self.img):
                waveVal_0 = 'waving'
            else:
                waveVal_0 = 'not waving'

            self.start_waveFrame_left = next_frame_left
        else:
            waveVal_0 = 'not waving'
            
        if self.handNo == 1:
            next_frame_right = coordinatesCalc(self.img,self.hand_landmarks,8)
            difference = length(self.start_waveFrame_right,next_frame_right)
            if difference > 30 and hi5Detection(self.hand_landmarks, self.img):
                waveVal_1 = 'waving'
            else:
                waveVal_1 = 'not waving'
            self.start_waveFrame_right = next_frame_right
        else:
            waveVal_1 = 'not waving'

        if waveVal_0 == 'waving' or waveVal_1 == 'waving':
            waveVal = 'waving'
        else:
            waveVal = 'not waving'
        
        return waveVal

    def classPredictor(self,img,debug_img):
        self.img = img
        self.debug_img = debug_img
        img.flags.writeable = False
        hand_results = self.hands.process(img)
        img.flags.writeable = True
        if hand_results.multi_hand_landmarks is not None:
            for self.handNo, self.hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                if self.draw:
                    self.mpDraw.draw_landmarks(debug_img, self.hand_landmarks, self.mpHands.HAND_CONNECTIONS)
                if self.pointingModel == True:
                    pointingVal = self.pointingModelProcess()
                else:
                    pointingVal = 'N/A'
                if self.wavingModel == True:
                    waveVal= self.wavingModelProcess()
                else:
                    waveVal = 'N/A'
        else:
            pointingVal = 'N/A'
            waveVal = 'N/A'

        return self.debug_img,pointingVal,waveVal


