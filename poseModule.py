import mediapipe as mp
from globalFuncs import * 

class poseDetector():
    def __init__(self,
                movementModel = True,
               staticMode=False,
               modelComplexity=1,
               smoothLandmarks = True,
               enableSegmentation=False,
               smoothSegmentation=True,
               minDetectionConfidence=0.7,
               minTrackingConfidence=0.8,
               draw = True):

        self.label0 = 'moving'
        self.label1 = 'not moving'



        self.draw = draw
        self.movementModel = movementModel

        #Pose Model Load
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(staticMode,modelComplexity,smoothLandmarks,enableSegmentation,smoothSegmentation,minDetectionConfidence,minTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils    

        # Initial frame values
        self.start_right_wrist = (0,0)
        self.start_right_elbow = (0,0)
        self.start_right_shoulder = (0,0)
        self.start_left_wrist = (0,0)
        self.start_left_elbow = (0,0)
        self.start_left_shoulder = (0,0)

    def movementModelProcess(self):    
        image_width, image_height = self.img.shape[1], self.img.shape[0]

        next_right_wrist = coordinatesCalc(self.img,self.pose_results.pose_landmarks,15)
        next_right_elbow = coordinatesCalc(self.img,self.pose_results.pose_landmarks,13)
        next_right_shoulder = coordinatesCalc(self.img,self.pose_results.pose_landmarks,11)
        next_left_wrist = coordinatesCalc(self.img,self.pose_results.pose_landmarks,16)
        next_left_elbow = coordinatesCalc(self.img,self.pose_results.pose_landmarks,14)
        next_left_shoulder = coordinatesCalc(self.img,self.pose_results.pose_landmarks,12)

        # Calculating difference of initial and next frame
        diff_right_wrist = length(self.start_right_wrist, next_right_wrist)
        diff_left_wrist = length(self.start_left_wrist, next_left_wrist)
        diff_right_elbow = length(self.start_right_elbow,next_right_elbow)
        diff_left_elbow = length(self.start_left_elbow, next_left_elbow)

        # Checking condition of movement
        if diff_right_wrist > 17 and next_right_wrist[1]/image_height< 0.98:
            moveVal_0 = self.label0
        else:
            moveVal_0 = self.label1
        self.start_right_wrist = next_right_wrist

        if diff_left_wrist > 17 and next_left_wrist[1]/image_height< 0.98:
            moveVal_1 = self.label0
        else:
            moveVal_1 = self.label1
        self.start_left_wrist = next_left_wrist
    
        if moveVal_0 == self.label0 or moveVal_1 == self.label0:
            movementVal = self.label0
        else:
            movementVal = self.label1
    
        return movementVal

    def classPredictor(self,img,debug_img):
        self.img = img
        self.debug_img = debug_img
        img.flags.writeable = False
        self.pose_results = self.pose.process(img)
        img.flags.writeable = True
        if self.pose_results.pose_landmarks is not None:
            # if self.draw:
            #     self.mpDraw.draw_landmarks(debug_img, self.pose_landmarks, self.mpHands.HAND_CONNECTIONS)
            if self.movementModel == True:
                movementVal = self.movementModelProcess()
            else:
                movementVal = 'N/A'
        else:
            movementVal = 'N/A'

        return self.debug_img,movementVal


