from globalFuncs import *
from handModule import handDetector
from faceModule import faceDetector
from poseModule import poseDetector
from report import Scoring

def main():
    handmodel = handDetector()
    facemodel = faceDetector()
    posemodel = poseDetector()
    scoring = Scoring()
    
    cap,prev_frame_time = camSetup(0)

    videoProcess = videoDisplayFunc(False)


    while True:
        # Reading frame 
        ret, image = cap.read()
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        
        # FPS calculation
        FPS,prev_frame_time = fps(prev_frame_time)

        # Performing necessary image manipulation
        image,display_img = imgManip(image)

        # Face Readings from face model
        display_image,focusVal,emotionVal,headVal  = facemodel.classPredictor(image,display_img)

        # Hand Readings from hand model
        display_image,pointingVal,waveVal = handmodel.classPredictor(image,display_image)  

        # Arm Readings from pose model
        display_image,movementVal = posemodel.classPredictor(image,display_image)  

        # LabelDisplay
        display_image = draw_info_text(FPS,display_image,focusVal,emotionVal,headVal,pointingVal,waveVal,movementVal)

        #Score generation
        scoring.singleFrameDate(focusVal,emotionVal,headVal,pointingVal,waveVal,movementVal,prev_frame_time)
        
        #Displaying final image
        cv.imshow('Facial Emotion and focus Recognition', display_image)

    # Close playback video 
    videoProcess.terminate() if videoProcess != None else print('No video played')

    # Final Report generation
    scoring.reportGeneration()

if __name__ == "__main__":
    main()