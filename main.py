from globalFuncs import *
from handModule import handDetector
from faceModule import faceDetector
from poseModule import poseDetector
from report import report
from threading import Thread
import sys

# Models to run
focusModel = True
emotionModel = True
HeadModel = True
pointingModel = True
wavingModel = True
movementModel = True

# Cam source parameters
inputSource = "cam" #inputSource Values can be "cam","video", depending on income stream source.
sourcePath = 0 # If inputsource is cam, sourcePath can be 0 or 1, if its video , then sourcePath is a videopath.
# sourcePath = 'videos/video6.mp4'
show_live = True

# Stimulant content video 
total_video_dur = 15
# video_display = True
# video_file_name = "demovideo.mp4"


handmodel = handDetector(pointingModel,wavingModel)
facemodel = faceDetector(focusModel,emotionModel,HeadModel,draw=False)
posemodel = poseDetector(movementModel)
scoring = report()
cap,dimensions= camSetup(inputSource,sourcePath)


def main():
    startTime,frame_time = startTimeFunc()
    # videoProcess = Thread(target = videoDisplayFunc, args = (video_file_name,video_display))
    # videoProcess.daemon = True
    # videoProcess.start()

    # videoProcess = videoDisplayFunc(video_file_name,video_display)
    start = True
    while start:

        start = programTimingFunction(startTime,total_video_dur)

        # Reading frame 
        ret, image = cap.read()
        key = cv.waitKey(10)
        
        if key == 27:  # ESC
            cap.release()
            cv.destroyAllWindows()
            raise Exception("Assessment Interrupted")

        # FPS calculation
        FPS,frame_time = fps(frame_time)

        # Performing necessary image manipulation
        image,display_img = imgManip(image,inputSource,dimensions)

        # Face Readings from face model
        display_image,focusVal,emotionVal,headVal  = facemodel.classPredictor(image,display_img)

        # Hand Readings from hand model
        display_image,pointingVal,waveVal = handmodel.classPredictor(image,display_image)  

        # Arm Readings from pose model
        display_image,movementVal = posemodel.classPredictor(image,display_image)  

        # LabelDisplay
        display_image = draw_info_text(FPS,display_image,focusVal,emotionVal,headVal,pointingVal,waveVal,movementVal)

        #Score generation
        scoring.singleFrameData(focusVal,emotionVal,headVal,pointingVal,waveVal,movementVal,frame_time)
        
        #Displaying final image
        outputDisplay(show_live,display_image,'Facial Emotion and focus Recognition')

    # Close playback video 
    # videoProcess.terminate() if videoProcess != None else print('No video played')
    cap.release()
    cv.destroyAllWindows()

    # Final Report generation
    results = scoring.reportGeneration()

    return results

# if __name__ == "__main__":
#     main()