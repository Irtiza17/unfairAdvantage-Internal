from globalFuncs import *
from handModule import handDetector
from faceModule import faceDetector
from poseModule import poseDetector
from report import report

# Models to run
focusModel = True
emotionModel = True
HeadModel = True
pointingModel = True
wavingModel = True
movementModel = True

# Cam source parameters
inputSource = "video" #inputSource Values can be "cam","video", depending on income stream source.
# sourcePath = 0 # If inputsource is cam, sourcePath can be 0 or 1, if its video , then sourcePath is a videopath.
sourcePath = 'D:/Internship/Tasks/Training/Moving hand while focus/handwhilefocus (10).mp4'
show_live = True

# Stimulant content video 
total_video_dur = 210
video_display = True
video_file_name = "demovideo.mp4"


def main():
    handmodel = handDetector(pointingModel,wavingModel)
    facemodel = faceDetector(focusModel,emotionModel,HeadModel)
    posemodel = poseDetector(movementModel)
    scoring = report()
    
    

    cap,startTime,frame_time= camSetup(inputSource,sourcePath)
    start = True
    videoProcess = videoDisplayFunc(video_file_name,video_display)
    
    while start:

        start = programTimingFunction(startTime,total_video_dur)

        # Reading frame 
        ret, image = cap.read()
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        # FPS calculation
        FPS,frame_time = fps(frame_time)

        # Performing necessary image manipulation
        image,display_img = imgManip(image,inputSource)

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
    videoProcess.terminate() if videoProcess != None else print('No video played')

    # Final Report generation
    scoring.reportGeneration()

if __name__ == "__main__":
    main()