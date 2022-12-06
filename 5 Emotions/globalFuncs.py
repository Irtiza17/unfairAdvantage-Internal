import cv2 as cv
import copy
import subprocess
import sys
import itertools

def camSetup():
    cap_device = 0
    cap_width = 1920
    cap_height = 1080    
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    return cap

def imgFunc(img):
    img = cv.flip(img, 1)  # Mirror display
    image_width2, image_height2 = img.shape[1], img.shape[0]
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

def get_OS_platform():
    platforms = {
        'linux' : 'Linux',
        'darwin' : 'OS X',
        'win32' : 'Windows',
        'nt' : 'Windows',
        'win64' : 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform

    return platforms[sys.platform]

def videoDisplayFunc(video_display,video_file_name):
    if video_display:
        systemOS = get_OS_platform()
        if systemOS == "Windows":
            child_process = subprocess.Popen(["C:/Program Files (x86)/VideoLAN/VLC/vlc.exe","videos/" + video_file_name])
            print("Windows")
        elif systemOS == "OS X":
            child_process = subprocess.Popen(["/Applications/VLC.app/contents/MacOS/vlc", "videos/" + video_file_name])
            print("Mac OS")
        else: print("Other OS")
    
    return child_process

def draw_info_text(image, pointing_text=''):
    y_coordinate = 30
    y_spacing = 35
    font_size = 0.7
    cv.rectangle(image, (0, 0), (290, y_coordinate + (7 * y_spacing) + 5),
                    (0, 0, 0), -1)

    if pointing_text != "":
        info_text4 = 'Pointing: ' + pointing_text
        cv.putText(image, info_text4, (5, y_coordinate + (3 * y_spacing)),
                    cv.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1, cv.LINE_AA)
    else:
        info_text4 = 'Pointing: N/A'
        cv.putText(image, info_text4, (5, y_coordinate + (3 * y_spacing)),
                    cv.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1, cv.LINE_AA)

    
    return image

def calc_landmark_list(image, landmarks,ROI=False,hand = False):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []
    # Keypoint
    if ROI == False:
        if hand == False:
            for _, landmark in enumerate(landmarks.landmark):
                landmark_x = min(int(landmark.x * image_width), image_width - 1)
                landmark_y = min(int(landmark.y * image_height), image_height - 1)
                landmark_point.append([landmark_x, landmark_y])
        elif hand == True:
            for landmark in enumerate(landmarks.landmark):
                landmark_x = min(int(landmark.x * image_width), image_width - 1)
                landmark_y = min(int(landmark.y * image_height), image_height - 1)
                landmark_point.append([landmark_x, landmark_y])
        # for i in ROI:
            # landmark = landmarks.landmark[i]
    elif ROI != False:
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
