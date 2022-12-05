import cv2 as cv
from deepface import DeepFace

one = cv.imread('G:/My Drive/JP/JP/Tasks/unfairAdvantage/Deepface/personal/girl1.jpg')
one = cv.imread('G:/My Drive/JP/JP/Tasks/unfairAdvantage/different approaches/Deepface/personal/images/smile.jpg')
# two = cv.imread('Img2.jpeg')

result = DeepFace.analyze(one, actions=['emotion'])

# result = DeepFace.stream('images')


# result = DeepFace.verify(img1_path = 'G:/My Drive/JP/JP/Tasks/unfairAdvantage/Deepface/personal/Img1.jpeg', img2_path = 'G:/My Drive/JP/JP/Tasks/unfairAdvantage/Deepface/personal/man1.jpg' )
print(result)
print('='*50)
print(result['dominant_emotion'])