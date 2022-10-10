import numpy as np
import cv2 as cv
import os

rubiks_cube = cv.CascadeClassifier('cascade.xml')

img = cv.imread('test2.png')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces_rect = rubiks_cube.detectMultiScale(gray, 1.1, 5)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+h]
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0),thickness=2)

cv.imshow('Faces', img)
cv.waitKey(0)
 
# import cv2 as cv

# def GrabVideo():
#     return cv.VideoCapture(0)

# capture = GrabVideo()

# haar_cascade = cv.CascadeClassifier('rubiks_cube.xml')

# while True:
#     isTrue, frame = capture.read()
#     if (isTrue):
#         gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#         faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)

#         for (x,y,w,h) in faces_rect:
#             cv.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), thickness=2)

#         cv.imshow('Face Detection', frame)

#         if cv.waitKey(20) & 0xFF == ord('d'):
#             break
#     else:
#         capture = GrabVideo()

# cv.waitKey(0)