import cv2 as cv
import numpy as np

img = cv.imread('C:/Users/tusha/Pictures/unknown.png')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))

cv.imshow('Edge Detection', lap)

cv.waitKey(0)