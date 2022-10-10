import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 4), dtype='uint8')

blank[20:100, 200:300] = 255

cv.circle(blank, (250, 250), 10, (0, 0, 140), thickness=-1)

cv.putText(blank, "TEST TEST", (255, 255), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)

cv.imshow('Blank Canvas', blank)

cv.waitKey(0)