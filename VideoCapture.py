import cv2 as cv

def Grayscale(frame):
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

def GaussianBlur(frame):
    return cv.GaussianBlur(frame, (9, 9), cv.BORDER_DEFAULT)

def CannyEdges(frame):
    return cv.Canny(frame, 125, 180)



def GrabVideo():
    return cv.VideoCapture('C:/Users/tusha/Videos/3d character.mp4')



capture = GrabVideo()

while True:
    isTrue, frame = capture.read()
    if (isTrue):
        frame = CannyEdges(frame)
        cv.imshow('Video', frame)

        if cv.waitKey(1) & 0xFF == ord('d'):
            break
    else:
        capture = GrabVideo()

capture.release()
cv.destroyAllWindows()

