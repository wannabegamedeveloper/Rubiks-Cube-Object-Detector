import cv2 as cv

def GrabVideo():
    return cv.VideoCapture(0)

capture = GrabVideo()

haar_cascade = cv.CascadeClassifier('haar_face.xml')

while True:
    isTrue, frame = capture.read()
    if (isTrue):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)

        for (x,y,w,h) in faces_rect:
            cv.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), thickness=2)

        cv.imshow('Face Detection', frame)

        if cv.waitKey(20) & 0xFF == ord('d'):
            break
    else:
        capture = GrabVideo()

cv.waitKey(0)