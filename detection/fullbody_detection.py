import cv2 as cv
import numpy as np

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

def resizeFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)


# --- detection in images ---
img = cv.imread('input/fullbody/de32ac2bff03784a0693cd7073341053.jpg')

img = resizeFrame(img, 0.8)
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

people = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)

if len(people[0]) > 0:
    for (x, y, w, h) in people[0]:
        cv.rectangle(img, (x, y), (x+w, y+h), (150, 125, 50), 2)
else:
    print('no people')

cv.imshow('people', img)

cv.waitKey(0)


# --- detection in videos ---
cv.startWindowThread()

capture = cv.VideoCapture('input/fullbody/5631765-uhd_3840_2160_24fps.mp4')
# capture = cv.VideoCapture('input/fullbody/13239139-hd_1920_1080_50fps.mp4')

while True:
    ret, frame = capture.read() 
    frame = resizeFrame(frame, scale=0.5)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8))
    boxes = np.array([[x, y, x+w, y+h] for (x, y, w, h) in boxes])

    for (xa, ya, xb, yb) in boxes:
        cv.rectangle(frame, (xa, ya), (xb, yb), (123, 23, 2), 2)
    
    cv.imshow('video', frame)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break

capture.release()

cv.destroyAllWindows()