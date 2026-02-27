import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade/haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier('haarcascade/haarcascade_smile.xml')

img = cv.imread('input/people/67f4956214dfe90e77953ba6b32b0786.jpg')
# img = cv.imread('input/people/694c2c4e7e991c8c2c49e2bda1e20445.jpg')

def resizeFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

img = resizeFrame(img, 0.75)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 7)

if len(faces) > 0:
    print(f'faces: {len(faces)}')
    for index, (x, y, w, h) in enumerate(faces):
        cv.rectangle(img, (x, y), (x+w, y+h), (50, 0, 25), 2)
        cv.putText(img, f'face_{index}', (x-10, y-10), cv.FONT_HERSHEY_PLAIN, 1, (50, 0, 25), 2)

        roi_color = img[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        # cv.imshow('roi', roi_gray)

        eyes = eye_cascade.detectMultiScale(roi_gray)
        smiles = smile_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (225, 25, 0), 2)
        
        for (sx, sy, sw, sh) in smiles:
            cv.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (25, 225, 0), 2)
        
        cv.imshow(f'faces', img)
else:
    print('no faces detected')


cv.waitKey(0)
cv.destroyAllWindows()