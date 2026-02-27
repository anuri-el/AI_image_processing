import cv2 as cv

# img = cv.imread('input/cats/4df2decc9e19ee56c0d6db606efdf943.jpg')
img = cv.imread('input/cats/b8e8dcac0164c6604c87906abb462a48.jpg')

def resizeFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

img = resizeFrame(img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cat_cascade = cv.CascadeClassifier('haarcascade/haarcascade_frontalcatface_extended.xml')
faces = cat_cascade.detectMultiScale(gray, 1.1, 3)

if len(faces) > 0:
    print(f'cat faces detected: {len(faces)}')

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (50, 0, 50), 2)
        cv.putText(img, 'it\'s a cat.', (x-10, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.65, (50, 0, 50), 2)
        cv.imshow('cascade', img)
else:
    print('no cat face detected')

cv.waitKey(0)
cv.destroyAllWindows()