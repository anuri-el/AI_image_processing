import cv2 as cv


def main():
    img_face = cv.imread('input/people/67f4956214dfe90e77953ba6b32b0786.jpg')
    cv.imshow('original_face', img_face)

    img_body = cv.imread("input/fullbody/de32ac2bff03784a0693cd7073341053.jpg")
    cv.imshow('original_fullbody', img_body)


    img_face = resizeFrame(img_face, 0.7)
    img_face_d = detect_faces(img_face)
    cv.imshow('faces', img_face_d)

    img_body = resizeFrame(img_body, 0.8)
    img_body_d = detect_fullbody(img_body)
    cv.imshow('fullbody', img_body_d)

    cv.waitKey(0)
    
    capture = cv.VideoCapture("input/fullbody/13239139-hd_1280_720_50fps.mp4")

    while capture.isOpened():
        ret, frame = capture.read()
        if ret is False:
            break

        frame = resizeFrame(frame, 0.7)
        frame_fb = detect_fullbody(frame)        
        cv.imshow("video", frame_fb)

        if cv.waitKey(1) & 0xFF==ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()


def resizeFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)


def detect_faces(img):
    face_cascade = cv.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('haarcascade/haarcascade_eye.xml')
    smile_cascade = cv.CascadeClassifier('haarcascade/haarcascade_smile.xml')

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.04, 5)

    if len(faces) > 0:
        print(f'faces: {len(faces)}')

        for index, (x, y, w, h) in enumerate(faces):
            cv.rectangle(img, (x, y), (x+w, y+h), (50, 0, 25), 2)
            cv.putText(img, f'face_{index + 1}', (x-10, y-10), cv.FONT_HERSHEY_PLAIN, 1, (225, 71, 117), 2)

            roi_color = img[y:y+h, x:x+w]
            roi_gray = gray[y:y+h, x:x+w]
            # cv.imshow('roi', roi_gray)

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.1, 5)

            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (225, 25, 0), 2)
            
            for (sx, sy, sw, sh) in smiles:
                cv.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (25, 225, 0), 2)
            
        return img
    else:
        cv.putText(img, 'no face detected', (20, 20), cv.FONT_HERSHEY_PLAIN, 1, (225, 71, 117), 2)
        return img


def detect_fullbody(img):
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    people, weight = hog.detectMultiScale(gray, winStride=(8, 8), padding=(32, 32), scale=1.05)

    if len(people) > 0:
        for (x, y, w, h) in people:
            cv.rectangle(img, (x,y), (x+w, y+h), (225, 71, 117), 2)
        return img
    else:
        cv.putText(img, "no people", (20,20), cv.FONT_HERSHEY_SIMPLEX, 1, (210, 210, 120), 2)
        return img


if __name__ == "__main__":
    main()