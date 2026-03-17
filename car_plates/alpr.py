import cv2 as cv
import numpy as np

from Detector import Detector


def main():
    frame = "input/cars/pexels-mikebirdy-132774.jpg"
    # frame = "input/cars/pexels-introspectivedsgn-4517064.jpg"
    # frame = "input/cars/pexels-pixabay-221270.jpg"
    # frame = "input/cars/pexels-introspectivedsgn-7725071.jpg"

    detector = Detector()
    boxes = detector.detect_vehicles(frame)
    print("vehicles:", boxes)

    img = cv.imread(frame)
    roi = img[boxes[0][1] : boxes[0][3], boxes[0][0] : boxes[0][2]]
    roi = resize_frame(roi, scale=0.3)
    cv.imshow("roi", roi)
    plate = Detector.find_plate_in_roi(img)
    print("plates:", plate)
    cv.waitKey(0)


def resize_frame(frame: np.ndarray, scale: float = 0.5):
    h = int(frame.shape[0] * scale)
    w = int(frame.shape[1] * scale)
    img = cv.resize(frame, (w, h), cv.INTER_AREA)
    return img

if __name__ == "__main__":
    main()
