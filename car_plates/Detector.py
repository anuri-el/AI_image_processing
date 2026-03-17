import cv2 as cv
import numpy as np
from ultralytics import YOLO


class Detector:
    CAR_CLASSES = {2, 3, 5, 7}


    def __init__(self, model_path: str = "models/yolov8n.pt"):
        self.model = YOLO(model_path)


    def detect_vehicles(self, frame: np.ndarray):
        results = self.model(frame, conf=0.4, verbose=False)

        boxes = []
        for r in results:
            for box in r.boxes:
                if int(box.cls) in self.CAR_CLASSES:
                    boxes.append(box.xyxy[0].numpy().astype(int))
        return boxes
    
    
    @staticmethod
    def find_plate_in_roi(roi: np.ndarray):
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        blur = cv.bilateralFilter(gray, 11, 17, 17)
        edges = cv.Canny(blur, 30, 210)
        contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        h, w = roi.shape[:2]

        for cnt in sorted(contours, key=cv.contourArea, reverse=True)[:20]:
            x, y, cw, ch = cv.boundingRect(cv.approxPolyDP(cnt, 0.02*cv.arcLength(cnt, True), True))
            aspect = cw / (ch + 1e-6)
            ratio = (cw * ch) / (w * h)
            if 1.5 < aspect < 6.0 and 0.02 < ratio < 0.25:
                return np.array([x, y, x + cw, y + ch])
        return None