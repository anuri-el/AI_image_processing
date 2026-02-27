import os
import cv2 as cv
from ultralytics import YOLO


def main():
    # img_path = "input/fruits/pexels-rachel-claire-5864750.jpg"
    img_path = "input/fruits/pexels-viktoria-slowikowska-5677917.jpg"
    vid_path = "input/fruits/8203772-uhd_3840_2160_24fps.mp4"

    annotated_img = detect_obj(img_path)
    annotated_img = resize_frame(annotated_img, scale=0.3)
    cv.imshow("detected objects", annotated_img)

    detect_obj_video(vid_path)

    cv.waitKey(0)
    cv.destroyAllWindows()


def resize_frame(img, scale=0.5):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    img = cv.resize(img, (width, height), cv.INTER_AREA)
    return img


# to-do: only prob >0.6
def detect_obj(source):
    model = YOLO("models/yolov8m.pt")
    results = model(source)

    annotated_frame = results[0].plot()

    if isinstance(source, str):
        basename = os.path.splitext(os.path.basename(source))[0]
        # output_path = f"output/{source[6:-4]}_out.jpg"
        output_path = f"output/{basename}_out.jpg"
        cv.imwrite(output_path, annotated_frame)

    return annotated_frame


def detect_obj_video(vid_path):
    capture = cv.VideoCapture(vid_path)

    while capture.isOpened():
        ret, frame = capture.read()
        if ret == False:
            print("to err is human")
        
        frame = resize_frame(frame)
        annotated_frame = detect_obj(frame)
        cv.imshow("vid", annotated_frame)
    
        if cv.waitKey(1) & 0xFF==ord('q'):
            break

    capture.release()

if __name__ == "__main__":
    main()
