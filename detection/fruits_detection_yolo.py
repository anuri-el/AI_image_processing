import cv2 as cv
from ultralytics import YOLO


def main():
    # img_path = "input/fruits/pexels-rachel-claire-5864750.jpg"
    img_path = "input/fruits/pexels-viktoria-slowikowska-5677917.jpg"
    vid_path = "input/fruits/8203772-uhd_3840_2160_24fps.mp4"

    img = cv.imread(img_path)
    img = resize_frame(img, scale=0.3)
    cv.imshow("img", img)

    annotated_frame = detect_obj(img)

    cv.imshow("objects detection", annotated_frame)
    cv.imwrite("output/out.jpg", annotated_frame)

    cv.waitKey(0)
    cv.destroyAllWindows()


def resize_frame(img, scale=0.5):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    img = cv.resize(img, (width, height), cv.INTER_AREA)
    return img

# to-do: only prob >0.6
def detect_obj(img):
    model = YOLO("models/yolov8n.pt")
    results = model(img)

    annotated_frame = results[0].plot()

    return annotated_frame



if __name__ == "__main__":
    main()
