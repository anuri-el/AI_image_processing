import os
import cv2 as cv
from ultralytics import YOLO


def main():
    # img_path = "input/fruits/pexels-rachel-claire-5864750.jpg"
    # img_path = "input/fruits/pexels-viktoria-slowikowska-5677917.jpg"
    img_path = "input/fruits/a60ae20e20eb2a9d9ede90d2c471b936.jpg"

    # vid_path = "input/fruits/8203772-uhd_3840_2160_24fps.mp4"
    # vid_path = "input/fruits/15645011-hd_1920_1080_25fps.mp4"
    vid_path = "input/fruits/14249403_1920_1080_25fps.mp4"

    annotated_img, counts = detect_obj(img_path, conf=0.5, scale=1)
    cv.imshow("detected objects", annotated_img)
    # for label, count in counts.items():
    #     print(f"{label} : {count}")

    detect_obj_video(vid_path, conf=0.3, scale=0.3)
    track_obj(vid_path, conf=0.3, scale=0.3)

    cv.waitKey(0)
    cv.destroyAllWindows()


def resize_frame(img, scale=0.5):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    img = cv.resize(img, (width, height), cv.INTER_AREA)
    return img


def detect_obj(source, conf=0.6, scale=1):
    model = YOLO("models/yolov8m.pt")
    results = model(source, conf=conf)

    annotated_frame = results[0].plot()
    annotated_frame = resize_frame(annotated_frame, scale=scale)

    counts = {}
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        counts[label] = counts.get(label, 0) + 1

    if isinstance(source, str):
        basename = os.path.splitext(os.path.basename(source))[0]
        # output_path = f"output/{source[6:-4]}_out.jpg"
        output_path = f"output/{basename}_{conf}_out.jpg"
        cv.imwrite(output_path, annotated_frame)

    return annotated_frame, counts


def detect_obj_video(video_path, conf=0.6, scale=1):
    capture = cv.VideoCapture(video_path)

    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv.CAP_PROP_FPS)

    basename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"output/{basename}_{conf}_frame_out.mp4"
    fourcc = cv.VideoWriter_fourcc(*"mp4v")

    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        
        annotated_frame, counts = detect_obj(frame, conf)

        out.write(annotated_frame)
        cv.imshow("per frame", resize_frame(annotated_frame, scale))

        if cv.waitKey(1) & 0xFF==ord('q'):
            break

    capture.release()
    out.release()


def track_obj(video_path, conf=0.6, scale=1):
    model = YOLO("models/yolov8m.pt")

    capture = cv.VideoCapture(video_path)

    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv.CAP_PROP_FPS))

    basename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"output/{basename}_{conf}_track_out.mp4"
    fourcc = cv.VideoWriter_fourcc(*"mp4v")

    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        results = model.track(frame, conf=conf, persist=True)
        annotated_frame = results[0].plot()

        out.write(annotated_frame)
        cv.imshow("tracking", resize_frame(annotated_frame, scale))

        if cv.waitKey(1) & 0xFF==ord('q'):
            break

    out.release()
    capture.release()


if __name__ == "__main__":
    main()
