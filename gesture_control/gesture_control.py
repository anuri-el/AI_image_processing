import cv2 as cv
import mediapipe as mp
import urllib.request
import threading
from pathlib import Path


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path("hand_landmarker.task")


_lock = threading.Lock()
_result: HandLandmarkerResult | None = None


def main():
    ensure_model()

    options = HandLandmarkerOpts(base_options=BaseOptions(model_asset_path=str(MODEL_PATH)), running_mode=VisionRunningMode.LIVE_STREAM, num_hands=2, min_hand_detection_confidence=0.6,  min_hand_presence_confidence=0.6, min_tracking_confidence=0.5, result_callback=_on_result)

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)


    with HandLandmarker.create_from_options(options) as detector:
        ts_ms = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms += 1
            detector.detect_async(mp_img, ts_ms)

            with _lock:
                result = _result

            fist_detected = False

            if result and result.hand_landmarks:
                for lm_norm, handedness in zip(result.hand_landmarks, result.handedness):
                    label = handedness[0].category_name
                    print(label)
                    pts = [(int(p.x * w), int(p.y * h)) for p in lm_norm]

                    skel_color = (60, 220, 100) if label == "Right" else (220, 180, 60)
                    draw_landmark_connection(frame, pts, skel_color)


            cv.imshow("video", frame)
            if cv.waitKey(1) & 0xFF==ord("q"):
                break

    cap.release()
    cv.destroyAllWindows()


def ensure_model():
    if not MODEL_PATH.exists():
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def _on_result(result: HandLandmarkerResult, _img: mp.Image, _ts: int):
    global _result
    with _lock:
        _result = result


def draw_landmark_connection(frame, pts:list[tuple], color=(0, 220, 120), r=5):
    chains = [
        [0,1,2,3,4],
        [0,5,6,7,8],
        [9,10,11,12],
        [13,14,15,16],
        [0,17,18,19,20],
        [5,9,13,17],
    ]
    for chain in chains:
        for i in range(len(chain) - 1):
            cv.line(frame, pts[chain[i]], pts[chain[i+1]], color, 2)
    for p in pts:
        cv.circle(frame, p, r, color, -1)

if __name__ == "__main__":
    main()