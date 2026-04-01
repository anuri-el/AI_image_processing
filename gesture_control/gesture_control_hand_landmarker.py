import cv2 as cv
import mediapipe as mp
import urllib.request
import threading
import math
import time
from pathlib import Path
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


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

    current_vol = get_volume()
    mic_muted = get_mic_mute()
    fist_prev = False
    fist_cooldown = 0.0



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
            r_hint = "-"
            l_hint = "-"

            if result and result.hand_landmarks:
                for lm_norm, handedness in zip(result.hand_landmarks, result.handedness):
                    label = handedness[0].category_name
                    pts = [(int(p.x * w), int(p.y * h)) for p in lm_norm]

                    skel_color = (60, 220, 100) if label == "Right" else (220, 180, 60)
                    draw_landmark_connection(frame, pts, skel_color)

                    if label == "Right":
                        thumb = pts[4]
                        index = pts[8]
                        dist = distance(thumb, index)

                        vol = (dist - 20) / 200.0
                        vol = max(0, min(1, vol))
                        set_volume(vol)
                        alpha = 0.2
                        current_vol = current_vol*(1-alpha) + vol*alpha

                        line_color = (0, int(220*(1-vol)+60), int(220*vol))

                        cv.line(frame, thumb, index, line_color, 3)
                        cv.circle(frame, thumb, 10, line_color, -1)
                        cv.circle(frame, index, 10, line_color, -1)

                        mid = ((thumb[0]+index[0])//2, (thumb[1]+index[1])//2)
                        cv.putText(frame, f"{int(vol*100)}%", (mid[0]+12, mid[1]-8), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        r_hint = f"Volume {int(vol*100)}%"

                    elif label == "Left":
                        if is_fist(pts, label):
                            fist_detected = True
                            l_hint = "fist - on/off microphone"
                            cv.circle(frame, pts[0], 24, (0, 80, 220), 3)
                        else:
                            n = sum(fingers_up(pts, label))
                            l_hint = f"fingers up: {n}"

            now = time.time()
            if fist_detected and not fist_prev and (now - fist_cooldown) > 0.8:
                mic_muted = not mic_muted
                set_mic_mute(mic_muted)
                fist_cooldown = now
            fist_prev = fist_detected

            draw_volume_bar(frame, current_vol)
            draw_mic_badge(frame, mic_muted)
            overlay_hint(frame, h, r_hint, l_hint)

            cv.rectangle(frame, (0, 0), (w, 32), (12, 12, 12), -1)
            cv.putText(frame, "Gesture Audio Control", (10, 22), cv.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
            cv.imshow("Gesture Audio Control", frame)
            if cv.waitKey(1) & 0xFF==ord("q"):
                break

    cap.release()
    cv.destroyAllWindows()


def _activate(device):
    iface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(iface, POINTER(IAudioEndpointVolume))


_spk = _activate(AudioUtilities.GetSpeakers())
_mic = None
try:
    _mic = _activate(AudioUtilities.GetMicrophone())
except Exception:
    pass

def get_volume():
    return _spk.GetMasterVolumeLevelScalar()


def set_volume(v: float):
    _spk.SetMasterVolumeLevelScalar(max(0, min(1, v)), None)


def get_mic_mute():
    if _mic:
        return bool(_mic.GetMute())
    return False


def set_mic_mute(muted: bool):
    if _mic:
        _mic.SetMute(int(muted), None)


def ensure_model():
    if not MODEL_PATH.exists():
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def _on_result(result: HandLandmarkerResult, _img: mp.Image, _ts: int):
    global _result
    with _lock:
        _result = result


def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def fingers_up(lm_list, hand_label):
    TIPS = [4, 8, 12, 16, 20]
    PIPS = [3, 6, 10, 14, 18]

    up = []
    if hand_label == "Right":
        up.append(lm_list[4][0] < lm_list[3][0])
    else:
        up.append(lm_list[4][0] > lm_list[3][0])
    for tip, pip in zip(TIPS[1:], PIPS[1:]):
        up.append(lm_list[tip][1] < lm_list[pip][1])
    return up


def is_fist(lm_list, hand_label):
    return not any(fingers_up(lm_list, hand_label))


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


def draw_volume_bar(frame, vol: float, x=30, y=100, w=28, h=230):
    cv.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), 1)
    filled = int(h * vol)
    color = (0, int(200 * (1 - vol) + 60), int(200 * vol))
    cv.rectangle(frame, (x, y + h - filled), (x + w, y + h), color, -1)
    cv.rectangle(frame, (x, y), (x + w, y + h), (220, 220, 220), 1)
    cv.putText(frame, "VOL", (x , y - 8), cv.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv.putText(frame, f"{int(vol*100)}%", (x - 2 , y + h + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 220), 1)


def draw_mic_badge(frame, muted: bool, x=30, y=360):
    color = (40, 40, 220) if muted else (40, 200, 60)
    label = "MIC OFF" if muted else "MIC ON"
    icon = "x" if muted else "o"
    cv.rectangle(frame, (x - 4, y - 24), (x + 118, y + 6), (25, 25, 25), -1)
    cv.rectangle(frame, (x - 4, y - 24), (x + 118, y + 6), color, 1)
    cv.putText(frame, icon, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv.putText(frame, label, (x + 18, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def overlay_hint(frame, h: int, right_txt: str, left_txt: str):
    bar_y = h - 52
    cv.rectangle(frame, (0, bar_y), (frame.shape[1], h), (18, 18, 18), -1)
    cv.putText(frame, f"[R] {right_txt}", (10, bar_y + 20),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 80), 1)
    cv.putText(frame, f"[L] {left_txt}",  (10, bar_y + 40),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 220), 1)
    

if __name__ == "__main__":
    main()