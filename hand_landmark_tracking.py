import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities

# Initialize audio volume control
devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume

# Track last-set values to avoid spamming updates every frame
last_volume = None
last_brightness = None
smoothed_volume = None
smoothed_brightness = None
SMOOTHING = 0.3  # lower = smoother/slower to respond, higher = snappier/more jittery

def smooth(new_value, prev_smoothed):
    if prev_smoothed is None:
        return new_value
    return prev_smoothed * (1 - SMOOTHING) + new_value * SMOOTHING

def set_volume(percent):
    global last_volume
    if last_volume is None or abs(percent - last_volume) > 1:
        volume.SetMasterVolumeLevelScalar(percent / 100, None)
        last_volume = percent

def set_brightness(percent):
    global last_brightness
    if last_brightness is None or abs(percent - last_brightness) > 1:
        sbc.set_brightness(int(percent))
        last_brightness = percent

def pinch_distance(hand_landmarks):
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    return np.sqrt(
        (thumb_tip.x - index_tip.x)**2 +
        (thumb_tip.y - index_tip.y)**2
    )

def map_to_percent(distance, min_dist=0.02, max_dist=0.20):
    """Maps raw pinch distance to 0-100. Tune min_dist/max_dist by testing your own pinch range."""
    clamped = np.clip(distance, min_dist, max_dist)
    percent = (clamped - min_dist) / (max_dist - min_dist) * 100
    return percent

base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,                          # changed from 1
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
landmarker = vision.HandLandmarker.create_from_options(options)

connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

cap = cv2.VideoCapture(0)
frame_timestamp_ms = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    frame_timestamp_ms += 33
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    h, w, _ = frame.shape

    if result.hand_landmarks:
        # handedness tells us Left vs Right for each detected hand
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name  # "Left" or "Right"
            label = "Right" if label == "Left" else "Left"   # correct for the mirror flip

            points = []
            for lm in hand_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                points.append((x, y))
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            for start, end in connections:
                cv2.line(frame, points[start], points[end], (255, 255, 255), 2)

            # Label which hand this is, near the wrist point
            cv2.putText(frame, label, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            d = pinch_distance(hand_landmarks)
            raw_percent = map_to_percent(d)

            action = "Volume" if label == "Left" else "Brightness"
            if label == "Left":
                smoothed_volume = smooth(raw_percent, smoothed_volume)
                percent = smoothed_volume
                set_volume(percent)
            else:
                smoothed_brightness = smooth(raw_percent, smoothed_brightness)
                percent = smoothed_brightness
                set_brightness(percent)
            cv2.putText(frame, f"{action}: {percent:.0f}%", (points[0][0], points[0][1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            

    cv2.imshow("Two-Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()