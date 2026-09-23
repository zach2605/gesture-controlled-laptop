import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities
import time
import pyautogui
import ctypes


user32 = ctypes.windll.user32

pyautogui.FAILSAFE = True

virtual_x = user32.GetSystemMetrics(76)   # left edge of virtual desktop (can be negative)
virtual_y = user32.GetSystemMetrics(77)   # top edge
virtual_w = user32.GetSystemMetrics(78)   # full combined width
virtual_h = user32.GetSystemMetrics(79)   # full combined height

smoothed_cursor = {"x": None, "y": None}
# CURSOR_SMOOTHING = 0.4

devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume

last_volume = None
last_brightness = None

OPEN_THRESHOLD = 6.80
FIST_THRESHOLD = 3.50
EXTENDED_THRESH = 1.15
ROTATION_SENSITIVITY = 0.8
MODE_STABILITY_FRAMES = 10   # how many consecutive frames a finger count must hold before switching mode
CLICK_THRESH = 0.4   # tune after testing — normalized ratio, not raw distance
click_active = False

MODE_NAMES = {0: "NONE", 1: "VOLUME", 2: "BRIGHTNESS", 3: "CURSOR", 4: "QUIT"}

# --- Right-hand mode state ---
right_state = {
    "mode": 0,                # 0 = no mode active
    "pending_count": None,    # finger count currently being held, waiting to stabilize
    "pending_frames": 0,
    "prev_angle": None,
    "volume_value": 50,
    "brightness_value": 50,
}

# --- Left-hand play/pause debounce ---
left_last_action = None

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

def hand_angle(hand_landmarks):
    wrist = hand_landmarks[0]
    mid_mcp = hand_landmarks[9]
    dx = mid_mcp.x - wrist.x
    dy = mid_mcp.y - wrist.y
    return np.degrees(np.arctan2(dy, dx))

def openness(hand_landmarks):
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y])
    palm_ref = np.array([hand_landmarks[9].x, hand_landmarks[9].y])
    palm_length = np.linalg.norm(palm_ref - wrist)
    total = 0
    for tip_id in [8, 12, 16, 20]:
        tip = np.array([hand_landmarks[tip_id].x, hand_landmarks[tip_id].y])
        total += np.linalg.norm(tip - wrist)
    return total / (palm_length + 1e-6)

def angle_delta(current, previous):
    if previous is None:
        return 0
    diff = current - previous
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff

def finger_extended_ratios(hand_landmarks):
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y])
    palm_ref = np.array([hand_landmarks[9].x, hand_landmarks[9].y])
    palm_length = np.linalg.norm(palm_ref - wrist)
    ratios = {}
    for name, tip_id in [("index", 8), ("middle", 12), ("ring", 16), ("pinky", 20)]:
        tip = np.array([hand_landmarks[tip_id].x, hand_landmarks[tip_id].y])
        ratios[name] = np.linalg.norm(tip - wrist) / (palm_length + 1e-6)
    return ratios

def count_extended_fingers(hand_landmarks):
    r = finger_extended_ratios(hand_landmarks)
    return sum(1 for v in r.values() if v > EXTENDED_THRESH)

def is_pointing(hand_landmarks, extended_thresh=1.3, curled_thresh=0.9):
    r = finger_extended_ratios(hand_landmarks)
    return (r["index"] > extended_thresh and
            r["middle"] < curled_thresh and
            r["ring"] < curled_thresh and
            r["pinky"] < curled_thresh)

def is_thumbs_up(hand_landmarks):
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y])
    thumb_tip = np.array([hand_landmarks[4].x, hand_landmarks[4].y])
    direction = thumb_tip - wrist
    angle = np.degrees(np.arctan2(-direction[1], direction[0])) 
    r = finger_extended_ratios(hand_landmarks)
    others_curled = all(v < 0.9 for v in r.values())
    return 60 < angle < 120 and others_curled 

def is_thumbs_down(hand_landmarks):
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y])
    thumb_tip = np.array([hand_landmarks[4].x, hand_landmarks[4].y])
    direction = thumb_tip - wrist
    angle = np.degrees(np.arctan2(-direction[1], direction[0]))
    r = finger_extended_ratios(hand_landmarks)
    others_curled = all(v < 0.9 for v in r.values())
    return -120 < angle < -60 and others_curled

def map_with_margin(value, margin=0.2):
    clamped = np.clip((value - margin) / (1 - 2 * margin), 0.01, 0.99)  
    return clamped

def is_fist(hand_landmarks, curled_thresh=0.75):
    r = finger_extended_ratios(hand_landmarks)
    return all(v < curled_thresh for v in r.values())

def pinch_ratio(hand_landmarks):
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y])
    palm_ref = np.array([hand_landmarks[9].x, hand_landmarks[9].y])
    palm_length = np.linalg.norm(palm_ref - wrist)
    thumb_tip = np.array([hand_landmarks[4].x, hand_landmarks[4].y])
    index_tip = np.array([hand_landmarks[8].x, hand_landmarks[8].y])
    return np.linalg.norm(thumb_tip - index_tip) / (palm_length + 1e-6)

def adaptive_smooth(new_value, prev_smoothed, distance_moved, slow_alpha=0.15, fast_alpha=0.6, speed_thresh=15):
    """Lower alpha (more smoothing) when barely moving, higher alpha (more responsive) when moving fast."""
    if prev_smoothed is None:
        return new_value
    alpha = fast_alpha if distance_moved > speed_thresh else slow_alpha
    return prev_smoothed * (1 - alpha) + new_value * alpha



base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
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
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name
            label = "Right" if label == "Left" else "Left"

            points = []
            for lm in hand_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                points.append((x, y))
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            for start, end in connections:
                cv2.line(frame, points[start], points[end], (255, 255, 255), 2)

            # ============= LEFT HAND: play/pause =============
            if label == "Left":
                if openness(hand_landmarks) > OPEN_THRESHOLD:
                    if left_last_action != "play":
                        pyautogui.press('playpause')
                        left_last_action = "play"
                elif is_fist(hand_landmarks):
                    if left_last_action != "pause":
                        pyautogui.press('playpause')
                        left_last_action = "pause"
                else:
                    left_last_action = None

            # ============= RIGHT HAND: mode select + actions =============
            else:
                o = openness(hand_landmarks)
                angle = hand_angle(hand_landmarks)
                finger_count = count_extended_fingers(hand_landmarks)

                # --- Mode switching (only when currently in NO mode) ---
                if right_state["mode"] == 0 and finger_count in (1, 2, 3, 4):
                    if right_state["pending_count"] == finger_count:
                        right_state["pending_frames"] += 1
                    else:
                        right_state["pending_count"] = finger_count
                        right_state["pending_frames"] = 1

                    if right_state["pending_frames"] >= MODE_STABILITY_FRAMES:
                        right_state["mode"] = finger_count
                        right_state["prev_angle"] = angle
                        right_state["pending_count"] = None
                        right_state["pending_frames"] = 0

                        if finger_count == 4:
                            cv2.putText(frame, "QUITTING...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                            cv2.imshow("Gesture Control", frame)
                            cv2.waitKey(500)
                            cap.release()
                            cv2.destroyAllWindows()
                            exit()
                else:
                    right_state["pending_count"] = None
                    right_state["pending_frames"] = 0

                # --- Exit current mode via fist ---
                if right_state["mode"] in (1, 2, 3) and is_fist(hand_landmarks):
                    right_state["mode"] = 0
                    right_state["prev_angle"] = None

                # --- Mode 1: Volume (rotation) ---
                elif right_state["mode"] == 1:
                    delta = angle_delta(angle, right_state["prev_angle"])
                    right_state["volume_value"] = np.clip(right_state["volume_value"] + delta * ROTATION_SENSITIVITY, 0, 100)
                    right_state["prev_angle"] = angle
                    set_volume(right_state["volume_value"])

                # --- Mode 2: Brightness (rotation) ---
                elif right_state["mode"] == 2:
                    delta = angle_delta(angle, right_state["prev_angle"])
                    right_state["brightness_value"] = np.clip(right_state["brightness_value"] + delta * ROTATION_SENSITIVITY, 0, 100)
                    right_state["prev_angle"] = angle
                    set_brightness(right_state["brightness_value"])

                # --- Mode 3: Cursor (index finger position, pinch to click) ---
                elif right_state["mode"] == 3:
                    if is_pointing(hand_landmarks):
                        index_tip = hand_landmarks[8]
                        target_x = virtual_x + map_with_margin(index_tip.x) * virtual_w
                        target_y = virtual_y + map_with_margin(index_tip.y) * virtual_h

                        if smoothed_cursor["x"] is None:
                            smoothed_cursor["x"], smoothed_cursor["y"] = target_x, target_y
                        else:
                            dist_moved = np.hypot(target_x - smoothed_cursor["x"], target_y - smoothed_cursor["y"])
                            smoothed_cursor["x"] = adaptive_smooth(target_x, smoothed_cursor["x"], dist_moved)
                            smoothed_cursor["y"] = adaptive_smooth(target_y, smoothed_cursor["y"], dist_moved)

                        try:
                            print(f"target: {smoothed_cursor['x']:.0f}, {smoothed_cursor['y']:.0f} | virtual origin: {virtual_x}, {virtual_y}")
                            pyautogui.moveTo(smoothed_cursor["x"], smoothed_cursor["y"])
                        except pyautogui.FailSafeException:
                            pass  # skip this frame's move, don't crash

                    # pinch (thumb-index close together) while NOT full pointing shape = click
                    p = pinch_ratio(hand_landmarks)
                    if p < CLICK_THRESH and not click_active:
                        pyautogui.click()
                        click_active = True
                        cv2.putText(frame, "CLICK", points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    elif p >= CLICK_THRESH:
                        click_active = False

                # --- HUD text ---
                mode_label = MODE_NAMES.get(right_state["mode"], "?")
                cv2.putText(frame, f"MODE: {mode_label}", points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                if right_state["mode"] == 1:
                    cv2.putText(frame, f"Volume: {right_state['volume_value']:.0f}%", (points[0][0], points[0][1] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                elif right_state["mode"] == 2:
                    cv2.putText(frame, f"Brightness: {right_state['brightness_value']:.0f}%", (points[0][0], points[0][1] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.putText(frame, f"fingers:{finger_count}", (points[0][0], points[0][1] + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()