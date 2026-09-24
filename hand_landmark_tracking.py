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
import threading
import sounddevice as sd
from openwakeword.model import Model

user32 = ctypes.windll.user32
pyautogui.FAILSAFE = True

virtual_x = user32.GetSystemMetrics(76)
virtual_y = user32.GetSystemMetrics(77)
virtual_w = user32.GetSystemMetrics(78)
virtual_h = user32.GetSystemMetrics(79)

smoothed_cursor = {"x": None, "y": None}

devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume

last_volume = None
last_brightness = None

OPEN_THRESHOLD = 6.80
FIST_THRESHOLD = 3.50
EXTENDED_THRESH = 1.15
ROTATION_SENSITIVITY = 0.8
MODE_STABILITY_FRAMES = 10
CLICK_THRESH = 0.4
click_active = False

MODE_NAMES = {0: "STANDBY", 1: "VOLUME", 2: "BRIGHTNESS", 3: "CURSOR", 4: "SHUTDOWN"}

GLOW_CYAN = (255, 220, 0)
GLOW_CYAN_DIM = (120, 90, 0)
ACCENT = (0, 255, 200)
WARNING = (0, 60, 255)
PANEL_BG = (40, 20, 0)

right_state = {
    "mode": 0,
    "pending_count": None,
    "pending_frames": 0,
    "prev_angle": None,
    "volume_value": 50,
    "brightness_value": 50,
}

left_last_action = None

# ============= WAKE WORD (toggle, no timer) =============
WAKE_PHRASE = "hey_jarvis"
WAKE_COOLDOWN = 1.5  # seconds — prevents one utterance from double-triggering the toggle

listening_active = False
last_wake_time = 0
wake_lock = threading.Lock()

def wakeword_listener():
    global listening_active, last_wake_time
    oww_model = Model(wakeword_models=[WAKE_PHRASE])

    def audio_callback(indata, frames, time_info, status):
        global listening_active, last_wake_time
        chunk = indata[:, 0]
        prediction = oww_model.predict(chunk)
        if prediction[WAKE_PHRASE] > 0.5:
            now = time.time()
            if now - last_wake_time > WAKE_COOLDOWN:
                with wake_lock:
                    listening_active = not listening_active
                last_wake_time = now
                print(f"Wake word detected — now {'ACTIVE' if listening_active else 'STANDBY'}")

    with sd.InputStream(channels=1, samplerate=16000, blocksize=1280,
                         dtype='int16', callback=audio_callback):
        while True:
            time.sleep(0.1)

threading.Thread(target=wakeword_listener, daemon=True).start()

# ============= HELPERS =============

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
    return np.clip((value - margin) / (1 - 2 * margin), 0.01, 0.99)

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
    if prev_smoothed is None:
        return new_value
    alpha = fast_alpha if distance_moved > speed_thresh else slow_alpha
    return prev_smoothed * (1 - alpha) + new_value * alpha

# ============= JARVIS-STYLE DRAWING HELPERS =============

def draw_glow_skeleton(frame, points, connections):
    for start, end in connections:
        cv2.line(frame, points[start], points[end], GLOW_CYAN_DIM, 4, cv2.LINE_AA)
        cv2.line(frame, points[start], points[end], GLOW_CYAN, 1, cv2.LINE_AA)
    for x, y in points:
        cv2.circle(frame, (x, y), 6, GLOW_CYAN_DIM, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 2, GLOW_CYAN, -1, cv2.LINE_AA)

def draw_progress_ring(frame, center, progress, radius=45, color=ACCENT):
    if progress <= 0:
        return
    cv2.ellipse(frame, center, (radius, radius), -90, 0, 360, GLOW_CYAN_DIM, 2, cv2.LINE_AA)
    cv2.ellipse(frame, center, (radius, radius), -90, 0, int(360 * progress), color, 3, cv2.LINE_AA)

def draw_hud_panel(frame, x, y, lines, w=220):
    line_h = 26
    panel_h = 20 + line_h * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + panel_h), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + panel_h), GLOW_CYAN, 1, cv2.LINE_AA)
    cv2.line(frame, (x, y), (x + 25, y), ACCENT, 2)
    cv2.line(frame, (x, y), (x, y + 25), ACCENT, 2)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (x + 12, y + 26 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

def draw_corner_brackets(frame, w, h, size=40, color=GLOW_CYAN_DIM, thickness=2):
    corners = [(0, 0, 1, 1), (w, 0, -1, 1), (0, h, 1, -1), (w, h, -1, -1)]
    for cx, cy, dx, dy in corners:
        cv2.line(frame, (cx, cy), (cx + dx * size, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * size), color, thickness, cv2.LINE_AA)

def draw_activation_banner(frame, w, active):
    if active:
        text = "ACTIVE"
        color = ACCENT
    else:
        text = 'say "Hey JARVIS" to activate'
        color = GLOW_CYAN_DIM
    cv2.putText(frame, text, (w // 2 - 140, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

# ============= MAIN =============

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
    draw_corner_brackets(frame, w, h)

    with wake_lock:
        active_now = listening_active

    draw_activation_banner(frame, w, active_now)

    if result.hand_landmarks and active_now:
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name
            label = "Right" if label == "Left" else "Left"

            points = []
            for lm in hand_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                points.append((x, y))
            draw_glow_skeleton(frame, points, connections)

            wrist_pt = points[0]

            # ============= LEFT HAND: play/pause =============
            if label == "Left":
                status_text = "IDLE"
                status_color = GLOW_CYAN_DIM
                if openness(hand_landmarks) > OPEN_THRESHOLD:
                    status_text, status_color = "PLAY", ACCENT
                    if left_last_action != "play":
                        pyautogui.press('playpause')
                        left_last_action = "play"
                elif is_fist(hand_landmarks):
                    status_text, status_color = "PAUSE", WARNING
                    if left_last_action != "pause":
                        pyautogui.press('playpause')
                        left_last_action = "pause"
                else:
                    left_last_action = None

                draw_hud_panel(frame, wrist_pt[0] - 60, wrist_pt[1] + 30,
                                [("MEDIA", ACCENT), (status_text, status_color)], w=140)

            # ============= RIGHT HAND: mode select + actions =============
            else:
                o = openness(hand_landmarks)
                angle = hand_angle(hand_landmarks)
                finger_count = count_extended_fingers(hand_landmarks)

                if right_state["mode"] == 0 and finger_count in (1, 2, 3, 4):
                    if right_state["pending_count"] == finger_count:
                        right_state["pending_frames"] += 1
                    else:
                        right_state["pending_count"] = finger_count
                        right_state["pending_frames"] = 1

                    draw_progress_ring(frame, wrist_pt, right_state["pending_frames"] / MODE_STABILITY_FRAMES)

                    if right_state["pending_frames"] >= MODE_STABILITY_FRAMES:
                        right_state["mode"] = finger_count
                        right_state["prev_angle"] = angle
                        right_state["pending_count"] = None
                        right_state["pending_frames"] = 0

                        if finger_count == 4:
                            overlay = frame.copy()
                            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 80), -1)
                            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                            cv2.putText(frame, "SYSTEM SHUTDOWN", (w // 2 - 180, h // 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, WARNING, 3, cv2.LINE_AA)
                            cv2.imshow("J.A.R.V.I.S.", frame)
                            cv2.waitKey(600)
                            cap.release()
                            cv2.destroyAllWindows()
                            exit()
                else:
                    right_state["pending_count"] = None
                    right_state["pending_frames"] = 0

                if right_state["mode"] in (1, 2, 3) and is_fist(hand_landmarks):
                    right_state["mode"] = 0
                    right_state["prev_angle"] = None

                elif right_state["mode"] == 1:
                    delta = angle_delta(angle, right_state["prev_angle"])
                    right_state["volume_value"] = np.clip(right_state["volume_value"] + delta * ROTATION_SENSITIVITY, 0, 100)
                    right_state["prev_angle"] = angle
                    set_volume(right_state["volume_value"])

                elif right_state["mode"] == 2:
                    delta = angle_delta(angle, right_state["prev_angle"])
                    right_state["brightness_value"] = np.clip(right_state["brightness_value"] + delta * ROTATION_SENSITIVITY, 0, 100)
                    right_state["prev_angle"] = angle
                    set_brightness(right_state["brightness_value"])

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
                            pyautogui.moveTo(smoothed_cursor["x"], smoothed_cursor["y"])
                        except pyautogui.FailSafeException:
                            pass

                    p = pinch_ratio(hand_landmarks)
                    if p < CLICK_THRESH and not click_active:
                        pyautogui.click()
                        click_active = True
                        cv2.circle(frame, wrist_pt, 55, ACCENT, 2, cv2.LINE_AA)
                    elif p >= CLICK_THRESH:
                        click_active = False

                mode_label = MODE_NAMES.get(right_state["mode"], "?")
                lines = [(f"MODE: {mode_label}", ACCENT)]
                if right_state["mode"] == 1:
                    lines.append((f"VOL  {right_state['volume_value']:.0f}%", GLOW_CYAN))
                elif right_state["mode"] == 2:
                    lines.append((f"BRT  {right_state['brightness_value']:.0f}%", GLOW_CYAN))
                lines.append((f"FINGERS: {finger_count}", GLOW_CYAN_DIM))

                draw_hud_panel(frame, wrist_pt[0] - 60, wrist_pt[1] + 30, lines, w=200)

    elif not active_now:
        cv2.putText(frame, "STANDBY", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GLOW_CYAN_DIM, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "NO SIGNAL", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GLOW_CYAN_DIM, 2, cv2.LINE_AA)

    cv2.imshow("J.A.R.V.I.S.", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()