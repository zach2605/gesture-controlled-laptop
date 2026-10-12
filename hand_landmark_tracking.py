"""
=======================================================================
 J.A.R.V.I.S. GESTURE + VOICE CONTROL
=======================================================================
WHAT THIS PROGRAM DOES, IN PLAIN ENGLISH:

  1. It watches your webcam and tracks your hands using a library
     called MediaPipe (it gives us 21 "landmark" points per hand,
     like skeleton joints).

  2. It listens to your microphone in the background for the phrase
     "Hey JARVIS" using a lightweight, fully-offline "wake word"
     detector (openWakeWord). Saying it toggles the whole system
     between STANDBY (does nothing) and ACTIVE (responds to you).

  3. Once ACTIVE, it ALSO listens for short command words (like
     "volume", "play", "quit") using a small offline speech-to-text
     engine called Vosk. This is a heavier tool than openWakeWord, so
     we only run it while ACTIVE, not all the time - that keeps the
     program light when you're not using it.

  4. Some things are controlled by VOICE (switching modes, play/pause,
     quitting) because a single spoken word is unambiguous - much
     more reliable than trying to detect a specific hand shape.

  5. Some things are controlled by GESTURE (turning the volume dial
     up/down, moving the mouse cursor) because these are continuous,
     "how much" or "where" actions - your voice has no natural way to
     say "a little more to the left", but your hand does.

  This is called a "hybrid" interface: voice for discrete one-shot
  commands, gesture for continuous/spatial control. Each tool is used
  for the job it's naturally good at.

WHY TWO SEPARATE AUDIO LISTENERS (openWakeWord AND Vosk)?
  openWakeWord is small and cheap to run constantly in the background
  - it does ONE thing: "does this sound like 'hey jarvis'?" That's
  all it can do, and that's why it's fast/light.

  Vosk is a full transcription engine - it can turn ANY speech into
  text. That flexibility costs more CPU. So we only turn it on while
  the system is ACTIVE, to avoid burning resources 24/7 for a feature
  you're only using part of the time.
=======================================================================
"""

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
import json
import math
import os
import subprocess
import queue
import re
import sounddevice as sd
from openwakeword.model import Model
from vosk import Model as VoskModel, KaldiRecognizer

# -----------------------------------------------------------------
# WINDOWS SYSTEM SETUP
# -----------------------------------------------------------------
# ctypes lets Python call directly into Windows' own system library
# (user32.dll). We use it here to find out the FULL size of your
# desktop, including any extra monitors plugged in - regular Python
# tools only know about your main screen by default.
user32 = ctypes.windll.user32
# pyautogui normally sleeps 0.1s after EVERY call. At that rate the cursor
# can only update ~10 times a second, which feels laggy. 0 removes the sleep.
pyautogui.PAUSE = 0
# The failsafe aborts if the cursor touches a screen corner. Now that your
# hand can legitimately reach corners (e.g. the window-close X in the top
# right), it would crash clicks there. Your safety exits are now: the
# voice command "exit", the "q" key, and a fist.
pyautogui.FAILSAFE = False

virtual_x = user32.GetSystemMetrics(76)   # left edge of ALL monitors combined
virtual_y = user32.GetSystemMetrics(77)   # top edge
virtual_w = user32.GetSystemMetrics(78)   # total width across all monitors
virtual_h = user32.GetSystemMetrics(79)   # total height across all monitors

# Where we remember the cursor's last smoothed position between frames
smoothed_cursor = {"x": None, "y": None}

# --- Windows audio (volume) control handle ---
devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume

# We track the LAST value we actually sent to the OS, so we don't
# spam the system with a new "set volume" call every single frame
# (30x/second) when the value barely changed.
last_volume = None
last_brightness = None

# -----------------------------------------------------------------
# TUNABLE THRESHOLDS
# -----------------------------------------------------------------
# These numbers were found by trial and error (see: earlier in this
# project's history). If gestures ever stop feeling right after a
# change of webcam/lighting/distance, THESE are the first things to
# re-test and adjust.
FIST_THRESHOLD = 3.50        # openness() below this = closed fist
ROTATION_SENSITIVITY = 0.8   # how many % volume/brightness changes per degree of rotation
CLICK_THRESH = 0.4           # pinch closeness (normalized) that counts as a click

# right_state["mode"] can be: "none", "volume", "brightness", "cursor"
# This single variable is the "brain" of what the right hand currently does.
right_state = {
    "mode": "none",
    "prev_angle": None,     # last frame's hand-rotation angle, used to compute deltas
    "volume_value": 50,     # current volume %, kept here so it persists across frames
    "brightness_value": 50,
}

click_active = False  # prevents one pinch from firing 30 clicks/second

# Colors used for the on-screen "HUD" (heads-up display), in BGR
# format because that's what OpenCV expects (NOT the usual RGB order!).
GLOW_CYAN = (255, 220, 0)
GLOW_CYAN_DIM = (120, 90, 0)
ACCENT = (0, 255, 200)
WARNING = (0, 60, 255)
PANEL_BG = (40, 20, 0)

MODE_NAMES = {"none": "STANDBY", "volume": "VOLUME", "brightness": "BRIGHTNESS", "cursor": "CURSOR"}


# =======================================================================
# LAYER 1: WAKE WORD (openWakeWord) - "Is the system ON or OFF?"
# =======================================================================
# This function runs FOREVER on its own background thread (see the
# threading.Thread(...) call near the bottom of this section). It has
# ONE job: listen to the mic, and flip `listening_active` between
# True/False every time it hears "Hey JARVIS".
#
# A "thread" is like a second, independent worker running alongside
# your main program - it lets us listen to audio continuously WITHOUT
# freezing the webcam/video loop, since they run at the same time
# instead of one waiting for the other.

WAKE_PHRASE = "hey_jarvis"
WAKE_COOLDOWN = 1.5  # seconds - stops one utterance from re-triggering itself

listening_active = False   # THE master on/off switch for the whole system
last_wake_time = 0
wake_lock = threading.Lock()
# A "lock" prevents two threads from reading/writing `listening_active`
# at the EXACT same instant and corrupting it - a classic bug in
# multi-threaded programs. Every time we touch shared state across
# threads, we grab the lock first.

def wakeword_listener():
    global listening_active, last_wake_time
    oww_model = Model(wakeword_models=[WAKE_PHRASE])

    def audio_callback(indata, frames, time_info, status):
        # This inner function is called automatically by sounddevice
        # every time a new small chunk of microphone audio is ready -
        # we never call it ourselves.
        global listening_active, last_wake_time
        chunk = indata[:, 0]  # take just one audio channel (mono)
        prediction = oww_model.predict(chunk)
        if prediction[WAKE_PHRASE] > 0.5:  # 0.5 = confidence threshold
            now = time.time()
            if now - last_wake_time > WAKE_COOLDOWN:
                with wake_lock:
                    listening_active = not listening_active  # TOGGLE, not just "turn on"
                last_wake_time = now
                print(f"[wake word] now {'ACTIVE' if listening_active else 'STANDBY'}")

    # This opens an ongoing connection to your microphone and keeps
    # calling audio_callback() automatically until the program ends.
    with sd.InputStream(channels=1, samplerate=16000, blocksize=1280,
                         dtype='int16', callback=audio_callback):
        while True:
            time.sleep(0.1)  # just keeps this thread alive; the real
                              # work happens inside audio_callback above


# =======================================================================
# LAYER 2: VOICE COMMANDS (Vosk) - "What should the system DO?"
# =======================================================================
# Unlike openWakeWord (which only recognizes ONE fixed phrase), Vosk
# can transcribe ANY speech into text. We take that text and check if
# it CONTAINS one of our command words. This only runs while
# `listening_active` is True, to save CPU while in standby.

VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"  # folder you extracted the zip into

# Single-word commands: spoken word -> internal command name
COMMAND_WORDS = {
    "volume": "volume",
    "brightness": "brightness",
    "cursor": "cursor",
    "exit": "exit_mode",
    "back": "exit_mode",       # two words that both mean "leave the current mode"
    "play": "play",
    "pause": "pause",
    "click": "click",
    "quit": "quit",
    "shutdown": "quit",        # two words that both mean "close the program"
    "dictate": "dictate",      # start typing what you say
    "search": "search",        # focus the address bar, then type what you say + Enter
    "confirm": "confirm_power",  # only does anything right after "power off"
    "cancel": "cancel_power",
}

# "open <app>" commands: spoken app name -> what Windows should launch.
# os.startfile() asks Windows to open the thing the same way double-clicking
# it would. "spotify:" is a URI Spotify registers when installed; the others
# are names Windows already knows. If an app won't open, replace the value
# with the full path to its .exe file.
APP_COMMANDS = {
    "spotify": "spotify:",
    "chrome": "chrome",
    "notepad": "notepad",
    "calculator": "calc",
    "explorer": "explorer",
}

def open_app(app_name):
    try:
        os.startfile(APP_COMMANDS[app_name])
        print(f"[app] opened {app_name}")
    except Exception as e:
        print(f"[app] could not open {app_name}: {e}")

# "close <app>" commands: spoken app name -> the Windows PROCESS name.
# Process names are what Task Manager shows (the "Details" tab).
# Note: explorer is deliberately NOT here. explorer.exe is also your
# desktop and taskbar, so killing it would make them vanish.
APP_PROCESSES = {
    "spotify": "Spotify.exe",
    "chrome": "chrome.exe",
    "notepad": "notepad.exe",
    "calculator": "CalculatorApp.exe",
}

# False = politely ask the app to close (it can still show "save your
# changes?"). True = force-kill immediately, which can LOSE unsaved work.
FORCE_CLOSE = False

def close_app(app_name):
    cmd = ["taskkill", "/IM", APP_PROCESSES[app_name]]
    if FORCE_CLOSE:
        cmd.append("/F")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[app] closed {app_name}")
    else:
        print(f"[app] could not close {app_name}: {result.stdout.strip() or result.stderr.strip()}")

# -----------------------------------------------------------------
# NAVIGATION COMMANDS: spoken phrase -> keyboard shortcut / scroll
# -----------------------------------------------------------------
# ("hotkey", ...) presses keys together, like Ctrl+Tab.
# ("scroll", n) scrolls the mouse wheel (negative = down).
# ("press", key) presses one key.
# These work in the window that currently has focus. Notepad (Windows 11)
# uses Ctrl+N for a new tab, while Chrome uses Ctrl+T, so "new tab"
# only opens a tab in browsers.
NAV_COMMANDS = {
    "next tab": ("hotkey", "ctrl", "tab"),
    "previous tab": ("hotkey", "ctrl", "shift", "tab"),
    "new tab": ("hotkey", "ctrl", "t"),
    "close tab": ("hotkey", "ctrl", "w"),
    "go back": ("hotkey", "alt", "left"),
    "go forward": ("hotkey", "alt", "right"),
    "scroll down": ("scroll", -500),
    "scroll up": ("scroll", 500),
    "undo": ("hotkey", "ctrl", "z"),
    "enter": ("press", "enter"),
    # --- Chrome ---
    "reload": ("hotkey", "ctrl", "r"),
    "reopen tab": ("hotkey", "ctrl", "shift", "t"),
    "new window": ("hotkey", "ctrl", "n"),
    "zoom in": ("hotkey", "ctrl", "="),
    "zoom out": ("hotkey", "ctrl", "-"),
    "find": ("hotkey", "ctrl", "f"),
    "bookmark": ("hotkey", "ctrl", "d"),
    # --- Works anywhere ---
    "mute": ("press", "volumemute"),
    "lock screen": ("run", ["rundll32.exe", "user32.dll,LockWorkStation"]),
    # Sleep = what a quick tap of the power button does on most laptops.
    # We use the .NET SetSuspendState call because the older rundll32
    # trick silently HIBERNATES instead of sleeping when hibernation is on.
    "go to sleep": ("run", ["powershell", "-NoProfile", "-Command",
                            "Add-Type -AssemblyName System.Windows.Forms; "
                            "[System.Windows.Forms.Application]::SetSuspendState('Suspend', $false, $false)"]),
    # --- YouTube (the video page must have focus) ---
    "skip ahead": ("press", "right"),     # +5 seconds
    "skip back": ("press", "left"),       # -5 seconds
    "fullscreen": ("press", "f"),
    "next video": ("hotkey", "shift", "n"),
}

# Phrases that start a multi-step action instead of a single keypress
PHRASE_COMMANDS = {"power off": "power_off"}

def run_nav(phrase):
    action = NAV_COMMANDS[phrase]
    if action[0] == "hotkey":
        pyautogui.hotkey(*action[1:])
    elif action[0] == "scroll":
        pyautogui.scroll(action[1])
    elif action[0] == "press":
        pyautogui.press(action[1])
    elif action[0] == "run":
        subprocess.Popen(action[1])   # Popen = start it and carry on (don't wait)

# -----------------------------------------------------------------
# DICTATION: while on, everything you say is typed into the focused window
# -----------------------------------------------------------------
# Say "dictate" to start, "stop dictation" to end. Dictation needs a
# recognizer with NO restricted vocabulary (it must be able to hear any
# word), so it is a separate Vosk recognizer from the command one.
dictating = False
search_mode = False   # True = one-shot: type ONE spoken query, press Enter, stop
typed_queue = queue.Queue()   # text waiting to be typed by the main loop

SPOKEN_PUNCTUATION = [("question mark", "?"), ("period", "."),
                      ("comma", ","), ("new line", "\n")]

def format_dictation(text):
    """Vosk returns lowercase words with no punctuation. We swap spoken
    punctuation words for symbols and capitalize the first letter."""
    for spoken, mark in SPOKEN_PUNCTUATION:
        text = re.sub(r"\s*\b" + spoken + r"\b", mark, text)
    return text[:1].upper() + text[1:]

VOICE_COOLDOWN = 1.0
last_voice_command = None      # what command we most recently acted on
last_voice_time = 0
voice_lock = threading.Lock()

def match_command(text):
    """Turn transcribed text into a command name, or None.
    We compare WHOLE WORDS, not substrings. Otherwise 'play' would
    wrongly match inside 'display'. Multi-word phrases are checked
    first, so "go back" is navigation while a lone "back" means
    "exit the current mode"."""
    words = text.split()
    padded = " " + " ".join(words) + " "
    for phrase, command in PHRASE_COMMANDS.items():
        if f" {phrase} " in padded:
            return command
    for phrase in NAV_COMMANDS:
        if f" {phrase} " in padded:
            return f"nav:{phrase}"
    if "open" in words:
        for app in APP_COMMANDS:
            if app in words:
                return f"open:{app}"
    if "close" in words:
        for app in APP_PROCESSES:
            if app in words:
                return f"close:{app}"
    for word in words:
        if word in COMMAND_WORDS:
            return COMMAND_WORDS[word]
    return None

def voice_command_listener():
    global last_voice_command, last_voice_time, dictating, search_mode
    vosk_model = VoskModel(VOSK_MODEL_PATH)

    # GRAMMAR: the command recognizer may ONLY choose from our words.
    # Anything else it hears becomes "[unk]". This makes recognition
    # faster and far more accurate, and helps when a video is playing.
    vocabulary = set(COMMAND_WORDS) | {"open", "close"} | set(APP_COMMANDS)
    for phrase in list(NAV_COMMANDS) + list(PHRASE_COMMANDS):
        vocabulary.update(phrase.split())
    vocabulary = list(vocabulary) + ["[unk]"]
    command_rec = KaldiRecognizer(vosk_model, 16000, json.dumps(vocabulary))

    # Dictation recognizer: no grammar, so it can transcribe anything.
    dictation_rec = KaldiRecognizer(vosk_model, 16000)

    def audio_callback(indata, frames, time_info, status):
        global last_voice_command, last_voice_time, dictating, search_mode

        # Only transcribe while ACTIVE (keeps CPU low in standby)
        with wake_lock:
            if not listening_active:
                dictating = False
                return

        audio_bytes = bytes(indata)

        # ---- DICTATION MODE: type what was said, nothing else ----
        if dictating:
            if dictation_rec.AcceptWaveform(audio_bytes):
                text = json.loads(dictation_rec.Result()).get("text", "")
                if search_mode:
                    # One-shot search: type the query, press Enter, done.
                    if text:
                        typed_queue.put(text)
                        typed_queue.put(("press", "enter"))
                        print(f"[search] {text}")
                        dictating = False
                        search_mode = False
                        command_rec.Reset()
                    return
                stop = "stop dictation" in text
                text = text.replace("stop dictation", "").strip()
                if text:
                    typed_queue.put(format_dictation(text) + " ")
                    print(f"[dictation] {text}")
                if stop:
                    dictating = False
                    command_rec.Reset()
                    print("[dictation] stopped")
            return

        # ---- COMMAND MODE ----
        # Vosk only declares an utterance "finished" after silence, and a
        # playing video never goes silent. So we also read PartialResult(),
        # the text heard so far, and react without waiting for silence.
        if command_rec.AcceptWaveform(audio_bytes):
            text = json.loads(command_rec.Result()).get("text", "")
        else:
            text = json.loads(command_rec.PartialResult()).get("partial", "")

        command = match_command(text)
        if command:
            now = time.time()
            if now - last_voice_time > VOICE_COOLDOWN:
                last_voice_time = now
                print(f"[voice] heard '{text}' -> command: {command}")
                if command in ("dictate", "search"):
                    dictating = True
                    search_mode = (command == "search")
                    dictation_rec.Reset()
                    if search_mode:
                        # Ctrl+L puts the cursor in Chrome's address bar
                        typed_queue.put(("hotkey", "ctrl", "l"))
                        print("[search] say what to search for")
                    else:
                        print("[dictation] started - say 'stop dictation' to end")
                else:
                    with voice_lock:
                        last_voice_command = command
            # Clear what Vosk has heard so far, so the same words are
            # not matched again on the next audio chunk.
            command_rec.Reset()

    with sd.InputStream(channels=1, samplerate=16000, blocksize=4000,
                         dtype='int16', callback=audio_callback):
        while True:
            time.sleep(0.1)


def consume_voice_command():
    """
    Called once per video frame. 'Consumes' means: read the command,
    then immediately clear it, so we only ever act on it ONE time,
    not on every single frame until the next word is spoken.
    """
    global last_voice_command
    with voice_lock:
        cmd = last_voice_command
        last_voice_command = None
    return cmd


# Start both audio listeners as background threads. `daemon=True`
# means: if the main program exits, kill these threads automatically
# rather than leaving them running forever in the background.
threading.Thread(target=wakeword_listener, daemon=True).start()
threading.Thread(target=voice_command_listener, daemon=True).start()


# =======================================================================
# HAND GEOMETRY HELPERS
# =======================================================================
# MediaPipe gives us 21 (x, y, z) points per hand, numbered 0-20.
# Point 0 is always the wrist. The rest trace up each finger.
# None of these functions know anything about "gestures" - they just
# do plain geometry (distances, angles) on those raw points. The
# ACTUAL gesture logic (what a "fist" or "rotation" means) lives
# further down, built on top of these basic measurements.

def set_volume(percent):
    """Actually changes the OS volume - but only if it changed enough
    to be worth bothering the operating system about."""
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
    """Returns which direction the hand is 'pointing', in degrees,
    using the line from the wrist (point 0) to the middle-finger
    knuckle (point 9). This is what we rotate to adjust volume/brightness."""
    wrist = hand_landmarks[0]
    mid_mcp = hand_landmarks[9]
    dx = mid_mcp.x - wrist.x
    dy = mid_mcp.y - wrist.y
    return np.degrees(np.arctan2(dy, dx))

def openness(hand_landmarks):
    """A single number describing 'how open is this hand'. We measure
    each fingertip's distance from the wrist, add them up, then divide
    by the palm's own length. Dividing by palm length is important:
    without it, this number would also depend on how close your hand
    is to the camera, which would make a fist at 20cm look 'open' to
    the code compared to an actual open hand at 60cm."""
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y])
    palm_ref = np.array([hand_landmarks[9].x, hand_landmarks[9].y])
    palm_length = np.linalg.norm(palm_ref - wrist)
    total = 0
    for tip_id in [8, 12, 16, 20]:  # index, middle, ring, pinky fingertips
        tip = np.array([hand_landmarks[tip_id].x, hand_landmarks[tip_id].y])
        total += np.linalg.norm(tip - wrist)
    return total / (palm_length + 1e-6)  # +1e-6 avoids a divide-by-zero crash

def angle_delta(current, previous):
    """How much did the rotation angle change between frames? This
    sounds trivial (just subtract!) but angles wrap around at 180/-180
    degrees, so a naive subtraction can produce a fake huge jump right
    at that boundary. This function corrects for that wraparound."""
    if previous is None:
        return 0
    diff = current - previous
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff

def finger_extended_ratios(hand_landmarks):
    """Same normalization idea as openness(), but per-finger instead
    of summed together. Returns something like
    {'index': 1.4, 'middle': 0.6, 'ring': 0.5, 'pinky': 0.55}
    - a high number means that finger is extended, low means curled."""
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y])
    palm_ref = np.array([hand_landmarks[9].x, hand_landmarks[9].y])
    palm_length = np.linalg.norm(palm_ref - wrist)
    ratios = {}
    for name, tip_id in [("index", 8), ("middle", 12), ("ring", 16), ("pinky", 20)]:
        tip = np.array([hand_landmarks[tip_id].x, hand_landmarks[tip_id].y])
        ratios[name] = np.linalg.norm(tip - wrist) / (palm_length + 1e-6)
    return ratios

def is_pointing(hand_landmarks, extended_thresh=1.3, curled_thresh=0.9):
    """True only when the index finger is clearly extended AND the
    other three are clearly curled - this is our 'move the cursor' shape."""
    r = finger_extended_ratios(hand_landmarks)
    return (r["index"] > extended_thresh and
            r["middle"] < curled_thresh and
            r["ring"] < curled_thresh and
            r["pinky"] < curled_thresh)

def is_fist(hand_landmarks, curled_thresh=0.75):
    """True only when ALL FOUR tracked fingers are curled. Note this
    is stricter than just 'low openness()' - openness() sums the
    fingers together, so 'pointing' (3 curled + 1 extended) can
    accidentally look similar to a real fist if you just check the
    sum. Checking each finger individually avoids that mix-up."""
    r = finger_extended_ratios(hand_landmarks)
    return all(v < curled_thresh for v in r.values())

def pinch_ratio(hand_landmarks):
    """Normalized distance between thumb tip and index tip - this is
    our 'click' gesture. Small number = fingers pinched together."""
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y])
    palm_ref = np.array([hand_landmarks[9].x, hand_landmarks[9].y])
    palm_length = np.linalg.norm(palm_ref - wrist)
    thumb_tip = np.array([hand_landmarks[4].x, hand_landmarks[4].y])
    index_tip = np.array([hand_landmarks[8].x, hand_landmarks[8].y])
    return np.linalg.norm(thumb_tip - index_tip) / (palm_length + 1e-6)

def map_with_margin(value, margin=0.2):
    """Takes a 0.0-1.0 landmark coordinate and rescales it so only the
    MIDDLE part of the camera's view (excluding a margin near each
    edge) maps to the full 0.0-1.0 output range. Without this, you'd
    have to physically reach the very edge of the webcam's view to get
    the cursor to the edge of your screen."""
    return np.clip((value - margin) / (1 - 2 * margin), 0.01, 0.99)

class OneEuroFilter:
    """Smooths a jittery signal WITHOUT adding lag during fast movement.
    The trick: when the signal is nearly still, smooth a lot (kills hand
    jitter); when it is moving fast, smooth very little (no lag). It is
    the standard filter for pointer tracking.
      min_cutoff: higher = less smoothing while still (more jitter)
      beta:       higher = less lag while moving fast (more jitter when fast)"""
    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt                      # current speed
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev     # smoothed speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)  # faster = looser
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat

CURSOR_MIN_CUTOFF = 1.0   # raise if the cursor feels sluggish when slow
CURSOR_BETA = 0.01        # raise if it lags behind fast hand movements
cursor_filter_x = OneEuroFilter(CURSOR_MIN_CUTOFF, CURSOR_BETA)
cursor_filter_y = OneEuroFilter(CURSOR_MIN_CUTOFF, CURSOR_BETA)

def move_cursor(x, y):
    """Place the cursor directly using the Windows API. This is instant,
    unlike pyautogui.moveTo, and works across multiple monitors."""
    user32.SetCursorPos(int(x), int(y))

def adaptive_smooth(new_value, prev_smoothed, distance_moved,
                     slow_alpha=0.15, fast_alpha=0.6, speed_thresh=15):
    """Blends a new position with the previous one to reduce jittery
    camera noise. The 'adaptive' part: when your hand is moving fast,
    we trust the NEW position more (feels responsive); when it's
    nearly still, we trust the OLD smoothed position more (kills
    jitter so the cursor doesn't shake while you're trying to hold still)."""
    if prev_smoothed is None:
        return new_value
    alpha = fast_alpha if distance_moved > speed_thresh else slow_alpha
    return prev_smoothed * (1 - alpha) + new_value * alpha


# =======================================================================
# DRAWING HELPERS ("JARVIS HUD" look) - purely cosmetic, no logic here
# =======================================================================

def draw_glow_skeleton(frame, points, connections):
    """Draws each hand bone TWICE - once thick and dim, once thin and
    bright, layered on top of each other. That layering is what fakes
    a 'glow' effect using only plain OpenCV drawing calls."""
    for start, end in connections:
        cv2.line(frame, points[start], points[end], GLOW_CYAN_DIM, 4, cv2.LINE_AA)
        cv2.line(frame, points[start], points[end], GLOW_CYAN, 1, cv2.LINE_AA)
    for x, y in points:
        cv2.circle(frame, (x, y), 6, GLOW_CYAN_DIM, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 2, GLOW_CYAN, -1, cv2.LINE_AA)

def draw_hud_panel(frame, x, y, lines, w=220):
    """Draws a semi-transparent dark box with a cyan border - our
    'status readout' style panel. `lines` is a list of (text, color) pairs."""
    line_h = 26
    panel_h = 20 + line_h * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + panel_h), PANEL_BG, -1)
    # addWeighted blends two images together - this is how we get the
    # "semi-transparent" look, by mixing the solid panel with the
    # original frame underneath it.
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + panel_h), GLOW_CYAN, 1, cv2.LINE_AA)
    cv2.line(frame, (x, y), (x + 25, y), ACCENT, 2)
    cv2.line(frame, (x, y), (x, y + 25), ACCENT, 2)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (x + 12, y + 26 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

def draw_corner_brackets(frame, w, h, size=40, color=GLOW_CYAN_DIM, thickness=2):
    """Draws four L-shaped brackets in the corners of the screen -
    a 'targeting reticle' look, purely decorative."""
    corners = [(0, 0, 1, 1), (w, 0, -1, 1), (0, h, 1, -1), (w, h, -1, -1)]
    for cx, cy, dx, dy in corners:
        cv2.line(frame, (cx, cy), (cx + dx * size, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * size), color, thickness, cv2.LINE_AA)

def draw_activation_banner(frame, w, active):
    text = "ACTIVE" if active else 'say "Hey JARVIS" to activate'
    color = ACCENT if active else GLOW_CYAN_DIM
    cv2.putText(frame, text, (w // 2 - 140, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


# =======================================================================
# MAIN PROGRAM
# =======================================================================

# This loads MediaPipe's hand-tracking model. `running_mode=VIDEO`
# tells it to expect a continuous stream of frames (as opposed to
# single unrelated photos), which lets it use info from previous
# frames to track more smoothly.
base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
landmarker = vision.HandLandmarker.create_from_options(options)

# This list defines which landmark points connect to which, so we
# know how to draw the "skeleton" lines between the 21 dots.
connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

pending_power_off = None     # time "power off" was heard, or None
POWER_CONFIRM_WINDOW = 10    # seconds you have to say "confirm"

# CAP_DSHOW is Windows' lower-latency camera backend. A 640x480 picture is
# plenty for hand tracking and is faster to process than HD. Buffer size 1
# stops old frames queueing up, which would add delay.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
prev_frame_time = time.time()
frame_timestamp_ms = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break  # webcam disconnected or errored - stop the program

    frame = cv2.flip(frame, 1)  # mirror the image so it feels like
                                  # looking in a mirror, not a selfie camera
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe wants
                                                          # RGB, OpenCV gives us BGR
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    frame_timestamp_ms += 33  # roughly 30 frames per second
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    h, w, _ = frame.shape
    draw_corner_brackets(frame, w, h)

    # FPS readout: if this number is low (under ~20), the lag comes from
    # processing speed rather than from cursor smoothing.
    _t = time.time()
    fps = 1.0 / max(_t - prev_frame_time, 1e-3)
    prev_frame_time = _t
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 110, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GLOW_CYAN_DIM, 1, cv2.LINE_AA)

    # Safely read the shared "is the system on?" flag set by the
    # wake-word thread. We use the lock so we never read it at the
    # exact instant the other thread is in the middle of writing it.
    with wake_lock:
        active_now = listening_active

    draw_activation_banner(frame, w, active_now)
    if dictating:
        banner = "SEARCH - say what to search for" if search_mode else "DICTATING - say 'stop dictation' to end"
        cv2.putText(frame, banner, (w // 2 - 230, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WARNING, 2, cv2.LINE_AA)

    # Power-off safety: "power off" only ARMS shutdown for 10 seconds.
    if pending_power_off and time.time() - pending_power_off > POWER_CONFIRM_WINDOW:
        pending_power_off = None
    if pending_power_off:
        left = POWER_CONFIRM_WINDOW - (time.time() - pending_power_off)
        cv2.putText(frame, f"POWER OFF? say 'confirm' ({left:.0f}s) or 'cancel'", (w // 2 - 260, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WARNING, 2, cv2.LINE_AA)

    # Type any dictated text waiting in the queue. We do this on the main
    # thread (not the audio thread) so slow typing can never stall audio.
    while not typed_queue.empty():
        item = typed_queue.get()
        if isinstance(item, tuple):          # a keypress, not text
            if item[0] == "hotkey":
                pyautogui.hotkey(*item[1:])
                time.sleep(0.15)             # give the window time to react
            elif item[0] == "press":
                pyautogui.press(item[1])
        else:
            pyautogui.write(item, interval=0.01)

    # ---- Handle any voice command that arrived since the last frame ----
    if active_now:
        voice_cmd = consume_voice_command()
        if voice_cmd == "volume":
            right_state["mode"] = "volume"
            right_state["prev_angle"] = None  # reset so rotation starts fresh
        elif voice_cmd == "brightness":
            right_state["mode"] = "brightness"
            right_state["prev_angle"] = None
        elif voice_cmd == "cursor":
            right_state["mode"] = "cursor"
            cursor_filter_x.reset()
            cursor_filter_y.reset()
        elif voice_cmd == "exit_mode":
            right_state["mode"] = "none"
            right_state["prev_angle"] = None
        elif voice_cmd == "play":
            pyautogui.press('playpause')
        elif voice_cmd == "pause":
            pyautogui.press('playpause')
        elif voice_cmd == "click":
            pyautogui.click()
        elif voice_cmd and voice_cmd.startswith("nav:"):
            run_nav(voice_cmd.split(":", 1)[1])
        elif voice_cmd and voice_cmd.startswith("open:"):
            open_app(voice_cmd.split(":", 1)[1])
        elif voice_cmd and voice_cmd.startswith("close:"):
            close_app(voice_cmd.split(":", 1)[1])
        elif voice_cmd == "power_off":
            pending_power_off = time.time()
        elif voice_cmd == "confirm_power":
            if pending_power_off:
                # /t 15 = Windows waits 15 seconds, so "cancel" can still stop it
                subprocess.run(["shutdown", "/s", "/t", "15"])
                pending_power_off = None
                print("[power] shutting down in 15s - say 'cancel' to abort")
        elif voice_cmd == "cancel_power":
            pending_power_off = None
            subprocess.run(["shutdown", "/a"], capture_output=True)
        elif voice_cmd == "quit":
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

    # ---- Handle gestures, only if the system is ACTIVE ----
    if result.hand_landmarks and active_now:
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            # MediaPipe's left/right label is based on the UN-flipped
            # camera image, but we flipped the frame for a mirror
            # view - so we have to flip the label too, or it'll say
            # "Left" while pointing at your actual right hand.
            label = handedness[0].category_name
            label = "Right" if label == "Left" else "Left"

            points = []
            for lm in hand_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                points.append((x, y))
            draw_glow_skeleton(frame, points, connections)
            wrist_pt = points[0]

            # Only the RIGHT hand does anything now - voice handles
            # play/pause/click/quit, so the left hand is currently
            # free/unused (a natural spot to add a new feature later).
            if label != "Right":
                continue

            angle = hand_angle(hand_landmarks)

            # --- Exit current mode via fist (kept as a gesture backup
            #     alongside the "exit"/"back" voice command) ---
            if right_state["mode"] != "none" and is_fist(hand_landmarks):
                right_state["mode"] = "none"
                right_state["prev_angle"] = None

            elif right_state["mode"] == "volume":
                delta = angle_delta(angle, right_state["prev_angle"])
                right_state["volume_value"] = np.clip(
                    right_state["volume_value"] + delta * ROTATION_SENSITIVITY, 0, 100)
                right_state["prev_angle"] = angle
                set_volume(right_state["volume_value"])

            elif right_state["mode"] == "brightness":
                delta = angle_delta(angle, right_state["prev_angle"])
                right_state["brightness_value"] = np.clip(
                    right_state["brightness_value"] + delta * ROTATION_SENSITIVITY, 0, 100)
                right_state["prev_angle"] = angle
                set_brightness(right_state["brightness_value"])

            elif right_state["mode"] == "cursor":
                p = pinch_ratio(hand_landmarks)

                # Anchor the cursor to the KNUCKLES (landmarks 5 and 9), not
                # the fingertip. Pinching to click moves the fingertip, which
                # used to drag the cursor off its target. Knuckles stay put.
                anchor_x = (hand_landmarks[5].x + hand_landmarks[9].x) / 2
                anchor_y = (hand_landmarks[5].y + hand_landmarks[9].y) / 2
                target_x = virtual_x + map_with_margin(anchor_x) * virtual_w
                target_y = virtual_y + map_with_margin(anchor_y) * virtual_h

                # Freeze the cursor while a pinch is closing, so clicks land
                # exactly where you aimed. No is_pointing() requirement
                # anymore: shape flicker used to make the cursor stutter.
                if p > CLICK_THRESH * 1.3:
                    now_t = time.time()
                    move_cursor(cursor_filter_x(target_x, now_t),
                                cursor_filter_y(target_y, now_t))

                # Pinch-to-click still works as a gesture backup
                # alongside the "click" voice command.
                if p < CLICK_THRESH and not click_active:
                    pyautogui.click()
                    click_active = True
                    cv2.circle(frame, wrist_pt, 55, ACCENT, 2, cv2.LINE_AA)
                elif p >= CLICK_THRESH:
                    click_active = False

            # ---- HUD readout for whatever mode we're in ----
            mode_label = MODE_NAMES.get(right_state["mode"], "?")
            lines = [(f"MODE: {mode_label}", ACCENT)]
            if right_state["mode"] == "volume":
                lines.append((f"VOL  {right_state['volume_value']:.0f}%", GLOW_CYAN))
            elif right_state["mode"] == "brightness":
                lines.append((f"BRT  {right_state['brightness_value']:.0f}%", GLOW_CYAN))
            draw_hud_panel(frame, wrist_pt[0] - 60, wrist_pt[1] + 30, lines, w=200)

    elif not active_now:
        cv2.putText(frame, "STANDBY", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GLOW_CYAN_DIM, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "NO SIGNAL", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GLOW_CYAN_DIM, 2, cv2.LINE_AA)

    cv2.imshow("J.A.R.V.I.S.", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # manual keyboard quit, always works
        break

cap.release()
cv2.destroyAllWindows()