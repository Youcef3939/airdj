import cv2 # type: ignore
import numpy as np # type: ignore 
import mediapipe as mp # type: ignore
import sounddevice as sd # type: ignore
import librosa # type: ignore
import pyrubberband as pyrb # type: ignore
import threading
import queue
import time

FILENAME = "song.mp3"

y, sr = librosa.load(FILENAME, sr=None)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def get_distance(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

def scale(val, src_min, src_max, dst_min, dst_max):
    val = max(min(val, src_max), src_min)
    return dst_min + (dst_max - dst_min) * ((val - src_min) / (src_max - src_min))

class SmoothValue:
    def __init__(self, alpha=0.15, init=0.0):
        self.alpha = alpha
        self.value = init
    def update(self, new_val):
        self.value = self.alpha * new_val + (1 - self.alpha) * self.value
        return self.value

speed_smooth = SmoothValue(init=1.0)
pitch_smooth = SmoothValue(init=0.0)
volume_smooth = SmoothValue(init=1.0)

chunk_duration = 0.2  # seconds - smaller chunk for lower latency and faster processing
chunk_size = int(sr * chunk_duration)
pos = 0
audio_queue = queue.Queue(maxsize=30)  # bigger queue for buffering

# Use time.monotonic for accurate timing
last_produce_time = time.monotonic()

def audio_callback(outdata, frames, time_info, status):
    try:
        chunk = audio_queue.get_nowait()
        length = len(chunk)
        if length < frames:
            outdata[:length, 0] = chunk
            outdata[length:, 0] = 0
        else:
            outdata[:, 0] = chunk[:frames]
    except queue.Empty:
        outdata.fill(0)

def audio_producer():
    global pos, last_produce_time
    while True:
        now = time.monotonic()
        elapsed = now - last_produce_time

        # Produce next chunk only if enough time has passed
        if elapsed < chunk_duration * 0.9:  # produce a bit earlier than duration to stay ahead
            time.sleep(0.005)
            continue

        if pos + chunk_size > len(y):
            pos = 0

        cur_speed = speed_smooth.value
        cur_pitch = pitch_smooth.value
        cur_volume = volume_smooth.value

        chunk = y[pos:pos+chunk_size]

        # Apply pitch and speed changes
        # If speed=1 and pitch=0, skip processing for speed
        if abs(cur_speed - 1.0) > 0.01 or abs(cur_pitch) > 0.01:
            try:
                chunk_mod = pyrb.time_stretch(chunk, sr, cur_speed)
                chunk_mod = pyrb.pitch_shift(chunk_mod, sr, cur_pitch)
            except Exception as e:
                print(f"Audio processing error: {e}")
                chunk_mod = chunk
        else:
            chunk_mod = chunk

        chunk_mod = chunk_mod * cur_volume
        chunk_mod = np.clip(chunk_mod, -1.0, 1.0)

        try:
            audio_queue.put(chunk_mod, timeout=0.5)
            last_produce_time = now
            pos += chunk_size
        except queue.Full:
            # Drop chunk if queue full (avoid stalling)
            pass

stream = sd.OutputStream(channels=1, samplerate=sr, callback=audio_callback, blocksize=chunk_size)
stream.start()

producer_thread = threading.Thread(target=audio_producer, daemon=True)
producer_thread.start()

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    left_hand = None
    right_hand = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_type in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_type.classification[0].label
            if label == 'Left':
                left_hand = hand_landmarks
            elif label == 'Right':
                right_hand = hand_landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if left_hand:
        dist_l = get_distance(left_hand.landmark[4], left_hand.landmark[8])
        raw_pitch = scale(dist_l, 0.01, 0.25, -12, 12)
        pitch_smooth.update(raw_pitch)

    if right_hand:
        dist_r = get_distance(right_hand.landmark[4], right_hand.landmark[8])
        raw_speed = scale(dist_r, 0.01, 0.25, 0.5, 2.0)
        speed_smooth.update(raw_speed)

    if left_hand and right_hand:
        center_l = np.array([left_hand.landmark[0].x, left_hand.landmark[0].y])
        center_r = np.array([right_hand.landmark[0].x, right_hand.landmark[0].y])
        hand_dist = np.linalg.norm(center_l - center_r)
        raw_volume = scale(hand_dist, 0.1, 0.6, 0.0, 2.0)
        volume_smooth.update(raw_volume)

    cv2.putText(img, f'Speed: {speed_smooth.value:.2f}x', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    cv2.putText(img, f'Pitch: {pitch_smooth.value:+.1f} semitones', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(img, f'Volume: {volume_smooth.value:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Air DJ 3.0", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stream.stop()
stream.close()
