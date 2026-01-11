import time
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import pickle
import requests
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("! Picamera2 khong kha dung. Se su dung webcam.")
from flask import Flask, Response
from enum import Enum
import json
from collections import deque

# ==================================================
# 1. CẤU HÌNH & KHỞI TẠO
# ==================================================
SERVER_URL = 'http://127.0.0.1:5000/alert'

FALL_CONFIRM_SECONDS = 4.0    
INACTIVITY_SECONDS = 3.0      
RESET_COOLDOWN_SECONDS = 10.0
MOTION_THRESHOLD = 0.05      

LANDMARK_BUFFER = deque(maxlen=10)

app = Flask(__name__)

class FallState(Enum):
    NORMAL = 0
    FALLING = 1
    CONFIRMED = 2
    INACTIVE = 3

current_state = FallState.NORMAL
state_start_time = None
alert_sent = False
last_alert_time = 0 

# Load Model
print("--> Dang load model...")
try:
    with open('pose_model2.pkl', 'rb') as f:
        model = pickle.load(f)
    print("--> Load model thanh cong!")
except FileNotFoundError:
    print("LOI: Khong tim thay file 'pose_model.pkl'")
    exit()

# Mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)

# ==================================================
# 2. CAMERA INITIALIZATION (PI CAM V2 OR WEBCAM)
# ==================================================
print("---> Khoi dong Camera...")
USE_PICAM = False
picam2 = None
cap = None

if PICAMERA2_AVAILABLE:
    try:
        picam2 = Picamera2()
        config = picam2.create_video_configuration(main={"size": (320, 240), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
        USE_PICAM = True
        print("---> Dang su dung Raspberry Pi Camera V2")
    except Exception as e:
        print(f"! Loi khoi dong Pi Camera V2: {e}")
        print("---> Su dung webcam...")

if not USE_PICAM:
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Khong the mo webcam")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        print("---> Dang su dung Laptop Webcam")
    except Exception as e:
        print(f"LOI: Khong the khoi dong camera: {e}")
        exit()

def calculate_motion(buffer):
    if len(buffer) < 2: return 1.0 
    curr = np.array(buffer[-1])
    prev = np.array(buffer[-2])
    diff = np.abs(curr - prev)
    return np.mean(diff) 

# ==================================================
# 3. VÒNG LẶP XỬ LÝ CHÍNH
# ==================================================
def generate_frames():
    global current_state, state_start_time, alert_sent, last_alert_time
    
    p_time = 0
    miss_counter = 0
    
    while True:
        try:
            if USE_PICAM:
                frame = picam2.capture_array()
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Loi: Khong doc duoc frame tu webcam")
                    break
                frame = cv2.resize(frame, (320, 240))
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            motion_val = 1.0
            prob = 0.0
            label_text = "NORMAL"
            box_color = (0, 255, 0) 
            is_fall_pose = False
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                landmarks = results.pose_landmarks.landmark
                pose_row = []
                for lm in landmarks:
                    pose_row.extend([lm.x, lm.y, lm.z, lm.visibility])
                
                LANDMARK_BUFFER.append(pose_row)
                motion_val = calculate_motion(LANDMARK_BUFFER)
                
                X = pd.DataFrame([pose_row])
                try:
                    current_class = model.predict(X)[0] 
                    probs = model.predict_proba(X)[0]
                    prob = max(probs)
                except:
                    current_class = "Normal"
                    prob = 0.0

                is_fall_pose = str(current_class).lower() in ["fall", "1"]
                now = time.time()

                # --- LOGIC FSM ---
                
                # 1. NORMAL
                if current_state == FallState.NORMAL:
                    if is_fall_pose:
                        print("-> Phat hien dang nga! Bat dau dem gio...")
                        current_state = FallState.FALLING
                        state_start_time = now
                        miss_counter = 0
                
                # 2. FALLING
                elif current_state == FallState.FALLING:
                    elapsed_time = now - state_start_time
                    if is_fall_pose:
                        miss_counter = 0
                        if elapsed_time >= FALL_CONFIRM_SECONDS:
                            print(f"-> Xac nhan nga. Check BAT DONG.")
                            current_state = FallState.INACTIVE
                            state_start_time = now
                    else:
                        miss_counter += 1
                        if miss_counter > 100:
                            current_state = FallState.NORMAL
                            miss_counter = 0

                # 3. INACTIVE 
                elif current_state == FallState.INACTIVE:
                    box_color = (0, 165, 255)
                    
                    if not alert_sent:
                        label_text = f"CHECK MOTION: {motion_val:.4f}"
                        if motion_val > MOTION_THRESHOLD:
                            print("-> Cu dong manh. Reset ve NORMAL.")
                            current_state = FallState.NORMAL
                        else:
                            if now - state_start_time >= INACTIVITY_SECONDS:
                                print("!!! GUI CANH BAO !!!")
                                try:
                                    _, img_encoded = cv2.imencode('.jpg', frame)
                                    files = {'image': ('alert.jpg', img_encoded.tobytes(), 'image/jpeg')}
                                    camera_id = 'pi-v2' if USE_PICAM else 'webcam'
                                    meta_data = {'camera_id': camera_id, 'motion': float(motion_val)}
                                    data = {'label': 'FALL_CONFIRMED', 'prob': str(prob), 'meta': json.dumps(meta_data)}
                                    
                                    requests.post(SERVER_URL, data=data, files=files, timeout=2)
                                    
                                    alert_sent = True
                                    last_alert_time = now 
                                    
                                except Exception as e:
                                    print(f"Loi gui: {e}")

                    else:
                        time_since_alert = now - last_alert_time
                        countdown = RESET_COOLDOWN_SECONDS - time_since_alert
                        
                        label_text = f"ALERT SENT! RESET IN {int(countdown)}s"
                        box_color = (0, 0, 255) # Đỏ

                        # Logic Auto-Reset:
                        # 1. Nếu hết thời gian Cooldown -> Reset
                        if time_since_alert >= RESET_COOLDOWN_SECONDS:
                            print("-> Auto-Reset sau bao dong.")
                            current_state = FallState.NORMAL
                            alert_sent = False
                        
                        # 2. Hoặc nếu có chuyển động mạnh (người khác đi vào) -> Reset sớm
                        elif motion_val > MOTION_THRESHOLD:
                            print("-> Phat hien chuyen dong moi. Reset ngay lap tuc.")
                            current_state = FallState.NORMAL
                            alert_sent = False

            # --- UI Display ---
            if current_state == FallState.NORMAL:
                label_text = f"NORMAL ({prob:.2f})"
            elif current_state == FallState.FALLING:
                time_left = max(0, FALL_CONFIRM_SECONDS - (time.time() - state_start_time))
                label_text = f"VERIFYING... {time_left:.1f}s"
                box_color = (0, 255, 255) 

            cv2.rectangle(frame, (0, 0), (320, 40), box_color, -1)
            cv2.putText(frame, label_text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            c_time = time.time()
            fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
            p_time = c_time
            cv2.putText(frame, f"FPS: {int(fps)}", (250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                   
        except Exception as e:
            print(f"Error: {e}")
            break

@app.route('/')
def index():
    return "<h1>Auto-Reset FSM System</h1><img src='/video_feed' width='640' />"
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)