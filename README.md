# Realtime Fall Detection System 🚨

A robust, computer-vision-based system designed to detect falls in real-time using specific pose estimation landmarks. The system uses a Client-Server architecture where the **Client** (Raspberry Pi or Laptop) monitors the video feed and the **Server** centrally manages alerts, evidence storage, and notifications.

## 📖 Table of Contents
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Requirements](#-requirements)
- [Installation & Setup](#-installation--setup)
- [Configuration (.env)](#-configuration)
- [How to Run](#-how-to-run)
- [Training Your Own Model](#-training-your-own-model-research-guide)
- [System Logic (FSM)](#-system-logic-finite-state-machine)
- [Troubleshooting](#-troubleshooting)

---

## 🏗 System Architecture

The project consists of two main components:

1.  **The Client (`client.py`)**:
    *   **Input**: Captures video from **Raspberry Pi Camera V2** (priority) or **Laptop Webcam** (fallback).
    *   **Processing**: Uses **MediaPipe Pose** to extract 33 skeletal landmarks.
    *   **Logic**: Runs a Finite State Machine (FSM) to distinguish between normal movement, falling, and lying inactive.
    *   **Model**: Uses a pre-trained **Random Forest Classifier** (`pose_model2.pkl`) to classify current poses.
    *   **Output**: Sends HTTP POST requests to the server with fall probability and evidence images.

2.  **The Server (`server.py`)**:
    *   **Technology**: Python Flask.
    *   **Function**: Receives alerts, saves snapshot images to `evidence_image/`, and maintains an alert history.
    *   **Notification**: Automatically sends email alerts via SMTP (Gmail) when a critical fall is confirmed.
    *   **Dashboard**: A local web interface (`http://localhost:5000`) to view status and history.

---

## ✨ Key Features
*   **Hybrid Camera Support**: detailed auto-detection logic that prefers PiCamera2 but seamlessly falls back to OpenCV Webcam.
*   **Dual-Stage Verification**: Prevents false alarms by checking for *Fall Action* → *Inactivity* before alerting.
*   **Real-time Dashboard**: Web interface to monitor alerts and view evidence photos.
*   **Email Notifications**: Instant alerts sent to configured email addresses with details and a snapshot.
*   **Data Integrity**: Auto-generated dataset tools (`make_data.py`) to streamline model retraining.

---

## 📦 Requirements

*   **Operating System**: Windows / Linux / Raspberry Pi OS
*   **Python**: 3.8+
*   **Key Libraries**:
    *   `opencv-python` (Video processing)
    *   `mediapipe` (Pose estimation)
    *   `scikit-learn` (Machine Learning model)
    *   `flask` (Web server)
    *   `pandas` & `numpy` (Data processing)
    *   `picamera2` (Only for Raspberry Pi)

---

## 🚀 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/NguyenTienKhoa2004/Realtime_Fall_Detection_System.git
    cd Realtime_Fall_Detection_System
    ```

2.  **Create a Virtual Environment (Optional but Recommended)**
    ```bash
    python -m venv env
    # Windows
    .\env\Scripts\activate
    # Linux/Mac
    source env/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚙ Configuration

### 1. Email Alerts Setup
Security is handled via environment variables.
1.  Copy the example file:
    ```bash
    cp .env.example .env
    ```
2.  Open `.env` and fill in your details:
    ```env
    EMAIL_SENDER=your_bot_email@gmail.com
    EMAIL_PASSWORD=your_app_specific_password
    EMAIL_RECEIVER=your_personal_email@gmail.com
    ```
    *Note: For Gmail, use an **App Password** (Manage Google Account > Security > 2-Step Verification > App Passwords).*

### 2. Tuning Detection Logic (`client.py`)
You can adjust these constants at the top of `client.py` to fit your specific scenario:
*   `FALL_CONFIRM_SECONDS = 1.0`: How long a "fall" pose must be detected to count as a fall.
*   `INACTIVITY_SECONDS = 1.0`: How long the person must be "still" after falling to trigger an alert.
*   `RESET_COOLDOWN_SECONDS = 10.0`: Time before the system resets after an alert.
*   `MOTION_THRESHOLD = 0.05`: Sensitivity for detecting movement (Inactivity check).

---

## ▶ How to Run

### Step 1: Start the Server
The server must be running first to receive alerts.
```bash
python server.py
```
*   **Access Dashboard**: Open `http://127.0.0.1:5000` in your browser.

### Step 2: Start the Client
Run this on the device with the camera (same machine or different one).
```bash
python client.py
```
*   It will attempt to load `picamera2`. If unavailable, it defaults to the webcam.
*   Access the live detection feed at `http://127.0.0.1:5001`.

---

## 🧠 Training Your Own Model (Research Guide)

If you want to train the system on your own dataset (e.g., adding "Yoga" or "Lying on Sofa" classes to reduce false positives), follow this workflow:

### 1. Prepare Data Folder Structure
Organize your images into folders by class name:
```
dataset_da_chia/
├── train/
│   ├── fall/       # Images of people falling/fallen
│   ├── normal/     # Images of people standing/walking
│   └── sitting/    # Images of people sitting
└── test/
    ├── fall/
    └── ...
```

### 2. Extract Features (`make_data.py`)
This script uses MediaPipe to extract (x, y, z, visibility) for 33 landmarks from every image and saves them to CSV.
```bash
python make_data.py
```
*   **Output**: Generates `train.csv` and `test.csv`.

### 3. Train the Classifier (`train_model.py`)
Trains a Random Forest model on the CSV data.
```bash
python train_model.py
```
*   **Logic**: 
    *   Loads `train.csv` and `test.csv`.
    *   Trains a Random Forest Classifier (`n_estimators=100`, `class_weight='balanced'`).
    *   Evaluates accuracy and Confusion Matrix.
    *   **Output**: Saves the trained model to `pose_model2.pkl`.

### 4. Update Client
Ensure `client.py` is loading your new model:
```python
# client.py
with open('pose_model2.pkl', 'rb') as f:
    model = pickle.load(f)
```

---

## 🔄 System Logic (Finite State Machine)

The system does **not** alert immediately upon detecting a fall. It follows a human-verified logic flow:

1.  **NORMAL State**:
    *   System monitors pose.
    *   If Model predicts "Fall", switch to **FALLING**.

2.  **FALLING State** (Verification):
    *   Wait for `FALL_CONFIRM_SECONDS`.
    *   If the "Fall" pose persists for this duration, we confirm it's not a glitch.
    *   Switch to **INACTIVE**.

3.  **INACTIVE State** (Critical Check):
    *   Monitor `motion_val` (movement intensity).
    *   **If Moving**: Person is getting up. -> **Reset to NORMAL**.
    *   **If Still** (for `INACTIVITY_SECONDS`): Person is unconscious or hurt. -> **🔴 TRIGGER ALERT**.

4.  **ALERT TRIGGERED**:
    *   Send POST request to Server.
    *   Server saves image and emails user.
    *   Client waits `RESET_COOLDOWN_SECONDS` before restarting monitoring.

---

## 🛠 Troubleshooting

**1. "Picamera2 not available. Se su dung webcam"**
   *   This is normal behavior on Windows/Mac or non-Pi devices. The system successfully switched to your USB webcam.

**2. Alert sent but no Email received**
   *   Check Server logs.
   *   Ensure `.env` file exists and has correct credentials.
   *   Verify `server.py` allows your label: `if str(label).lower() in ["fall", "1", "confirmed", "inactive", "fall_confirmed"]:`.

**3. False Alarms (Detecting Fall when Sitting)**
   *   Retrain the model. Add more "sitting" images to your dataset.
   *   Adjust `FALL_CONFIRM_SECONDS` higher (e.g., 2.0s) in `client.py`.

---
*Created by [Your Name/Group]*
