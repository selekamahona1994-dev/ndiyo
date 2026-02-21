import streamlit as st
import cv2
from ultralytics import YOLO
import face_recognition
import numpy as np
import math
import time
import sqlite3
import os
import pandas as pd
import requests
from io import BytesIO


# --- 1. DATABASE LOGIC ---
def init_db():
    conn = sqlite3.connect('security_suite.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS incidents
                 (timestamp TEXT, names TEXT, violations INTEGER)''')
    conn.commit()
    conn.close()


def log_incident(names, count):
    conn = sqlite3.connect('security_suite.db')
    c = conn.cursor()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO incidents VALUES (?, ?, ?)", (timestamp, names, count))
    conn.commit()
    conn.close()


# --- 2. TELEGRAM LOGIC ---
def send_telegram(df, message):
    # Enter your credentials here
    TOKEN = "YOUR_BOT_TOKEN"
    CHAT_ID = "YOUR_CHAT_ID"
    try:
        # Send Text
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={'chat_id': CHAT_ID, 'text': message})
        # Send CSV Log
        csv_buf = BytesIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendDocument",
                      data={'chat_id': CHAT_ID}, files={'document': ('incident_log.csv', csv_buf)})
    except:
        pass


# --- 3. FACE RECOGNITION SETUP ---
KNOWN_DIR = "known_faces"
if not os.path.exists(KNOWN_DIR): os.makedirs(KNOWN_DIR)


@st.cache_resource
def get_known_faces():
    encodings, names = [], []
    for file in os.listdir(KNOWN_DIR):
        if file.endswith((".jpg", ".png", ".jpeg")):
            img = face_recognition.load_image_file(f"{KNOWN_DIR}/{file}")
            enc = face_recognition.face_encodings(img)
            if enc:
                encodings.append(enc[0])
                names.append(os.path.splitext(file)[0])
    return encodings, names


# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="AI Security Suite", layout="wide")
init_db()

st.sidebar.title("🛡️ System Control")
mode = st.sidebar.radio("Navigation", ["Live Monitor", "History/Search", "Face Enrollment"])
threshold = st.sidebar.slider("Proximity Threshold (px)", 50, 400, 150)
enable_blur = st.sidebar.checkbox("Privacy Blur", value=True)

# Global session states
if 'violation_timer' not in st.session_state: st.session_state.violation_timer = None
if 'last_alert' not in st.session_state: st.session_state.last_alert = 0

# --- MODE: FACE ENROLLMENT ---
if mode == "Face Enrollment":
    st.header("👤 New User Enrollment")
    name = st.text_input("Name of Person")
    photo = st.camera_input("Take Enrollment Photo")
    if photo and name:
        if st.button("Register Face"):
            with open(f"{KNOWN_DIR}/{name.lower()}.jpg", "wb") as f:
                f.write(photo.getbuffer())
            st.success(f"Registered {name}!")
            st.cache_resource.clear()

# --- MODE: HISTORY ---
elif mode == "History/Search":
    st.header("🔍 Incident Database")
    search = st.text_input("Search by Name")
    if st.button("Run Query"):
        conn = sqlite3.connect('security_suite.db')
        df = pd.read_sql_query(f"SELECT * FROM incidents WHERE names LIKE '%{search}%'", conn)
        st.dataframe(df, use_container_width=True)
        conn.close()

# --- MODE: LIVE MONITOR ---
elif mode == "Live Monitor":
    st.header("🎥 Live Security Feed")
    run = st.checkbox("Start AI Engine")
    frame_window = st.image([])

    model = YOLO('yolov8n.pt')
    known_enc, known_names = get_known_faces()

    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret: break

            # 1. Detection & Face ID
            results = model(frame, classes=[0], conf=0.5, verbose=False)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb_frame)
            face_encs = face_recognition.face_encodings(rgb_frame, face_locs)

            frame_ids = []
            for enc in face_encs:
                matches = face_recognition.compare_faces(known_enc, enc)
                name = "Unknown"
                if True in matches:
                    name = known_names[matches.index(True)]
                frame_ids.append(name)

            # 2. Proximity Calculation
            centers = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Apply Privacy Blur
                    if enable_blur:
                        h_h = int((y2 - y1) * 0.25)
                        frame[y1:y1 + h_h, x1:x2] = cv2.GaussianBlur(frame[y1:y1 + h_h, x1:x2], (51, 51), 0)
                    centers.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            violations = 0
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    if math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1]) < threshold:
                        violations += 1
                        cv2.line(frame, centers[i], centers[j], (0, 0, 255), 3)

            # 3. Alert & DB Logging
            if violations > 0:
                now = time.time()
                if st.session_state.violation_timer is None: st.session_state.violation_timer = now

                # If sustained for 5 seconds
                if now - st.session_state.violation_timer > 5:
                    who = ", ".join(set(frame_ids)) if frame_ids else "Unknowns"
                    log_incident(who, violations)
                    # Telegram cooldown (2 mins)
                    if now - st.session_state.last_alert > 120:
                        send_telegram(pd.DataFrame([{"Time": time.ctime(), "Who": who}]),
                                      f"🚨 Alert: {who} violating proximity!")
                        st.session_state.last_alert = now
            else:
                st.session_state.violation_timer = None

            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()