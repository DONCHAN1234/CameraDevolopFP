# -------------------------- Imports --------------------------
import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from twilio.rest import Client
import pyttsx3
import sqlite3
import requests
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import time
import os

# -------------------------- Page Config --------------------------
st.set_page_config(page_title="Fire Detection System", layout="centered")

# -------------------------- Twilio Setup --------------------------
account_sid = 'ACe115119f2129f817eb767997e304ffc0'
auth_token = '1cf92e18c8b977d2f71581691e542181'
twilio_number = '+13022484056'
recipients = ['+917363064067', '+917811050018', '+918670516762', '+917047283086']

def send_alert_sms(message):
    try:
        client = Client(account_sid, auth_token)
        for number in recipients:
            client.messages.create(body=message, from_=twilio_number, to=number)
    except Exception as e:
        st.warning(f"SMS Error: {e}")

def text_to_speech(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        st.warning(f"TTS Error: {e}")

# -------------------------- SQLite Setup --------------------------
DB_NAME = "temp_db_converted.sqlite"

def connect_db():
    return sqlite3.connect(DB_NAME)

def create_tables():
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS smoke_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                smoke INTEGER,
                temperature REAL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                smoke REAL,
                temperature REAL,
                fire REAL
            )
        """)
        conn.commit()

def insert_temp_data(smoke, temperature):
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO smoke_data (smoke, temperature) VALUES (?, ?)", (smoke, temperature))
        cursor.execute("DELETE FROM smoke_data WHERE id NOT IN (SELECT id FROM smoke_data ORDER BY id DESC LIMIT 10)")
        conn.commit()

def insert_fire_data(smoke, temperature, fire):
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sensor_data (smoke, temperature, fire) VALUES (?, ?, ?)", (smoke, temperature, fire))
        conn.commit()

def display_data(table_name):
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY id DESC")
        data = cursor.fetchall()
        columns = [desc[0].capitalize() for desc in cursor.description]
        st.dataframe(pd.DataFrame(data, columns=columns))

# -------------------------- Load Model --------------------------
@st.cache_resource
def load_model():
    if os.path.exists("model.h5"):
        return tf.keras.models.load_model("model.h5")
    else:
        st.error("🔥 Model file not found: model.h5")
        return None

model = load_model()

def predict_image(img_path):
    if model is None:
        return 0
    try:
        target_size = model.input_shape[1:3]
        img = image.load_img(img_path, target_size=target_size)
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        prediction = model.predict(x)[0][0]
        return float(prediction)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0

# -------------------------- Sensor API --------------------------
flask_url = "https://aurdinodatatransferingbackend-4.onrender.com/get_data"

def get_sensor_data():
    try:
        response = requests.get(flask_url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.warning(f"Sensor API Error: {e}")
    return {"smoke": 0, "temp": 0}

# -------------------------- Fire Detection Logic --------------------------
class FireDetector(VideoTransformerBase):
    def __init__(self):
        self.count = 0
        self.last_time = time.time()

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()

        if current_time - self.last_time < 0.5:
            return img

        self.last_time = current_time

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (18, 50, 50), (35, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        fire_detected = False

        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                fire_detected = True
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, "🔥", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if fire_detected:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                cv2.imwrite(tmp.name, img)
                camera_pred = predict_image(tmp.name)

            sensor = get_sensor_data()
            smoke = sensor.get("smoke", 0)
            temp = sensor.get("temp", 0)

            if camera_pred > 0.2 and smoke >= 250 and temp >= 13:
                self.count += 1
                st.session_state.fire_result = "🔥 FIRE DETECTED!"
                if self.count >= 3:
                    insert_fire_data(smoke, temp, 1)
                    text_to_speech("Fire detected on the 1st floor! Please evacuate immediately.")
                    send_alert_sms("🔥 Fire detected on the 1st floor! Please evacuate immediately using the nearest exit.")
                    self.count = 0
            else:
                st.session_state.fire_result = "✅ No Fire Detected"
                insert_temp_data(smoke, temp)
        else:
            st.session_state.fire_result = "✅ No Fire Detected"

        return img

# -------------------------- UI Setup --------------------------
st.title("🔥 Fire Detection System")
create_tables()

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

if "fire_result" not in st.session_state:
    st.session_state.fire_result = "Waiting for camera input..."

st.sidebar.title("Controls")

if st.sidebar.button("Start Camera"):
    st.session_state.camera_on = True

if st.sidebar.button("Stop Camera"):
    st.session_state.camera_on = False

if st.session_state.camera_on:
    st.info("📷 Allow camera access to start fire detection.")
    ctx = webrtc_streamer(
        key="fire-detect",
        video_processor_factory=FireDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

    st.subheader("Live Fire Detection Result")
    st.markdown(f"### {st.session_state.fire_result}")
else:
    st.warning("Camera is OFF. Click 'Start Camera' to activate live fire detection.")

# -------------------------- Sidebar Features --------------------------
if st.sidebar.button("Show Sensor Data"):
    st.markdown("### 🌡️ Smoke + Temperature Readings")
    display_data("smoke_data")

if st.sidebar.button("Show Fire Events"):
    st.markdown("### 🔥 Confirmed Fire Events")
    display_data("sensor_data")

if st.sidebar.button("About Components"):
    st.markdown("### ⚙️ Hardware Used")
    st.write("- Arduino UNO")
    st.write("- MQ-2 Smoke Sensor")
    st.write("- Temperature Sensor")
    st.write("- Camera + CNN Fire Model")
