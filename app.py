import os
import uuid
import cv2 as cv
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify, send_file,after_this_request
from flask_cors import CORS
from tensorflow.keras.models import load_model
from scipy.spatial import distance as dist
import mediapipe as mp
import fer
from collections import Counter
import pandas as pd
import time
app = Flask(__name__)
CORS(app)
def Del(path):
    time.sleep(5)
    os.remove(path)
def Del(path):
    time.sleep(5)
    os.remove(path)
cwd=os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(cwd,"TempReLie")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = load_model(os.path.join(cwd,"ReLie_model.h5"))
scaler = joblib.load(os.path.join(cwd,"scaler.pkl"))
le1 = joblib.load(os.path.join(cwd,"le_emotion.pkl"))

fer_detector = fer.FER()
haar = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
facemesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

def extract_features(landmarks, shape, frame):
    def px(idx): return (int(landmarks[idx].x * shape[1]), int(landmarks[idx].y * shape[0]))
    def eye_ratio(t, b, l, r): return dist.euclidean(t, b) / dist.euclidean(l, r)

    face = frame
    faces = haar.detectMultiScale(frame, 1.1, 4)
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        break

    r_eye = eye_ratio(px(159), px(145), px(33), px(133))
    l_eye = eye_ratio(px(386), px(374), px(263), px(362))
    blink = int(((r_eye + l_eye) / 2) < 0.20)
    lip_gap = dist.euclidean(px(13), px(14))
    brow_gap = dist.euclidean(px(105), px(66))

    emotions = fer_detector.detect_emotions(face)
    if emotions and "emotions" in emotions[0]:
        emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
    else:
        emotion = "neutral"

    return emotion, blink, lip_gap, brow_gap

def summarize_window(features, emotion):
    features = np.array(features)
    return {
        "blink_sum": np.sum(features[:, 0]),
        "blink_mean": np.mean(features[:, 0]),
        "lipgap_mean": np.mean(features[:, 1]),
        "lipgap_max": np.max(features[:, 1]),
        "lipgap_std": np.std(features[:, 1]),
        "browgap_mean": np.mean(features[:, 2]),
        "browgap_std": np.std(features[:, 2]),
        "emotion": emotion
    }

@app.route("/VideoUpload", methods=["POST"])
def video_upload():
    video_file = request.files['video']
    input_path = os.path.join(OUTPUT_DIR, f"input_{uuid.uuid4().hex}.mp4")
    video_file.save(input_path)

    cap = cv.VideoCapture(input_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    temp_raw_output_path = os.path.join(OUTPUT_DIR, "temp_raw_output.mp4")
    out = cv.VideoWriter(temp_raw_output_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frames, feature_window, emotion_window = [], [], []
    prediction_labels, prediction_scores = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = facemesh.process(rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            emotion, blink, lipgap, browgap = extract_features(landmarks, frame.shape, frame)
            feature_window.append([blink, lipgap, browgap])
            emotion_window.append(emotion)
        else:
            feature_window.append([0, 0, 0])
            emotion_window.append("neutral")

        frames.append(frame)

        if len(feature_window) == 30:
            majority_emotion = Counter(emotion_window).most_common(1)[0][0]
            summary = summarize_window(feature_window, majority_emotion)
            df = pd.DataFrame([summary])
            df['emotion'] = le1.transform(df['emotion'])
            scaled = scaler.transform(df)
            prediction = model.predict(scaled)[0][0]
            label = "True" if prediction > 0.5 else "Lie"
            surety = round(prediction * 100 if label == "True" else (1 - prediction) * 100, 2)
            prediction_labels.extend([label] * 30)
            prediction_scores.extend([surety] * 30)
            feature_window.clear()
            emotion_window.clear()

    if feature_window:
        majority_emotion = Counter(emotion_window).most_common(1)[0][0]
        summary = summarize_window(feature_window, majority_emotion)
        df = pd.DataFrame([summary])
        df['emotion'] = le1.transform(df['emotion'])
        scaled = scaler.transform(df)
        prediction = model.predict(scaled)[0][0]
        label = "True" if prediction > 0.5 else "Lie"
        surety = round(prediction * 100 if label == "True" else (1 - prediction) * 100, 2)
        prediction_labels.extend([label] * len(feature_window))
        prediction_scores.extend([surety] * len(feature_window))

    for i, frame in enumerate(frames):
        if i < len(prediction_labels):
            label = prediction_labels[i]
            surety = prediction_scores[i]
            color = (0, 0, 255) if label == "Lie" else (0, 255, 0)
            text = f"{label} ({surety:.1f}%)"
            cv.putText(frame, text, (0, 25), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=2)
        out.write(frame)

    cap.release()
    out.release()

    final_output_path = os.path.join(OUTPUT_DIR, f"output_{uuid.uuid4().hex}.mp4")
    os.system(f"ffmpeg -y -i \"{temp_raw_output_path}\" -vcodec libx264 -preset ultrafast \"{final_output_path}\"")

    if not os.path.exists(final_output_path):
        return jsonify({"error": "Video processing failed"}), 500
    @after_this_request
    def idk(response):
        delt=threading.Thread(target=Del, args=(final_output_path,))
        delt.start()
        return response
    return send_file(final_output_path, mimetype="video/mp4", as_attachment=False)
@app.route("/")
def home():
    return "ReLie backend is live."

    
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from collections import Counter
import pandas as pd

camera_window = []
emotion_window = []
frame_counter = 0

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

@app.route('/CameraUpload', methods=['POST'])
def camera_upload():
    global camera_window, emotion_window, frame_counter

    data = request.json
    img_data = base64.b64decode(data['frame'])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = facemesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            emotion, *features = extract_features(landmarks.landmark, frame.shape[:2], frame)
            camera_window.append(features)
            emotion_window.append(emotion)
            frame_counter += 1

            break

    prediction = None
    surety = None

    if frame_counter == 30:
        summarized = summarize_window(camera_window, Counter(emotion_window).most_common(1)[0][0])
        summarized['emotion'] = le1.transform([summarized['emotion']])[0]
        df = pd.DataFrame([summarized])
        df_scaled = scaler.transform(df)

        pred = model.predict(df_scaled)
        prediction = int((pred > 0.5).astype(int)[0][0])
        surety = float(pred[0][0])
        label = "Lie" if prediction == 0 else "True"
        color = (0, 0, 255) if prediction == 0 else (0, 255, 0)
        cv2.putText(frame, f'{label} ({surety:.2f})', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        camera_window.clear()
        emotion_window.clear()
        frame_counter = 0

    frame_base64 = encode_image_to_base64(frame)
    return jsonify({
        'frame': frame_base64
    })

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import threading
import socket
import queue

@app.route("/PortScan",methods=['POST'])
def Portscan():
    q=queue.Queue()
    data=request.get_json()
    print(data)
    target=data.get("target","")
    timeout=data.get("timeout",1.0)
    ports=data.get("ports","").strip()
    openports=[]
    closedports=[]
    try:
        spltc=ports.split(",")
        for no in spltc:
            if "-" in no:
                splth=no.split("-")
                for i in range(int(splth[0]),int(splth[-1])+1):
                    q.put(i)
            else:
                no=int(no)
                q.put(no)
    except:
        return jsonify({"Error":"The Port Is Invalid"}),400
    def worker():
        while not q.empty():
            port=q.get()
            try:
                print(f"Scanning {target}:{port}")
                sc=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                sc.settimeout(timeout)
                r=sc.connect_ex((target,port))
                if r==0:
                    openports.append(port)
                else:
                    closedports.append(port)
                sc.close()
            except:
                pass
    threadl=[]
    for _ in range(min(100,q.qsize())):
        thread=threading.Thread(target=worker)
        thread.start()
        threadl.append(thread)
    for threadt in threadl:
        threadt.join()
    print(openports)
    return jsonify({
                    "Open Ports": (openports),
                    "Closed Ports" : (closedports)
                    })
@app.route("/PortScanIPv6",methods=['POST'])
def PortscanIPv6():
    q=queue.Queue()
    data=request.get_json()
    print(data)
    target=data.get("target","")
    timeout=data.get("timeout",1.0)
    ports=data.get("ports","").strip()
    openports=[]
    closedports=[]
    try:
        spltc=ports.split(",")
        for no in spltc:
            if "-" in no:
                splth=no.split("-")
                for i in range(int(splth[0]),int(splth[-1])+1):
                    q.put(i)
            else:
                no=int(no)
                q.put(no)
    except:
        return jsonify({"Error":"The Port Is Invalid"}),400
    def worker():
        while not q.empty():
            port=q.get()
            try:
                print(f"Scanning {target}:{port}")
                sc=socket.socket(socket.AF_INET6,socket.SOCK_STREAM)
                addr_info = socket.getaddrinfo(target, port, socket.AF_INET6)
                ip = addr_info[0][4][0]
                sc.settimeout(timeout)
                r=sc.connect_ex((ip,port,0,0))
                if r==0:
                    openports.append(port)
                else:
                    closedports.append(port)
                sc.close()
            except:
                pass
    threadl=[]
    for _ in range(min(100,q.qsize())):
        thread=threading.Thread(target=worker)
        thread.start()
        threadl.append(thread)
    for threadt in threadl:
        threadt.join()
    print(openports)
    return jsonify({
                    "Open Ports": (openports),
                    "Closed Ports" : (closedports)
                    })
if __name__ == "__main__":
    app.run(debug=True)