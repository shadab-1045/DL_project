import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tempfile
import os
import time
import random
from collections import deque
from tensorflow.keras.models import load_model

# ================= CONFIGURATION =================
MODEL_PATH = "models/faceguard_phase2_finetuned.h5"
IMG_SIZE = (224, 224)
FRAME_INTERVAL = 30
FAKE_THRESHOLD = 0.40

# ================= PAGE SETUP =================
st.set_page_config(
    page_title="FaceGuard System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 🎨 SCI-FI THEME & CRT SCANLINES =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600&family=Roboto+Mono&display=swap');

.stApp {
    background: radial-gradient(circle at top, #2b004d, #050012 90%);
    color: #00ffff;
    font-family: 'Roboto Mono', monospace;
}

[data-testid="stSidebar"] {
    background-color: #0d001a !important;
    border-right: 1px solid rgba(0, 255, 255, 0.1);
    box-shadow: 2px 0 15px rgba(0, 255, 255, 0.15);
}

h2, h3, h4, h5 {
    font-family: 'Orbitron', sans-serif;
    color: #ff00ff;
    text-shadow: 0 0 10px rgba(255, 0, 255, 0.7);
}

.stButton>button {
    width: 100%;
    border-radius: 5px;
    background: transparent;
    border: 2px solid #00ffff;
    font-family: 'Orbitron', sans-serif;
    font-weight: bold;
    color: #00ffff;
    transition: 0.3s;
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.2);
}
.stButton>button:hover {
    box-shadow: 0 0 20px #ff00ff;
    border: 2px solid #ff00ff;
    color: #ff00ff;
    transform: scale(1.02);
}

.scanlines {
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), 
                linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
    background-size: 100% 4px, 6px 100%;
    z-index: 9999;
    pointer-events: none;
    opacity: 0.4;
    animation: flicker 0.15s infinite;
}
@keyframes flicker {
    0% { opacity: 0.3; }
    50% { opacity: 0.4; }
    100% { opacity: 0.3; }
}

.info-box {
    background: rgba(0, 255, 255, 0.05);
    border-left: 3px solid #00ffff;
    padding: 10px;
    margin-top: 15px;
    font-size: 0.9em;
    color: #a0ffff;
}
</style>
<div class="scanlines"></div>
""", unsafe_allow_html=True)

# ================= 🧠 MEDIAPIPE SETUP =================
VISUALS_ENABLED = False
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    sci_fi_connections_style = mp_drawing.DrawingSpec(color=(0, 255, 200), thickness=1, circle_radius=0) 
    sci_fi_landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 200), thickness=0, circle_radius=0) 
    
    VISUALS_ENABLED = True
except Exception as e:
    print(f"MediaPipe Warning: {e}")
    VISUALS_ENABLED = False

# ================= HELPER FUNCTIONS =================

def apply_glitch(image, score=None):  # Added score=None to fix TypeError
    """Adds a red transparent overlay and a small neon alert in the bottom corner."""
    red_layer = np.full_like(image, (255, 0, 0)) 
    overlay = cv2.addWeighted(image, 0.75, red_layer, 0.25, 0)
    
    glitched = np.copy(overlay)
    shift = 8
    glitched[:, shift:, 0] = overlay[:, :-shift, 0] 
    glitched[:, :-shift, 2] = overlay[:, shift:, 2] 
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    if score is not None:
        text = f"!!! MANIPULATION ALERT ({score*100:.0f}%) !!!"
    else:
        text = "!!! MANIPULATION ALERT !!!"
    text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
    
    x_pos = image.shape[1] - text_size[0] - 20
    y_pos = image.shape[0] - 20
    
    cv2.putText(glitched, text, (x_pos+1, y_pos+1), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(glitched, text, (x_pos, y_pos), font, 0.6, (255, 0, 100), 2, cv2.LINE_AA)
    
    return glitched

def draw_cyber_hud(image, face_landmarks, score=None, sensitivity=0.40):
    h, w, c = image.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    for lm in face_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        if x < x_min: x_min = x
        if x > x_max: x_max = x
        if y < y_min: y_min = y
        if y > y_max: y_max = y

    pad = 20
    x_min, y_min, x_max, y_max = max(0, x_min - pad), max(0, y_min - pad), min(w, x_max + pad), min(h, y_max + pad)

    if score is None:
        color, status_text = (0, 255, 255), "SCANNING..."
    elif score > sensitivity:
        color, status_text = (255, 0, 0), f"THREAT LVL: {score*100:.1f}%"
    else:
        color, status_text = (0, 255, 255), f"SECURE: {(1-score)*100:.1f}%"

    thickness, line_len = 2, 30
    cv2.line(image, (x_min, y_min), (x_min + line_len, y_min), color, thickness)
    cv2.line(image, (x_min, y_min), (x_min, y_min + line_len), color, thickness)
    cv2.line(image, (x_max, y_max), (x_max - line_len, y_max), color, thickness)
    cv2.line(image, (x_max, y_max), (x_max, y_max - line_len), color, thickness)
    
    cv2.putText(image, status_text, (x_max + 10, y_min + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image

def draw_visuals(image, score=None, sensitivity=0.40):
    if not VISUALS_ENABLED: return image
    try:
        output = image.copy()
        results = face_mesh.process(output)
        if results.multi_face_landmarks:
            for lm in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(output, lm, mp_face_mesh.FACEMESH_TESSELATION, 
                                          landmark_drawing_spec=sci_fi_landmark_style,
                                          connection_drawing_spec=sci_fi_connections_style)
                output = draw_cyber_hud(output, lm, score, sensitivity)
        return output
    except: return image

def preprocess_image(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def simulate_terminal_logging():
    terminal = st.empty()
    steps = ["Initiating Neural Protocols...", "Scanning Biometric Vectors...", 
             "Analyzing Pixel Variances...", "Isolating Anomalies...", "Compiling Forensic Verdict..."]
    for i, step in enumerate(steps):
        progress = (i + 1) * 20
        bar = "█" * (progress // 5) + "░" * (20 - (progress // 5))
        terminal.markdown(f"```shell\n> {step}\n> [{bar}] {progress}%\n```")
        time.sleep(0.3)
    terminal.empty()

# ================= MAIN APP =================
@st.cache_resource
def load_faceguard_model():
    return load_model(MODEL_PATH)

try:
    model = load_faceguard_model()
except:
    st.error("❌ Model not found!")
    st.stop()

st.sidebar.title("⚙️ CONTROL PANEL")
mode = st.sidebar.radio("SELECT MODE", ["📸 Image Analysis", "🎥 Video Forensics", "🛑 Live Webcam Scan"])
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎚️ SETTINGS")
sensitivity = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.40, 0.05)
if st.sidebar.button("🔄 RESET SYSTEM"):
    st.cache_resource.clear()
    st.rerun()

st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 5px;">
    <span style="font-size: 3rem; margin-right: 15px; text-shadow: 0 0 10px #00d4ff;">🛡️</span>
    <h1 style="margin: 0; padding: 0; font-family: 'Orbitron', sans-serif; font-size: 3.5rem; letter-spacing: 2px;">
        <span style="color: #00d4ff; text-shadow: 0 0 15px rgba(0, 212, 255, 0.8);">Face</span><span style="color: #ff00ff; text-shadow: 0 0 15px rgba(255, 0, 255, 0.8);">Guard</span> 
        <span style="color: #008b8b; font-size: 0.35em; margin-left: 10px; vertical-align: middle; text-shadow: none;">OS v3.0</span>
    </h1>
</div>
<div style="color: #00d4ff; font-family: 'Roboto Mono', monospace; margin-bottom: 30px;">> Advanced Deepfake Detection & Forensic Analysis</div>
""", unsafe_allow_html=True)

# ================= MODE: LIVE WEBCAM =================
if mode == "🛑 Live Webcam Scan":
    st.markdown("### 🔴 REAL-TIME SURVEILLANCE")
    
    with st.sidebar:
        run = st.checkbox('🔴 START CAMERA FEED')
    
    if run:
        # Split layout: Video on left, Gauge on right
        video_col, gauge_col = st.columns([2, 1])
        
        frame_placeholder = video_col.empty()
        info_placeholder = video_col.empty() # Placeholder for the stats text
        
        with gauge_col:
            st.markdown("#### 📡 LIVE RISK MONITOR")
            st.caption("Risk probability gauge. Red zone indicates active manipulation.")
            verdict_placeholder = st.empty()
            chart_placeholder = st.empty()
            
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        
        while run:
            ret, frame = cap.read()
            if not ret: break
            
            # Calculate simple FPS
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
            uptime = int(time.time() - start_time)
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred = model.predict(preprocess_image(rgb), verbose=0)[0][0]
            
            # Apply visuals and red overlay
            visual_frame = draw_visuals(rgb, pred, sensitivity)
            if pred > sensitivity: 
                visual_frame = apply_glitch(visual_frame, pred)
            
            frame_placeholder.image(visual_frame, use_container_width=True)
            
            # Draw the info box below the video
            info_placeholder.markdown(f"""
            <div class="info-box">
                <b>Stream Telemetry:</b><br>
                📡 Signal: Active | 👁️ Target Lock: 1 Face Detected<br>
                ⏱️ Uptime: {uptime}s | 🎞️ Buffer: ~{int(fps)} FPS | 🔍 Scan Res: 1080p High-Fidelity
            </div>
            """, unsafe_allow_html=True)
            
            # Verdict Text
            if pred > sensitivity:
                verdict_placeholder.error(f"🚨 THREAT DETECTED ({pred*100:.1f}%)")
            else:
                verdict_placeholder.success(f"✅ STATUS: AUTHENTIC ({(1-pred)*100:.1f}%)")
            
            # Vertical Indicator Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "white"},
                    'bar': {'color': "#ff4b4b" if pred > sensitivity else "#00f2ff"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': sensitivity * 100
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#00ffff", family="Roboto Mono"),
                height=300, margin=dict(l=20, r=20, t=20, b=20)
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
        cap.release()

# ================= MODE: VIDEO FORENSICS =================
elif mode == "🎥 Video Forensics":
    uploaded_video = st.file_uploader("UPLOAD VIDEO EVIDENCE", type=["mp4"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()
        col_orig, col_proc = st.columns(2)
        with col_orig: st.video(tfile.name)
        with col_proc: st_frame, bar = st.empty(), st.progress(0)
        
        # Fixed the duplicate button logic and indentation
        if st.button("INITIATE FORENSIC SCAN"):
            simulate_terminal_logging()
            
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            score_buffer = deque(maxlen=5)
            smoothed_predictions = []
            suspicious_frames = []
            
            frame_id = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                if frame_id % FRAME_INTERVAL == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    raw_pred = model.predict(preprocess_image(rgb), verbose=0)[0][0]
                    score_buffer.append(raw_pred)
                    smoothed_score = sum(score_buffer) / len(score_buffer)
                    smoothed_predictions.append(smoothed_score)
                    
                    clean_evidence_frame = rgb.copy()
                    vis = draw_visuals(rgb, raw_pred, sensitivity)
                    if raw_pred > sensitivity:
                        vis = apply_glitch(vis, raw_pred) # Fixed missing argument
                        suspicious_frames.append((clean_evidence_frame, raw_pred))
                    
                    st_frame.image(vis, caption=f"Processing Frame {frame_id}", use_container_width=True)
                    bar.progress(min(frame_id/total_frames, 1.0))
                    
                frame_id += 1
            cap.release()
            
            # --- START OF UPDATED VISUAL REPORT ---
            import plotly.graph_objects as go

            st.markdown("---")
            st.markdown("### 📋 FORENSIC ANALYSIS REPORT")
            
            if smoothed_predictions: # Added check to prevent mean of empty slice
                avg_score = np.mean(smoothed_predictions)
                c1, c2 = st.columns(2)
                
                if avg_score > sensitivity:
                    c1.error("🚨 VERDICT: MANIPULATION DETECTED")
                    c2.metric("THREAT LEVEL", f"{avg_score*100:.1f}%", delta="HIGH RISK", delta_color="inverse")
                else:
                    c1.success("✅ VERDICT: CONTENT IS GENUINE")
                    c2.metric("REALITY SCORE", f"{(100 - avg_score*100):.1f}%", delta="SECURE")

                st.markdown("#### 📊 TEMPORAL RISK TIMELINE")
                st.caption("Red bars indicate frames exceeding safety thresholds. The forensic line tracks digital stability.")

                # Prepare Data
                timeline_sec = [round(i * (FRAME_INTERVAL / 30), 2) for i in range(len(smoothed_predictions))]
                scores = [p * 100 for p in smoothed_predictions]
                bar_colors = ['#ff4b4b' if s > (sensitivity * 100) else '#00f2ff' for s in scores]

                # Create Plotly Figure
                fig = go.Figure()

                # 1. Add Bars
                fig.add_trace(go.Bar(
                    x=timeline_sec, y=scores,
                    marker_color=bar_colors, opacity=0.5
                ))

                # 2. Add Line with Dots
                fig.add_trace(go.Scatter(
                    x=timeline_sec, y=scores,
                    mode='lines+markers',
                    line=dict(color='#ffffff', width=2),
                    marker=dict(size=8, symbol='circle', color='#ff00ff', line=dict(color='#ffffff', width=1))
                ))

                # 3. Danger Threshold Line
                fig.add_hline(y=sensitivity*100, line_dash="dash", line_color="#ff4b4b", 
                              annotation_text="DANGER THRESHOLD")

                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#00ffff", family="Roboto Mono"),
                    height=400,
                    xaxis=dict(title="Video Timeline (Seconds)"),
                    yaxis=dict(title="Risk Level (%)", range=[0, 105]),
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)
            # --- END OF UPDATED VISUAL REPORT ---

            if suspicious_frames:
                st.markdown("#### 🚩 SUSPICIOUS ARTIFACTS")
                cols = st.columns(4) 
                for i, (f, s) in enumerate(suspicious_frames[:4]):
                    cols[i].image(f, caption=f"Risk: {s*100:.0f}%", use_container_width=True)

# ================= MODE: IMAGE ANALYSIS =================
elif mode == "📸 Image Analysis":
    uploaded = st.file_uploader("UPLOAD IMAGE EVIDENCE", type=["jpg", "png"])
    if uploaded:
        img_rgb = cv2.cvtColor(cv2.imdecode(np.asarray(bytearray(uploaded.read()), dtype=np.uint8), 1), cv2.COLOR_BGR2RGB)
        
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### 🖼️ ORIGINAL EVIDENCE")
            st.image(img_rgb, use_container_width=True)
        with col_r:
            st.markdown("#### 🤖 FORENSIC SCAN")
            scan_placeholder = st.empty()
            scan_placeholder.image(draw_visuals(img_rgb), use_container_width=True)
            
        if st.button("RUN DIAGNOSTIC"):
            simulate_terminal_logging()
            pred = model.predict(preprocess_image(img_rgb))[0][0]
            
            # Apply visuals
            final_vis = apply_glitch(draw_visuals(img_rgb, pred, sensitivity), pred) if pred > sensitivity else draw_visuals(img_rgb, pred, sensitivity)
            scan_placeholder.image(final_vis, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📋 DIAGNOSTIC RESULTS")
            
            c1, c2 = st.columns(2)
            if pred > sensitivity: 
                c1.error(f"🚨 VERDICT: MANIPULATION DETECTED")
                c1.markdown(f"**Confidence:** {pred*100:.2f}%")
                c2.metric("THREAT LEVEL", f"{pred*100:.1f}%", delta="HIGH RISK", delta_color="inverse")
            else: 
                c1.success(f"✅ VERDICT: AUTHENTIC")
                c1.markdown(f"**Confidence:** {(1-pred)*100:.2f}%")
                c2.metric("AUTHENTICITY SCORE", f"{(1-pred)*100:.1f}%", delta="SECURE")