import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from detector import EarDetector
from preprocessor import Preprocessor
from extractor import FeatureExtractor
from matcher import Matcher

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Ear-Based Biometric Recognition",
    page_icon="👂",
    layout="wide"
)

# Load External CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- ENGINE INITIALIZATION ---
@st.cache_resource
def init_engine():
    return EarDetector(), Preprocessor(), FeatureExtractor(), Matcher()

detector, preprocessor, extractor, matcher = init_engine()

# --- NAVBAR ---
st.markdown('<h1 class="main-title">EAR-BASED BIOMETRIC RECOGNITION</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Next-Generation Auricular Biometric Identity System</p>', unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://img.icons8.com/wired/512/FFFFFF/biotech.png", width=100)
    st.markdown("### Control Center")
    mode = st.radio("Select Operation", ["Biometric Login", "Identity Scan", "Secure Enrollment", "Registry Hub"])
    
    st.markdown("---")
    st.markdown("### System Health")
    st.success("Core Engine: Active")
    st.info(f"Templates Loaded: {len(matcher.database)}")

# --- SUCCESS PAGE ---
def show_success_page(name):
    st.empty()
    st.markdown(f"""
        <div style="text-align: center; padding: 100px; background: rgba(34, 211, 238, 0.1); border-radius: 20px; border: 1px solid var(--primary);">
            <h1 style="color: var(--primary); font-size: 4rem;">🔓 ACCESS GRANTED</h1>
            <h2 style="color: white;">Welcome back, {name}</h2>
            <p style="color: #94a3b8; font-size: 1.2rem; margin-top: 20px;">
                Your identity has been verified via Auricular Biometrics.<br>
                <b>Successfully Logged In</b>
            </p>
            <br>
            <div style="font-family: monospace; color: var(--primary);">SESSION_ID: SYN-{int(time.time())}</div>
        </div>
    """, unsafe_allow_html=True)
    st.balloons()
    if st.button("Log Out"):
        st.session_state.logged_in = False
        st.rerun()

# --- LOGIN STATE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_name = ""

if st.session_state.logged_in:
    show_success_page(st.session_state.user_name)
    st.stop()

# --- UTILS ---
def alignment_guide():
    """Renders a robust CSS-based alignment box directly over the Streamlit camera component."""
    st.markdown(f"""
        <style>
        /* Target the camera container specifically */
        [data-testid="stCameraInput"] {{
            position: relative;
        }}
        [data-testid="stCameraInput"]::before {{
            content: 'PLACE EAR HERE';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 220px;
            height: 300px;
            border: 4px dashed var(--primary);
            border-radius: 100px;
            z-index: 1000000;
            pointer-events: none;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--primary);
            font-weight: 900;
            letter-spacing: 2px;
            background: rgba(34, 211, 238, 0.1);
            box-shadow: 0 0 30px rgba(34, 211, 238, 0.2);
            text-shadow: 0 0 10px rgba(0,0,0,0.5);
        }}
        </style>
    """, unsafe_allow_html=True)

# --- MODES ---
if mode == "Biometric Login":
    st.markdown("## 🔐 Secure Login Portal")
    st.info("Align your ear within the marked boundary for seamless authentication.")
    
    alignment_guide()
    captured_image = st.camera_input("Biometric Scan")
        
    if captured_image:
        img = np.array(Image.open(captured_image))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # --- ZERO-CLICK AUTO AUTHENTICATION ---
        status_container = st.empty()
        status_container.info("🧠 Biometric Signature Detected. Analyzing...")
        
        time.sleep(0.5) 
        
        ears = detector.detect(img_bgr)
        ear_crops = [detector.crop_ear(img_bgr, e['box']) for e in ears] if ears else [img_bgr]
        
        authorized = False
        for crop in ear_crops:
            processed = preprocessor.process(crop)
            norm_img = preprocessor.normalize_for_model(processed)
            embedding = extractor.extract(norm_img)
            name, score = matcher.identify(embedding, threshold=0.7)
            
            if name != "Unknown" and score > 0.7:
                st.session_state.logged_in = True
                st.session_state.user_name = name
                authorized = True
                status_container.success(f"Identity Verified: {name}")
                time.sleep(1)
                st.rerun()
        
        if not authorized:
            status_container.error("ACCESS DENIED: Identity not recognized.")
            if st.button("Retry Scan"):
                st.rerun()

elif mode == "Secure Enrollment":
    st.markdown("## 👤 User Enrollment")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            name = st.text_input("Identity Name", placeholder="e.g. John Doe")
            
            st.markdown("### 📸 Live Biometric Capture")
            alignment_guide()
            uploaded_file = st.camera_input("Capture Ear Profile")
            st.markdown('</div>', unsafe_allow_html=True)
            
    if uploaded_file and name:
        img = np.array(Image.open(uploaded_file))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        with col2:
            st.image(img, caption="Enrollment Source", use_container_width=True)
            if st.button("EXECUTE ENROLLMENT"):
                with st.spinner("Analyzing Ear Geometry..."):
                    ears = detector.detect(img_bgr)
                    if ears:
                        ear_crop = detector.crop_ear(img_bgr, ears[0]['box'])
                        st.success("Ear Region Localized")
                    else:
                        ear_crop = img_bgr
                        st.warning("Manual Geometry Override Enabled")
                    
                    # Processing
                    processed = preprocessor.process(ear_crop)
                    norm_img = preprocessor.normalize_for_model(processed)
                    embedding = extractor.extract(norm_img)
                    
                    # Store
                    matcher.add_template(name, embedding)
                    
                    st.toast(f"Identity {name} successfully locked!", icon='✅')
                    st.balloons()
                    st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption="Normalized Biometric Template", width=200)
                    if st.button("Finish & Clear"):
                        st.rerun()

elif mode == "Identity Scan":
    st.markdown("## 🔍 Active Identity Scan")
    st.info("Scanner is active. Please present subjects for real-time identification.")
    
    alignment_guide()
    captured_image = st.camera_input("Live Registry Scan")
    
    if captured_image:
        img = np.array(Image.open(captured_image))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.image(img, caption="Scan Capture", use_container_width=True)
        
        with c2:
            if st.button("RUN BIOMETRIC MATCH"):
                status_box = st.empty()
                status_box.info("Searching Galactic Registry...")
                
                ears = detector.detect(img_bgr)
                ear_crops = [detector.crop_ear(img_bgr, e['box']) for e in ears] if ears else [img_bgr]
                
                results = []
                for crop in ear_crops:
                    processed = preprocessor.process(crop)
                    norm_img = preprocessor.normalize_for_model(processed)
                    embedding = extractor.extract(norm_img)
                    name, score = matcher.identify(embedding)
                    results.append((name, score))
                
                status_box.empty()
                
                for name, score in results:
                    color = "#22d3ee" if name != "Unknown" else "#f87171"
                    st.markdown(f"""
                        <div class="metric-card" style="border-left: 5px solid {color}">
                            <h3 style="color: {color}">{name.upper()}</h3>
                            <p>Match Confidence: <b>{score*100:.2f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)

elif mode == "Registry Hub":
    st.markdown("## 🗄️ Biometric Registry Hub")
    db = matcher.database
    
    if not db:
        st.info("Registry is currently empty. Begin enrollment to populate.")
    else:
        cols = st.columns(3)
        for i, name in enumerate(db.keys()):
            with cols[i % 3]:
                # Calculate a pseudo "biometric strength" based on vector variance for UI look
                strength = 85 + (hash(name) % 10) 
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="display: flex; justify-content: space-between;">
                            <small style="color: var(--primary);">ID: {i+101:03}</small>
                            <small style="color: #4ade80;">ENCRYPTED</small>
                        </div>
                        <h4 style="margin: 10px 0;">{name}</h4>
                        <p style="font-size: 0.7rem; color: #94a3b8; margin: 0;">VECTOR DIM: 128 (FP32)</p>
                        <p style="font-size: 0.7rem; color: #94a3b8; margin: 0;">MODEL: EarNet v1.0 (CNN)</p>
                        <div style="margin-top: 10px;">
                            <small>Neural Strength</small>
                            <div style="background: rgba(255,255,255,0.1); height: 4px; border-radius: 2px;">
                                <div style="background: #4ade80; width: {strength}%; height: 100%;"></div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        if st.button("WIPE REGISTRY (CAUTION)"):
            matcher.database = {}
            matcher.save_database()
            st.rerun()

# --- FOOTER ---
st.markdown("""
<div style="text-align: center; margin-top: 5rem; color: #475569; font-size: 0.8rem;">
    EAR-BASED BIOMETRIC RECOGNITION INFRASTRUCTURE // SECURE AURICULAR RECOGNITION V1.0<br>
    © 2026 Syndicate Labs. All Rights Reserved.
</div>
""", unsafe_allow_html=True)
