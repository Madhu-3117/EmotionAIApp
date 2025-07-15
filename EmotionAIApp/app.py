import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from dummy_model import dummy_model as model

# ------------------ UI CONFIG ------------------
st.set_page_config(page_title="Emotion AI 🎭", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main {
        background-color: #fff;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    }
    .emoji-title {
        font-size: 42px;
        animation: emojiPulse 2s infinite;
    }
    @keyframes emojiPulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.08); }
        100% { transform: scale(1); }
    }
    .glow {
        padding: 10px;
        border-radius: 10px;
        background-color: #f9f9f9;
        box-shadow: 0 0 20px rgba(0, 153, 255, 0.6);
        font-size: 20px;
        font-weight: bold;
        color: #222;
        margin-bottom: 20px;
    }
    footer {
        visibility: visible;
    }
    footer:after {
        content:'Made by Madhu Mitha 🧑‍💻';
        display:block;
        text-align:center;
        color: gray;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ EMOTION STUFF ------------------
emotion_dict = {
    0: "Angry", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"
}

emotion_emoji = {
    "Angry": "😠", "Disgust": "🤢", "Fear": "😨",
    "Happy": "😄", "Sad": "😢", "Surprise": "😲", "Neutral": "😐"
}

emotion_quotes = {
    "Angry": "Anger doesn’t solve anything; it builds nothing.",
    "Disgust": "Even in the ugliest of moments, find the beauty.",
    "Fear": "Feel the fear and do it anyway!",
    "Happy": "Happiness is not by chance, but by choice.",
    "Sad": "Sadness flies away on the wings of time.",
    "Surprise": "Magic happens when you least expect it.",
    "Neutral": "A calm mind brings inner strength."
}

emotion_music = {
    "Angry": "https://www.youtube.com/watch?v=9X_ViIPA-Gc",
    "Disgust": "https://www.youtube.com/watch?v=krnxaZ21p0o",
    "Fear": "https://www.youtube.com/watch?v=sCNrK-n68CM",
    "Happy": "https://www.youtube.com/watch?v=ZbZSe6N_BXs",
    "Sad": "https://www.youtube.com/watch?v=4N3N1MlvVc4",
    "Surprise": "https://www.youtube.com/watch?v=YQHsXMglC9A",
    "Neutral": "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
}

# ------------------ SIDEBAR ------------------
st.sidebar.title("💡 About EmotionAI")
st.sidebar.markdown("""
This app detects **emotions** from your uploaded face image  
using a dummy AI model. Built using Python, Streamlit, and ❤️.

🎯 Upload → Detect → Feel → Smile

🔧 Developer: Madhu Mitha
""")

# ------------------ MAIN APP ------------------
st.markdown("<h1 class='emoji-title'>🎭 AI Emotion Detector</h1>", unsafe_allow_html=True)
st.markdown("Upload your photo and discover your current emotion...")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Your Image", use_column_width=True)

    # Convert to grayscale and resize
    gray_img = img.convert("L")
    resized = gray_img.resize((48, 48))
    normalized = np.array(resized) / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))

    # Predict with dummy model
    prediction = model.predict(reshaped)
    predicted_class = int(np.argmax(prediction))
    emotion = emotion_dict[predicted_class]

    # Output
    st.markdown(f"<div class='glow'>🎯 Emotion Detected: {emotion_emoji[emotion]} **{emotion}**</div>", unsafe_allow_html=True)
    st.info(f"💬 **Quote**: _{emotion_quotes[emotion]}_")
    st.markdown(f"🔊 **Listen:** [🎵 Play Emotion Music]({emotion_music[emotion]})")

    st.markdown("---")
    st.write("Try uploading another photo!")

# ------------------ FOOTER ------------------
st.markdown("<footer></footer>", unsafe_allow_html=True)
