import cv2
import streamlit as st
import numpy as np
from util import load_sign_model, predict_sign
from sentence_builder import SentenceBuilder

st.title("Sign Language Detection with Sentence Builder (CNN Model)")

# UI setup
run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])
pred_placeholder = st.empty()   # Placeholders for text output
sentence_placeholder = st.empty()

# Load model + labels
model, labels = load_sign_model()
builder = SentenceBuilder(max_words=10, min_conf=0.01, cooldown=1.5)

# Open webcam
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Preprocess frame for CNN
    img_resized = cv2.resize(frame, (64, 64))
    img_resized = img_resized.astype("float32") / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)  # shape (1, 64, 64, 3)

    # Predict
    word, conf = predict_sign(model, labels, img_resized)
    # ✅ only add to sentence if confidence is high
    if conf > 0.8:  
        builder.add_word(word, conf)

    # Overlay prediction
    cv2.putText(frame, f"{word} ({conf:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, builder.get_sentence(), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)

    FRAME_WINDOW.image(frame, channels="BGR")
    pred_placeholder.write(f"**Prediction:** {word} ({conf:.2f})")
    sentence_placeholder.write(f"**Sentence:** {builder.get_sentence()}")

cap.release()
