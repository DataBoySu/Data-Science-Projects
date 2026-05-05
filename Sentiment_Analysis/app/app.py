from attention_layer import AttentionLayer 
import streamlit as st
import pickle
import numpy as np
import re
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import random
import time

# Load tokenizer
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load model
model = load_model("model/emotion_lstm_model.keras")

# Emotion labels
label_map = {
    0: "joy", 1: "sadness", 2: "anger", 3: "fear", 4: "love", 5: "surprise"
}

emoji_map = {
    "joy": "🙂", "sadness": "😢", "anger": "😠", 
    "fear": "😨", "love": "💖", "surprise": "😲"
}

score_weights = {
    "love": 3, "joy": 2, "surprise": 1, 
    "sadness": -1, "anger": -2, "fear": -3
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9.\s]", "", text)
    return text

def classify_sentences(text):
    cleaned = clean_text(text)
    sentences = [s.strip() for s in cleaned.split(".") if s.strip()]
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=80)
    preds = model.predict(padded)
    results = []
    freq = {}
    for i, prob in enumerate(preds):
        idx = np.argmax(prob)
        label = label_map[idx]
        conf = float(np.max(prob))
        results.append((sentences[i], label, conf))
        freq[label] = freq.get(label, 0) + 1
    return results, freq

def summarize_sentiment(freq):
    total = sum(freq.values())
    if total == 1:
        single_emotion = next(iter(freq))
        return f"{emoji_map[single_emotion]} The text is **overwhelmingly {single_emotion}**."
    elif total == 2:
        emotions = list(freq.keys())
        return f"{emoji_map[emotions[0]]}{emoji_map[emotions[1]]} A mix of emotions, but the tone feels **overall positive**."

    weighted_sum = sum(freq.get(k, 0) * v for k, v in score_weights.items())
    love = freq.get("love", 0)
    fear = freq.get("fear", 0)
    surprise = freq.get("surprise", 0)

    if love >= 1 and weighted_sum >= 0:
        return "💖 The emotion leans **strongly positive**, driven by love."
    elif fear >= 2 and weighted_sum <= -3:
        return "😨 The emotion leans **strongly negative**, dominated by fear."
    elif weighted_sum > 2:
        return "🙂 The user seems **positive overall**."
    elif weighted_sum < -2:
        return "🙁 The user seems **negative overall**."
    elif surprise >= 2:
        return "😲 Mixed tone, but surprise adds complexity — possibly reflective or startled."
    else:
        return "😐 The sentiment is leaning positive." if weighted_sum >= 0 else "😐 The sentiment is leaning negative."

# Streamlit UI
st.set_page_config(page_title="Emotion Feedback Analyzer", page_icon="💬", layout="centered")
st.markdown("""
    <style>
    .animated-bg {
        position: fixed;
        right: 0;
        top: 0;
        width: 100px;
        height: 100vh;
        z-index: -1;
        overflow: hidden;
    }
    .leaf {
        position: absolute;
        width: 30px;
        height: 30px;
        background-image: url('https://cdn-icons-png.flaticon.com/512/415/415733.png');
        background-size: contain;
        background-repeat: no-repeat;
        animation: fall 80s linear infinite;
        opacity: 0.8;
    }
    @keyframes fall {
        0% { top: -50px; transform: rotate(0deg); }
        100% { top: 100vh; transform: rotate(360deg); }
    }
    .loader-container {
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0, 0, 0, 0.4);
        z-index: 1000;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .bunny-loader {
        width: 80px;
        height: 80px;
        background: url('https://www.gifcen.com/wp-content/uploads/2022/04/hopping-bunny-gif.gif') no-repeat center center;
        background-size: contain;
        display: flex;
    }
    .bunny-container {
        display: flex;
        gap: 10px;
    }
    .fullscreen-image {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color: black;
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 2000;
    }
    .fullscreen-image img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }
    </style>
    <div class="animated-bg">
        <div class="leaf" style="left:10px; animation-delay: 0s;"></div>
        <div class="leaf" style="left:30px; animation-delay: 2s;"></div>
        <div class="leaf" style="left:50px; animation-delay: 4s;"></div>
    </div>
""", unsafe_allow_html=True)

st.title("🎯 Emotion-Aware Feedback Analyzer")
st.markdown("Type a paragraph or feedback message. We'll detect emotional tone and classify sentence by sentence.")

with st.sidebar:
    st.markdown("### 🤖 Emotion Assistant")
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712139.png", width=100)
    st.markdown("**Emotion Legend:**")
    for k in label_map.values():
        st.markdown(f"{emoji_map[k]} {k.capitalize()}")

input_text = st.text_area("Enter your message here:", height=200)

if st.button("🔍 Analyze"):
    if input_text.strip():
        loader_placeholder = st.empty()
        loader_html = """
        <div class='loader-container'>
            <div class='bunny-container'>
                <div class='bunny-loader'></div>
                <div class='bunny-loader'></div>
                <div class='bunny-loader'></div>
            </div>
        </div>
        """
        loader_placeholder.markdown(loader_html, unsafe_allow_html=True)
        time.sleep(2.5)
        loader_placeholder.empty()

        results, freq = classify_sentences(input_text)

        st.subheader("📄 Sentence-wise Emotion Classification")
        for i, (sent, label, conf) in enumerate(results):
            st.markdown(f"""
            <div style='padding: 1rem; margin-bottom: 0.5rem; border-left: 6px solid #555; background-color: #f9f9f9; border-radius: 8px;'>
                <strong>Sentence {i+1}</strong>: {sent}<br>
                → Emotion: <b style='color: #0055cc;'>{emoji_map[label]} {label}</b> (Confidence: {conf:.2f})
            </div>
            """, unsafe_allow_html=True)

        num_emotions = len(freq)
        if num_emotions > 1:
            st.subheader("📊 Emotion Frequency")
            labels = list(freq.keys())
            counts = list(freq.values())
            emo_labels = [f"{emoji_map[l]} {l}" for l in labels]

            fig, ax = plt.subplots(figsize=(4, 2.5))
            if num_emotions < 4:
                ax.pie(counts, labels=emo_labels, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
            else:
                ax.barh(emo_labels, counts, color="#4a90e2")
            st.pyplot(fig)

        st.subheader("🧠 Summary")
        st.markdown(f"""
        <div style='padding: 1rem; border-radius: 10px; background-color: #e6f7ff; border: 2px solid #91d5ff;'>
            <h4>{summarize_sentiment(freq)}</h4>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to analyze.")

# Thank You Image Button
import base64

def get_image_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

if st.button("Message"):
    img_base64 = get_image_base64("ty.jpg")
    st.markdown(f"""
    <style>
    .fullscreen-image {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color: black;
        z-index: 2000;
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    .fullscreen-image img {{
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }}
    </style>
    <div class="fullscreen-image">
        <img src="data:image/jpeg;base64,{img_base64}" alt="Thank You">
    </div>
    """, unsafe_allow_html=True)
