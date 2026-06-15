import streamlit as st
import torch
import tempfile
import json
import pandas as pd

from model import (Wav2Vec2Classifier,preprocess_audio)

st.markdown("""
<style>
.big-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#4F8BF9;
}

.sub-title{
    text-align:center;
    color:gray;
    margin-bottom:30px;
}

.result-box{
    padding:15px;
    border-radius:10px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='big-title'>🎙️ Deepfake Audio Detection</div>",unsafe_allow_html=True)

st.markdown("<div class='sub-title'>Upload a WAV file and detect whether it is Real or AI Generated</div>",unsafe_allow_html=True)

# load Config


# Load config
with open("config.json") as f:
    config = json.load(f)

THRESHOLD = config["threshold"]

# Load Model

@st.cache_resource
def load_model():

    model = Wav2Vec2Classifier()

    state = torch.load("best_classifier.pth",map_location="cpu")

    model.classifier.load_state_dict(state)

    model.eval()

    return model

model = load_model()

# Audio file Upload

uploaded_file = st.file_uploader("Upload WAV Audio",type=["wav"])


# Prediction

if uploaded_file is not None:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False,suffix=".wav") as tmp:

        tmp.write(uploaded_file.read())

        audio_path = tmp.name

    with st.spinner("Analyzing Audio..."):

        waveform = preprocess_audio(audio_path)

        with torch.no_grad():

            logits = model(waveform)

            probs = torch.softmax(logits,dim=1)

            fake_prob = probs[0][1].item()
            real_prob = probs[0][0].item()


# Result Banner

    if fake_prob >= THRESHOLD:

        st.error(f" ❎ Deepfake Detected ")

    else:

        st.success(f" ✅Genuine Audio ")

    st.divider()

# metric
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="🎵 REAL Probability",value=f"{real_prob:.2%}")

    with col2:
        st.metric(label="🤖 FAKE Probability",value=f"{fake_prob:.2%}")

    st.divider()


# probability
    
    st.subheader("Prediction Probabilities")

    df = pd.DataFrame({"Class": ["Real Audio", "Fake Audio"],
            "Probability (%)": [round(real_prob * 100, 2),round(fake_prob * 100, 2)]})


# Bar Chart
    
    st.subheader(" Confidence Visualization")

    chart_df = pd.DataFrame({"Probability": [real_prob,fake_prob]},index=["Real","Fake"])

    st.bar_chart(chart_df)


    st.subheader("🎯 Final Verdict")

    if fake_prob >= THRESHOLD:

        st.markdown("""
<div style="
    background-color:#ff4b4b;
    padding:15px;
    border-radius:10px;
    color:white;
    font-size:20px;
    text-align:center;">
    ⚠️ This audio is likely AI Generated (Deepfake)
</div>""",unsafe_allow_html=True)

    else:

        st.markdown("""
<div style="
    background-color:#00c853;
    padding:15px;
    border-radius:10px;
    color:white;
    font-size:20px;
    text-align:center;">
    ✅ This audio appears Genuine
</div>""",unsafe_allow_html=True)

   

