#  Deepfake Audio Detection using Wav2Vec2

##  Project Overview

Deepfake audio generated using modern generative AI models has become increasingly realistic, making it difficult to distinguish synthetic speech from genuine human recordings. Such audio can be misused for impersonation, fraud, misinformation, and social engineering attacks.

This project presents an end-to-end Deepfake Audio Detection system that leverages pretrained Wav2Vec2 representations and a custom neural classifier to identify whether an audio sample is:

*  Genuine (Real Human Speech)
*  Deepfake (AI-Generated Speech)

The solution is deployed as an interactive Streamlit web application where users can upload a WAV audio file and instantly receive predictions along with confidence scores.

---

#  Project Pipeline

Audio Upload

   ↓

Preprocessing

   ↓

Wav2Vec2 Feature Extraction

   ↓

Mean Pooling

   ↓

MLP Classifier

   ↓

Softmax Probabilities

   ↓

Real / Fake Prediction

   ↓

Visualization in Streamlit

---


#  Features

* Upload WAV audio files
* Real-time Deepfake detection
* Wav2Vec2-based feature extraction
* Deep neural network classifier
* Confidence score visualization
* Streamlit web interface
* Deployable on Render

---

#  Methodology

## Step 1: Audio Preprocessing

Each audio file undergoes preprocessing before being fed into the model:

* Convert stereo audio to mono
* Resample audio to 16 kHz
* Trim or pad audio to 4 seconds
* Z-score normalization

Output:

Audio Tensor Shape:

(64000,)

---

## Step 2: Feature Extraction

A pretrained Facebook Wav2Vec2 model is used as the backbone.

Model:

facebook/wav2vec2-base-960h

This backbone model extracts contextual speech representations from raw audio waveforms.

Feature Dimension:

768

The Wav2Vec2 parameters are frozen during training to reduce computational cost and prevent overfitting.

---

## Step 3: Classification Head

A custom Multi-Layer Perceptron (MLP) classifier is trained on top of the extracted embeddings.

Architecture:

Input (768)

   ↓

Linear(768 → 512)

   ↓

LayerNorm + ReLU + Dropout

   ↓

Linear(512 → 256)

   ↓

LayerNorm + ReLU + Dropout

   ↓

Linear(256 → 128)

   ↓

ReLU + Dropout

   ↓

Linear(128 → 2)

   ↓

Softmax

Output Classes:

* Class 0 → Genuine Audio
* Class 1 → Deepfake Audio

---

# 📂 Project Structure

```text
Deepfake-Audio-Detection/
│
├── app.py
├── model.py
├── best_classifier.pth
├── config.json
├── requirements.txt
├── README.md
└── test_deepfake_audio.py
```

---

#  Streamlit Application

The web application allows users to:

* Upload audio files
* Listen to uploaded audio
* View prediction probabilities
* Visualize confidence scores
* Receive final verdict

Example Output:

| Class | Probability |
| ----- | ----------- |
| Real  | 0.42%       |
| Fake  | 99.58%      |

Prediction:

⚠️ Deepfake Detected

---

#  Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Classification Metrics:

Accuracy = Correct Predictions / Total Samples

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 Score = 2 × Precision × Recall / (Precision + Recall)

---

#  Technologies Used

### Deep Learning

* PyTorch
* Transformers
* Wav2Vec2

### Audio Processing

* Torchaudio
* SoundFile
* NumPy

### Deployment

* Streamlit
* Render

### Development

* Python
* Git
* GitHub

---

# ⚙️ Installation

Clone Repository

```bash
git clone https://github.com/sqcvt/mars_deepfake_audio_detection.git

cd deepfake_audio_detection
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Run Application

```bash
streamlit run app.py
```
---

#  Future Improvements

* Support longer audio recordings
* Real-time microphone inference
* Ensemble deepfake detection models
* Multi-language support
* Explainable AI visualizations
* Model quantization for faster deployment

---

#  Author

Mayur Arya 
---
