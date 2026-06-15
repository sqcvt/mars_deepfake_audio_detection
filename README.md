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

**Evaluation Metrics Used**

* Accuracy
* F1 Score
* Equal Error Rate (EER)
* Confusion Matrix
* Class-wise Accuracy (Real and Fake)

---

# Performance Report

The Detection system was evaluated on the held-out test dataset using multiple classification metrics to assess its effectiveness in distinguishing genuine speech from AI-generated speech.

| Metric                 | Score      |
| ---------------------- | ---------- |
| Accuracy               | **84.07%** |
| F1 Score               | **85.02%** |
| Equal Error Rate (EER) | **8.08%** |
| Real Audio Accuracy    | **79.55%** |
| Fake Audio Accuracy    | **88.40%** |

# Accuracy

Accuracy measures the overall proportion of correctly classified audio samples. The model achieved an accuracy of **84.07%**, indicating strong performance in distinguishing real and fake speech.

### F1 Score

The F1 Score balances precision and recall, making it particularly useful for binary classification tasks. The model achieved an F1 Score of **85.02%**, demonstrating reliable detection performance across both classes.

### Equal Error Rate (EER)

EER represents the operating point where the False Acceptance Rate (FAR) equals the False Rejection Rate (FRR). Lower EER values indicate better discrimination between genuine and deepfake audio. The proposed model achieved an EER of **8.08%**.

### Confusion Matrix

|             | Predicted Real | Predicted Fake |
| ----------- | -------------- | -------------- |
| Actual Real | TN             | FP             |
| Actual Fake | FN             | TP             |

The confusion matrix provides a detailed breakdown of correct and incorrect predictions, helping analyze the model's strengths and failure cases.

### Performance Visualization

#### Score Distribution

The probability distributions of real and fake audio samples show a clear separation between the two classes, with the EER threshold used as the optimal decision boundary.

#### Confusion Matrix Heatmap

A confusion matrix heatmap is generated to visualize classification performance and identify false positive and false negative predictions.

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
