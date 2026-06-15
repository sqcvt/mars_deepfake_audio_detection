import torch
import torch.nn as nn
import torchaudio 
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Model

SAMPLE_RATE = 16000
MAX_LENGTH = 64000

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(768,512),nn.LayerNorm(512),nn.ReLU(),nn.Dropout(0.4),

            nn.Linear(512,256),nn.LayerNorm(256),nn.ReLU(),nn.Dropout(0.3),

            nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.2),

            nn.Linear(128,2)
        )
    
    def forward(self,x):
        return self.net(x)
    

class Wav2Vec2Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.wav2vec2=Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

        for p in self.wav2vec2.parameters():
            p.requires_grad = False

        self.classifier = Classifier()

    def forward(self,waveform):
        outputs = self.wav2vec2(input_values=waveform)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)
    

def preprocess_audio(audio_path):

    data,sr = sf.read(audio_path,dtype="float32",always_2d=True)

    waveform = torch.from_numpy(data.T)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0,keepdim=True)

    waveform = waveform.squeeze(0)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform,sr,SAMPLE_RATE)

    if len(waveform) < MAX_LENGTH:
        waveform = nn.functional.pad(waveform,(0, MAX_LENGTH-len(waveform)))
    else:
        waveform = waveform[:MAX_LENGTH]

    waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var() + 1e-7)

    return waveform.unsqueeze(0)