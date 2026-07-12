# 세그먼트 개수 오류 방지를 위해 ... 
# .wav 파일의 길이(초)와 segment 수를 확인..
# 한시간 쌩으로 날리기 싫으면 한번 돌려서 확인할 것.

import os
import torchaudio

folder = "data_wav" 
sample_rate = 16000
segment_duration = 2.0
hop_duration = segment_duration * 0.5

for fname in os.listdir(folder):
    if fname.endswith(".wav"):
        path = os.path.join(folder, fname)
        waveform, sr = torchaudio.load(path)
        duration = waveform.shape[1] / sr
        n_segments = int((duration - segment_duration) // (segment_duration - hop_duration)) + 1
        print(f"{fname}: {duration:.2f}초, segment 수 ≈ {n_segments}")
