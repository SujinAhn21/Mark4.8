# plot_audio.py

import os  
import sys
import argparse
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
UTILS_DIR = os.path.join(PROJECT_ROOT, 'utils')
VILD_DIR = os.path.join(PROJECT_ROOT, 'vild')
for p in (PROJECT_ROOT, UTILS_DIR, VILD_DIR):
    if p not in sys.path:
        sys.path.append(p)

from seed_utils import set_seed
from vild_config import AudioViLDConfig

def create_dummy_audio(file_path, sample_rate=16000, duration=1):
    """테스트용 무작위 오디오 파일 생성"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data = np.random.uniform(-0.5, 0.5, sample_rate * duration).astype(np.float32)
    sf.write(file_path, data, sample_rate)
    print(f"[샘플 생성] 임시 테스트 오디오 생성됨: {file_path}")

def plot_waveform(waveform, filename, save_dir, mark_version):
    """Waveform 시각화"""
    try:
        plt.figure(figsize=(10, 3))
        plt.plot(waveform[0].cpu().numpy())
        plt.title("Waveform: " + filename)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        wave_path = os.path.join(save_dir, f"{filename}_waveform_{mark_version}.png")
        plt.savefig(wave_path, dpi=150)
        plt.close()
        print(f"[완료] Waveform plot 저장됨: {wave_path}")
        return wave_path
    except Exception as e:
        print(f"[Error] Waveform 시각화 실패: {e}")
        return "Waveform plot failed"

def plot_mel_spectrogram(waveform, sr, filename, save_dir, config, mark_version):
    """Mel spectrogram 시각화"""
    try:
        if sr != config.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=config.sample_rate)(waveform)
            sr = config.sample_rate

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=config.fft_size,
            hop_length=config.hop_length,
            n_mels=config.n_mels
        )
        mel = mel_transform(waveform)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

        plt.figure(figsize=(10, 4))
        plt.imshow(mel_db.squeeze(0).cpu().numpy(), origin='lower', aspect='auto', cmap='viridis')
        plt.title("Mel Spectrogram: " + filename)
        plt.xlabel("Time")
        plt.ylabel("Mel bins")
        mel_path = os.path.join(save_dir, f"{filename}_mel_{mark_version}.png")
        plt.savefig(mel_path, dpi=150)
        plt.close()
        print(f"[완료] Mel spectrogram plot 저장됨: {mel_path}")
        return mel_path
    except Exception as e:
        print(f"[Error] Mel spectrogram 시각화 실패: {e}")
        return "Mel plot failed"

def plot_waveform_and_mel(path, save_dir=None, seed_value=42, mark_version="mark4.1"):  # change needed
    set_seed(seed_value)

    # 저장 경로 설정
    if save_dir is None:
        save_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(save_dir, exist_ok=True)

    # torchaudio 백엔드 설정 (경고 무시)
    try:
        torchaudio.set_audio_backend("soundfile")
    except Exception:
        pass

    # 오디오 파일 로딩
    try:
        waveform_np, sr = sf.read(path, dtype='float32')
        waveform = torch.from_numpy(waveform_np)
        if waveform.ndim == 2:
            # (T, C) 또는 (C, T) 불문하고 평균이 되도록 처리
            if waveform.shape[0] >= waveform.shape[1]:
                waveform = torch.mean(waveform, dim=1)
            else:
                waveform = torch.mean(waveform, dim=0)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]
    except Exception as e:
        print(f"[Error] 오디오 파일 로딩 실패: {path}\n{e}")
        return

    filename = os.path.splitext(os.path.basename(path))[0]
    config = AudioViLDConfig(mark_version=mark_version)

    wave_path = plot_waveform(waveform, filename, save_dir, mark_version)
    mel_path = plot_mel_spectrogram(waveform, sr, filename, save_dir, config, mark_version)

    print(f"[시도 완료] 저장된 시각화 결과 -> Wave: {wave_path}, Mel: {mel_path}")


if __name__ == "__main__":
    """
    [Deprecated] 고정 경로(구조 불일치)
    test_audio_file = os.path.join(BASE_DIR, "data/test_heavy_impact", "heavy_impact_501.wav") # change needed
    if not os.path.exists(test_audio_file):
        print(f"[경고] 테스트 오디오 파일 없음: {test_audio_file}")
        create_dummy_audio(test_audio_file)
    plot_waveform_and_mel(test_audio_file, seed_value=42)
    """

    # [변경] mark4.x 구조 대응 + 선택적 인자 처리
    parser = argparse.ArgumentParser()
    parser.add_argument('--mark_version', type=str, required=True)
    parser.add_argument('--path', type=str, default=None, help="시각화할 오디오 파일 경로")
    args = parser.parse_args()

    mv = args.mark_version
    candidate = args.path

    if candidate is None:
        # data/test 아래 임의의 .wav 파일을 선택
        test_dir = os.path.join(PROJECT_ROOT, "data", "test")
        os.makedirs(test_dir, exist_ok=True)
        wavs = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(".wav")])
        if wavs:
            candidate = wavs[0]
        else:
            # 없으면 더미 생성
            candidate = os.path.join(test_dir, "sample.wav")
            create_dummy_audio(candidate, sample_rate=16000, duration=1)

    plot_waveform_and_mel(candidate, seed_value=42, mark_version=mv)
    