# parser_utils.py

import torch
import torchaudio
import soundfile as sf
import os
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def load_audio_file(file_path, target_sample_rate, resampler_cache=None):
    """
    오디오 파일을 로드하고 모노화, 정규화, 리샘플링하여 [1, T] 텐서로 반환.

    Args:
        file_path (str): 오디오 파일 경로
        target_sample_rate (int): 타겟 샘플레이트
        resampler_cache (dict, optional): Resample 객체 캐시 (성능 향상용)

    Returns:
        torch.Tensor or None: [1, T] 형태 waveform 또는 실패 시 None
    """
    file_path_norm = os.path.normpath(file_path).replace("\\", "/")
    if not os.path.isfile(file_path_norm):
        logging.warning(f"[Load Fail] 파일 없음: {file_path_norm}")
        return None

    try:
        waveform_np, sr = sf.read(file_path_norm, dtype='float32')
        waveform = torch.from_numpy(waveform_np)
    except Exception as e:
        logging.error(f"[Load Fail] 파일 로딩 실패: {file_path_norm} -> {e}")
        return None

    if waveform.ndim == 0 or waveform.numel() == 0:
        logging.warning(f"[Empty] 비어있는 파일: {file_path_norm}")
        return None

    """
    [Deprecated] 채널 평균화가 (T, C) 가정만 사용
    # 모노 변환
    if waveform.ndim > 1:
        try:
            waveform = waveform.mean(dim=1)  # (채널, 샘플 수) -> 모노
        except Exception as e:
            logging.error(f"[Mono Fail] 다채널 평균화 실패: {file_path_norm} -> {e}")
            return None
    """

    # [변경] (T, C) 또는 (C, T) 모두 대응. 작은 축을 채널로 간주해 평균.
    if waveform.ndim == 2:
        try:
            dim_time, dim_chan = waveform.shape
            if dim_time >= dim_chan:
                # (T, C)로 판단 → 채널 평균
                waveform = waveform.mean(dim=1)
            else:
                # (C, T)로 판단 → 채널 평균
                waveform = waveform.mean(dim=0)
        except Exception as e:
            logging.error(f"[Mono Fail] 다채널 평균화 실패: {file_path_norm} -> {e}")
            return None
    elif waveform.ndim > 2:
        logging.error(f"[Shape Error] 예상 외 차원: {waveform.shape} @ {file_path_norm}")
        return None
    # 1D는 그대로 사용

    waveform = waveform.view(1, -1)  # [1, T]

    if waveform.numel() == 0:
        logging.warning(f"[Empty] 모노 변환 후 비어 있음: {file_path_norm}")
        return None

    # 볼륨 정규화
    max_amp = waveform.abs().max()
    if max_amp > 0:
        waveform = waveform / max_amp

    # 리샘플링
    if sr != target_sample_rate:
        try:
            if resampler_cache is not None:
                key = (sr, target_sample_rate)
                if key not in resampler_cache:
                    resampler_cache[key] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
                resampler = resampler_cache[key]
            else:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)

            waveform = resampler(waveform)

        except Exception as e:
            logging.error(f"[Resample Fail] {file_path_norm} (sr: {sr} -> {target_sample_rate}, shape: {waveform.shape}) -> {e}")
            return None

    return waveform
    