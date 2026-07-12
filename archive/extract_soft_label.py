# extract_soft_label.py

import os
import sys
import torch
import pickle
import torch.nn.functional as F
from tqdm import tqdm

# 유틸 경로 설정
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

from vild_config import AudioViLDConfig
from vild_model import SimpleAudioEncoder, ViLDTextHead
from vild_parser_teacher import AudioParser
from seed_utils import set_seed
from vild_utils import normalize_mel_shape


def extract_soft_labels(top_k=5):
    set_seed(42)
    config = AudioViLDConfig()
    parser = AudioParser(config, segment_mode=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로딩
    model = SimpleAudioEncoder(config).to(device)
    label_head = ViLDTextHead(config).to(device)
    model.eval()
    label_head.eval()

    ckpt_path = f"teacher_checkpoint_{config.mark_version}.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[Error] Checkpoint 파일이 존재하지 않습니다: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    label_head.load_state_dict(checkpoint["head"])

    dataset = parser.get_all_audio_files()
    results = []
    print(f"총 오디오 파일 수: {len(dataset)}")

    for path in tqdm(dataset, desc="Extracting Hard Labels from Top-K Segments"):
        segments = parser.load_and_segment(path)
        if not segments:
            print(f"[Skip] 유효한 세그먼트 없음: {path}")
            continue

        mel_tensor_list = []
        for seg in segments:
            norm = normalize_mel_shape(seg)
            if norm is None:
                print(f"[Skip] 정규화 실패: {path}")
                mel_tensor_list = []
                break
            mel_tensor_list.append(norm)

        if not mel_tensor_list:
            continue

        try:
            mel_tensor = torch.stack(mel_tensor_list).to(device)  # [N, 1, 64, 101]
        except Exception as e:
            print(f"[Error] torch.stack 실패: {path}, {e}")
            continue

        with torch.no_grad():
            audio_embed = model(mel_tensor)  # [N, D]
            text_embed = config.get_class_text_embeddings().to(device)  # [C, D]
            soft_labels = label_head(audio_embed, text_embed)  # [N, C]
            probs = F.softmax(soft_labels, dim=-1)  # 확률로 변환

        # max prob 기준으로 K개 선택
        max_conf, _ = probs.max(dim=1)  # [N]
        topk_idx = torch.topk(max_conf, min(top_k, len(max_conf)), largest=True).indices
        topk_labels = probs[topk_idx]  # [K, C]

        # hard label로 변환
        hard_labels = torch.argmax(topk_labels, dim=1)  # [K]

        results.append({
            "path": path,
            "hard_labels": hard_labels.cpu().tolist()  # e.g., [0, 2, 2, 0, 1]
        })

    print(f"생성된 hard label 샘플 수: {len(results)} / {len(dataset)}")

    out_path = f"hard_labels_{config.mark_version}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    print(f"[완료] Hard labels 저장됨: {out_path}")


if __name__ == "__main__":
    extract_soft_labels()
    