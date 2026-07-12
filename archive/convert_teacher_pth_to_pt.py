# convert_teacher_pth_to_pt.py
'''
.pth 파일만 있고 .pt 파일은 없어서
.pth 파일 2개를 불러와서 .pt 로 저장함.

'''
import os
import torch

def convert_teacher_checkpoint(mark_version="mark4.1"):  # change needed
    encoder_path = f"best_teacher_encoder_{mark_version}.pth"
    classifier_path = f"best_teacher_classifier_{mark_version}.pth"
    output_path = f"teacher_checkpoint_{mark_version}.pt"

    if not os.path.isfile(encoder_path) or not os.path.isfile(classifier_path):
        raise FileNotFoundError("필요한 .pth 파일이 존재하지 않습니다.")

    # 모델 로드
    encoder_state = torch.load(encoder_path, map_location="cpu")
    classifier_state = torch.load(classifier_path, map_location="cpu")

    # 합쳐서 저장
    torch.save({
        "audio_encoder": encoder_state,
        "text_head": classifier_state
    }, output_path)

    print(f"[완료] {output_path} 생성 완료")


if __name__ == "__main__":
    convert_teacher_checkpoint(mark_version="mark4.1")  # change needed
