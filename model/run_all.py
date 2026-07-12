# run_all.py (지식 증류 파이프라인)

import os
import subprocess
import logging
from datetime import datetime
import argparse
import sys

# ===== 파라미터 설정 =====
parser = argparse.ArgumentParser(description="소음 분류 전체 학습 파이프라인 (지식 증류 포함)")
parser.add_argument("--mark_version", type=str, required=True, help="실행할 모델 버전")
args = parser.parse_args()
mark_version = args.mark_version

# ===== 경로 및 로그 설정 =====
# [변경] model/ 기준이 아닌 프로젝트 루트 기준으로 스크립트 경로를 호출
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      # .../model
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # 프로젝트 루트
PRE_DIR = os.path.join(PROJECT_ROOT, "preprocessing")
EXT_DIR = os.path.join(PROJECT_ROOT, "extraction")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

# [변경] 로그는 프로젝트 루트/logFiles 사용 (작업 전반의 단일 로그 보관)
LOG_DIR = os.path.join(PROJECT_ROOT, "logFiles")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(LOG_DIR, f"run_pipeline_distillation_{mark_version}_{timestamp}.txt")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# ===== 데코레이터 =====
def timed_step(func):
    def wrapper(*args, **kwargs):
        step_name = func.__name__.replace("run_", "").replace("_", " ").title()
        logging.info(f"\n[실행 시작] --> {step_name}")
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        duration = (end - start).total_seconds()
        logging.info(f"[완료] --> {step_name} (소요시간: {duration:.2f}초)")
        return result
    return wrapper

# ===== 서브프로세스 실행 함수 =====
def run_subprocess(command_list):
    """하위 스크립트를 실행하되 출력을 실시간으로 흘려보낸다.
    - stdout(print 기반 epoch/INFO 로그): 한 줄씩 실시간으로 터미널+로그파일 양쪽에 출력.
    - stderr(tqdm 배치 진행바): 파이프로 잡지 않고 터미널로 직접 보내 진행바가 실시간으로 렌더링됨.
    - PYTHONUNBUFFERED=1: 하위 파이썬의 print가 버퍼에 안 쌓이고 즉시 flush되게 함.
    (기존엔 capture_output=True로 출력을 다 잡아뒀다가 단계 종료 후에야 한꺼번에 찍어서,
     학습이 도는 동안 터미널에 아무것도 안 보이던 문제를 해결.)"""
    try:
        logging.info(f"[CMD] {' '.join(command_list)}")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=None,          # tqdm(stderr)은 터미널로 직접 -> 실시간 진행바
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,            # 라인 버퍼링
            env=env,
        )
        for line in process.stdout:
            logging.info(line.rstrip("\n"))   # 터미널(StreamHandler)+로그파일(FileHandler) 실시간
        process.wait()
        return process.returncode
    except Exception as e:
        logging.error(f"[ERROR] Subprocess 실행 중 예외 발생: {e}")
        return 1

# [Deprecated] 상대경로 없이 현재 디렉터리에 스크립트가 있다고 가정하고 호출
#   -> mark4.x 폴더 분리 구조(preprocessing/, extraction/, model/)에서 실패 위험.
#   -> 아래와 같이 PROJECT_ROOT 기반의 절대경로로 호출하도록 변경.

# ===== 단계별 실행 함수 정의 (지식 증류 파이프라인) =====
@timed_step
def run_step0_preprocess_audio():
    """오디오 파일을 고정된 길이로 전처리."""
    return run_subprocess([sys.executable, os.path.join(PRE_DIR, "fix_audio_length.py"), "--mark_version", mark_version])

@timed_step
def run_step1_generate_dataset_index():
    """데이터셋 인덱스 CSV 파일을 생성."""
    return run_subprocess([sys.executable, os.path.join(PRE_DIR, "generate_dataset_index.py"), "--mark_version", mark_version])

@timed_step
def run_step2_teacher_model_train():
    """Teacher 모델을 학습시킴."""
    return run_subprocess([sys.executable, os.path.join(MODEL_DIR, "teacher_train.py"), "--mark_version", mark_version])

@timed_step
def run_step3_extract_hard_labels():
    """학습 데이터로부터 Hard Label을 추출."""
    return run_subprocess([sys.executable, os.path.join(EXT_DIR, "extract_hard_labels.py"), "--mark_version", mark_version])

@timed_step
def run_step4_extract_soft_labels():
    """학습된 Teacher 모델로부터 Soft Label을 추출. (지식 증류 핵심)"""
    return run_subprocess([sys.executable, os.path.join(EXT_DIR, "extract_soft_labels.py"), "--mark_version", mark_version])

@timed_step
def run_step5_student_distillation_train():
    """Hard Label과 Soft Label을 함께 사용하여 Student 모델을 학습시킴."""
    return run_subprocess([sys.executable, os.path.join(MODEL_DIR, "student_train_distillation.py"), "--mark_version", mark_version])

@timed_step
def run_step6_evaluate_model():
    """학습된 Student 모델의 성능을 평가함."""
    return run_subprocess([sys.executable, os.path.join(MODEL_DIR, "eval.py"), "--mark_version", mark_version])

@timed_step
def run_step7_plot_results():
    """결과 시각화 (샘플 플롯)"""
    return run_subprocess([sys.executable, os.path.join(PRE_DIR, "plot_audio.py"), "--mark_version", mark_version])

# ===== 메인 실행 =====
if __name__ == "__main__":
    logging.info("="*50)
    logging.info("  소음 분류 전체 학습 파이프라인 (지식 증류 Ver.) 시작  ")
    logging.info("="*50)
    logging.info(f"모델 버전: {mark_version}")
    logging.info("현재 모델은 Teacher의 Soft Label과 실제 Hard Label을 함께 사용하는\n"
                 "지식 증류(Knowledge Distillation) 방식으로 학습됩니다.")

    steps = [
        run_step0_preprocess_audio,
        run_step1_generate_dataset_index,
        run_step2_teacher_model_train,
        run_step3_extract_hard_labels,
        run_step4_extract_soft_labels,
        run_step5_student_distillation_train,
        run_step6_evaluate_model,
        run_step7_plot_results
    ]

    for step in steps:
        return_code = step()
        if return_code != 0:
            logging.error(f"\n[CRITICAL ERROR] 파이프라인 실패: '{step.__name__}' 단계에서 오류 발생 (종료 코드: {return_code}).")
            logging.error("이후 단계를 생략하고 파이프라인 중단...")
            break

    logging.info("="*50)
    logging.info(f"[종료] 전체 파이프라인 완료. 로그 파일: {log_file_path}")
    logging.info("="*50)
    
