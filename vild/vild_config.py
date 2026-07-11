# vild_config.py

import torch
from sentence_transformers import SentenceTransformer
import os

SHARED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared_vild"))
if SHARED_DIR not in os.sys.path:
    os.sys.path.append(SHARED_DIR)

from prompt_bank import get_class_synonyms, get_prompt_templates, get_prompt_texts_for_class

class AudioViLDConfig:
    def __init__(self, mark_version="mark4.1"):  # change needed
        self.mark_version = mark_version

        # === 클래스 설정 ===
        # 기존 if-elif 구조 유지. 표기 중복 오타만 점검.
        if self.mark_version == "mark4.1":  # check needed
            self.classes = ["heavy_impact", "others"]
        elif self.mark_version == "mark4.2":
            self.classes = ["dragging", "others"]
        elif self.mark_version == "mark4.3":
            self.classes = ["construction", "others"]
        elif self.mark_version == "mark4.4":
            self.classes = ["machine_noise", "others"]
        elif self.mark_version == "mark4.5":
            self.classes = ["media_talking", "others"]
        elif self.mark_version == "mark4.6":
            self.classes = ["water_toilet", "others"]
        elif self.mark_version == "mark4.7":
            self.classes = ["water_shower", "others"]
        elif self.mark_version == "mark4.8":
            self.classes = ["dog_bark", "others"]
        else:
            raise ValueError(
                f"[Error] Unknown or unsupported mark_version: '{self.mark_version}'.\n"  # check needed
                f"지원되는 값: ['mark4.1', 'mark4.2', 'mark4.3', 'mark4.4', 'mark4.5', 'mark4.6', 'mark4.7', 'mark4.8']"
            )

        self.labeled_classes = self.classes
        self.unlabeled_class_identifier = "unlabeled"
        self.num_distinct_labeled_classes = len(self.labeled_classes)

        # === 오디오 파라미터 ===
        self.sample_rate = 16000
        # 세그먼트는 1초 단위로 처리. 전처리(fix_audio_length.py)는 파일을 특정 길이로 강제 통일하지 않고
        # 최소 3초(=5개 세그먼트를 뽑을 수 있는 최소 길이)만 보장(짧으면 패딩, 길어도 안 자름, mark5와 동일).
        # 세그먼트는 1초 × max_segments 사용.
        self.segment_duration = 1.0
        self.segment_samples = int(self.sample_rate * self.segment_duration)

        self.fft_size = 1024
        self.hop_length = 160
        self.n_mels = 64

        # === segment 단위 처리 ===
        self.segment_length = 101   # Mel spectrogram time frame 수
        self.segment_hop = 50       # Segment 간 stride
        self.max_segments = 5       # Teacher/Student 공통 최대 segment 수

        # === 모델 파라미터 ===
        self.embedding_dim = 384
        self.use_background_embedding = True
        self.background_embedding_weight = 0.1
        self.distill_branch_eval_weight = 0.5
        self.use_text_aligned_student = True
        self.use_feature_kd = True
        self.feature_kd_weight = 0.3
        self.feature_kd_loss_type = "cosine_l1"
        self.visual_view_type = "mel_delta"
        self.segment_selection_mode = "salient_topk"
        self.max_visual_segments = self.max_segments
        self.logit_temperature = 0.07
        self.segment_aggregation_mode = "confidence_saliency"
        self.segment_confidence_power = 2.0
        self.segment_saliency_power = 1.0
        self.others_confidence_threshold = 0.60
        self.others_margin_threshold = 0.08
        self.others_entropy_threshold = 0.72
        self.explain_topk_segments = 3
        self.save_visual_explanations = True

        # === 학습 파라미터 ===
        self.batch_size = 16
        self.num_epochs = 80 # 100에서 80으로 줄임
        self.learning_rate = 1e-4
        # [추가 2026-07-11] teacher_train.py의 EarlyStopping patience가 2로 하드코딩돼 있어
        # num_epochs=80까지 갈 수 있는데도 val loss가 한 번만 반등해도 5epoch 안에 조기종료되던 버그.
        # student(student_train_distillation.py의 EarlyStopping)와 동일하게 10으로 통일.
        self.teacher_patience = 10

        self.text_loss_weight = 1.0
        self.image_loss_weight = 1.0

        self.device = "cuda" if torch.cuda.is_available() else "cpu" # 코랩에서 gpu 쓰기

        """
        [Deprecated: data_wav 기반 단일 디렉터리]
        self.audio_dir = os.path.join("data_wav")  # mark_version 별 하위 폴더화 가능
        mark4.x에서는 data/{train,val,test} 구조를 사용하므로 위 필드는 참조되지 않거나 혼선을 줄 수 있음.
        """

        # [변경] 절대경로 기반 프로젝트 루트 계산 후 분할 데이터 경로 지정
        # vild/ 기준 상위가 프로젝트 루트라고 가정
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.audio_dir = os.path.join(self.project_root, "data")  # train/val/test 하위에 존재
        self.prompt_bank_path = os.path.join(SHARED_DIR, "resources", "prompt_bank.json")

        # === 내부 캐시 ===
        self._text_emb = None
        self._prompt_texts = None

        self.prompt_templates = get_prompt_templates(self.prompt_bank_path)
        self.class_synonyms = get_class_synonyms(self.prompt_bank_path)

    def get_class_index(self, class_name: str) -> int:
        if class_name in self.labeled_classes:
            return self.labeled_classes.index(class_name)
        elif class_name == self.unlabeled_class_identifier:
            return -1
        else:
            raise ValueError(
                f"[Config Error] '{class_name}'는 mark_version '{self.mark_version}'에 등록되지 않은 클래스입니다.\n"
                f"=> 현재 사용 가능한 클래스: {self.labeled_classes}"
            )

    def get_classes_for_text_prompts(self) -> list:
        return self.labeled_classes

    def get_target_label_map(self) -> dict:
        return {class_name: i for i, class_name in enumerate(self.get_classes_for_text_prompts())}

    @property
    def num_input_channels(self) -> int:
        if self.visual_view_type == "mel":
            return 1
        if self.visual_view_type == "mel_delta":
            return 3
        if self.visual_view_type == "mel_energy":
            return 2
        return 1

    def get_prompt_texts_for_class(self, class_name: str) -> list:
        return get_prompt_texts_for_class(class_name, self.prompt_bank_path)

    def get_prompt_texts(self) -> dict:
        if self._prompt_texts is None:
            self._prompt_texts = {
                class_name: self.get_prompt_texts_for_class(class_name)
                for class_name in self.get_classes_for_text_prompts()
            }
        return self._prompt_texts

    def get_class_text_embeddings(self) -> torch.Tensor:
        if self._text_emb is None:
            model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            aggregated = []
            for class_name in self.get_classes_for_text_prompts():
                prompts = self.get_prompt_texts_for_class(class_name)
                emb = model.encode(prompts, convert_to_tensor=True).to(self.device)
                aggregated.append(emb.mean(dim=0))
            self._text_emb = torch.stack(aggregated, dim=0)
        return self._text_emb
    
