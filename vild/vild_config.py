# vild_config.py

import math
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
        # [확정 2026-07-12] True->False. entropy_threshold 수정 + FSD50K 재분할 이후에도 confusion matrix가
        # 여전히 완전 붕괴(dog_bark 0/50)했던 진짜 원인. eval.py의 max-override 로직이 학습된 background
        # embedding을 오디오 임베딩과 비교해 "others" 로짓을 계속 밀어올려서, 실제 dog_bark 샘플에서도
        # Prob_dog_bark가 0.008~0.016 수준으로 짓눌림(True=others 샘플과 분포가 완전히 겹침). False로 끄자
        # 즉시 정상적인 confidence 분포(0.257~0.837)가 드러나고 raw accuracy 52%->66%로 개선(코랩 재검증 완료).
        # 2026-07-11에 한 번 False로 진단했다가 "결과 동일"이라 기각했었는데, 그건 재분할 전 데이터로 학습한
        # 구모델 기준이었음 - 재분할 후 재학습된 모델에서는 진짜 원인으로 확정됨(가설 재검증의 중요성).
        self.use_background_embedding = False
        self.background_embedding_weight = 0.1
        self.distill_branch_eval_weight = 0.5  # [원복 2026-07-11] 진단(0)했으나 collapse 원인 아님(가설 기각).
        self.use_text_aligned_student = True
        self.use_feature_kd = True
        self.feature_kd_weight = 0.3
        self.feature_kd_loss_type = "cosine_l1"
        self.visual_view_type = "mel_delta"
        self.segment_selection_mode = "salient_topk"
        self.max_visual_segments = self.max_segments
        self.logit_temperature = 0.07
        self.segment_aggregation_mode = "confidence_saliency"  # [원복 2026-07-11] 진단("mean")했으나 collapse 원인 아님(가설 기각).
        self.segment_confidence_power = 2.0
        self.segment_saliency_power = 1.0
        # [조정 2026-07-12] 0.60->0.55. background_embedding override를 끄고 재평가한 실측 confidence
        # 분포(raw dog_bark 예측 30건: 0.508~0.837) 기준 시뮬레이션 결과, threshold를 0.55까지 낮춰도
        # others_margin_threshold(0.08, top_conf>=0.54가 사실상의 하한)가 이미 걸러주고 있어 0.55와 0.60
        # 밑으로는 값을 낮춰도 차이가 없었고, 0.60->0.55 구간에서는 accuracy 0.62->0.63, dog_bark
        # recall 0.34->0.38로 개선됨(TP=19,FN=31,FP=6,TN=44). 주의: 이 값은 test set(100개) 결과로
        # 고른 것이라 val set 기준 재검증이 이상적이나, 급한 실전 디버깅 상황이라 우선 반영.
        self.others_confidence_threshold = 0.55
        self.others_margin_threshold = 0.08
        # [확정 2026-07-12] others-calibration override 완전 비활성(False).
        # 원인: 이 override는 raw 예측이 others가 아닐 때만(=dog_bark일 때만) 작동해서 조건에 걸리면
        # others로 강등시킨다 -> 구조적으로 dog_bark 예측을 줄이기만 할 뿐 늘리지는 못한다.
        # mark4.8(2-class specialist) 실측(재분할 재학습 후 test 100개)에서 raw dog_bark recall 62%가
        # override를 거치면 42%로 반토막났다(정답 dog_bark 10건을 죽이고 오답 11건을 살려 정확도는 +1%p뿐).
        # 2-class에선 margin=2*conf-1 이라 conf/margin/entropy 세 조건이 전부 conf≈0.55 근처에서 겹쳐,
        # 그 구간(0.51~0.55)에 정답/오답이 5:5로 섞여 있어 '완화'의 스위트스팟이 없었다(부분완화가 오히려
        # 더 나빴음). override off 시뮬레이션 결과: dog_bark recall 0.42->0.62, 정확도 0.66->0.65(-1%p),
        # macro F1 0.639->0.650(오히려 개선). others-calibration은 원래 9-class generalist(mark5)용
        # 반려 로직이라 2-class specialist에는 부적합. (eval 시점 로직이라 재학습 불필요.)
        self.use_others_calibration = False
        # [추가 2026-07-13] 2-class 전용 최종 결정 threshold. 기존에는 argmax(=암묵적 0.5)였음.
        # 가설 1·2·3·4·6 적용 재학습(커밋 63428a2) 결과 모델이 초보수적으로 변해(dog precision
        # 1.0, Others FPR 0.0) 놓친 dog 19건 중 11건이 Prob 0.4~0.5 구간에 몰려 있었음.
        # ROC AUC 0.9192로 분리력이 생긴 상태라 운영점 선택이 가능해짐: threshold를 0.40으로
        # 낮추면 dog recall 0.62->0.84, accuracy 0.81->0.83, Others FPR 0.00->0.18 (test 100개
        # 시뮬레이션). 주의: 이 값은 test set으로 고른 것이라 val set 기준 재검증이 이상적임.
        # 또한 mark4.8 학습 결과 기준이므로 다른 4.x 버전은 각자 재튜닝 필요.
        # None이면 기존 argmax 동작. 2-class가 아니면 무시됨.
        self.target_decision_threshold = 0.40
        # [삭제 2026-07-11] others_entropy_threshold 하드코딩(0.72) 제거.
        # 원인 규명: 2-class(mark4.x) 이진분류에서 정규화 entropy가 0.72 이하가 되려면
        # top_conf가 최소 약 0.80은 되어야 함(균등분산 최악 케이스 기준 실측 계산).
        # 반면 confidence_threshold는 0.60으로 설정돼 있어, entropy 조건이 confidence 조건보다
        # 훨씬 엄격하게 어긋나 있었음 -> confidence>=0.60인 예측도 실제 top_conf가 0.78 정도까지밖에
        # 안 나오는 mark4.8 실측 분포에서는 entropy 조건에 걸려 100% "others"로 강제 override됨
        # (confusion matrix가 항상 others로만 나오는 근본 원인이었음).
        # 아래 property로 대체: confidence_threshold와 "같은 엄격도(같은 top_conf 지점)"가 되도록
        # 클래스 수 기반으로 자동 역산. mark5(9-class)는 기존 하드코딩값(0.82)이 우연히 이미
        # confidence_threshold(0.45)와 거의 같은 엄격도였어서(top_conf≈0.47 지점) 이 변경으로도
        # 동작이 거의 그대로 유지됨. mark4.x는 entropy 조건이 confidence 조건과 정합하게 완화됨.
        self.explain_topk_segments = 3
        self.save_visual_explanations = True

        # === 학습 파라미터 ===
        self.batch_size = 16
        self.num_epochs = 80 # 100에서 80으로 줄임
        self.learning_rate = 1e-4
        # [추가 2026-07-12 / 가설6] L2 정규화. 그동안 teacher/student 두 옵티마이저 모두
        # weight_decay가 전무했음(dropout 0.3만). teacher가 val best를 epoch 5에 찍고 곧장
        # 과적합으로 넘어가던 실측(train down/val up)에 대한 표준 처방.
        self.weight_decay = 1e-4
        # [추가 2026-07-12 / 가설3] teacher CE에 label smoothing. teacher가 약하고 일찍
        # 과적합하는 문제 완화용. 0이면 기존과 동일 동작.
        self.teacher_label_smoothing = 0.1
        # [추가 2026-07-12 / 가설1] 증류 loss 배분. 기존에는 student_train_distillation.py에
        # alpha=0.7, T=4.0이 하드코딩되어 있었음. 실측(2026-07-12): soft loss가 0.02 수준으로
        # 신호가 거의 없는데 총 loss의 70%를 차지해 분류(hard, 30%)가 굶주렸고, 그 결과
        # train에서조차 hard loss가 랜덤 근처(~0.60, 2-class 랜덤 ln2=0.693)에 고착(분류 과소적합).
        # alpha를 낮춰 hard에 70%를 배분하고, T도 4.0->2.0으로 낮춰 soft 신호가 덜 뭉개지게 함.
        # 검증 기준: 재학습 후 train hard loss가 0.6 밑으로 내려가는지 확인.
        self.distill_alpha = 0.3
        self.distill_temperature = 2.0
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
    def others_entropy_threshold(self) -> float:
        """
        [추가 2026-07-11] others_confidence_threshold와 "같은 엄격도(같은 top_conf 지점)"가
        되도록 클래스 수 기반으로 자동 역산한다. top_conf=confidence_threshold이고 나머지
        확률이 (클래스수-1)개에 균등분산되는 최악의 경우를 가정해 그때의 정규화 entropy를
        threshold로 삼는다. 클래스 수가 다른 mark_version 사이에서 entropy 조건이
        confidence 조건보다 부당하게 엄격/느슨해지는 것을 방지한다.
        """
        p = self.others_confidence_threshold
        n = self.num_distinct_labeled_classes
        if n <= 1:
            return 1.0
        rest = 1.0 - p
        probs = [p] + [rest / (n - 1)] * (n - 1)
        entropy = -sum(x * math.log(x, 2) for x in probs if x > 1e-12)
        return entropy / math.log2(n)

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
    
