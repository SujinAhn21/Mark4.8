# autoNor_utils.py

# getnerate_dataset_index.py에서 자동정규화 용으로 쓰임.

"""
[Deprecated] 단순 매핑(중복/변형 처리 미흡)
def normalize_label(label: str) -> str:
    label = label.lower().strip()

    label_map = {    # check needed
        "heavy_impact": ["heavy_impact", "heavy_impact"],
        "dragging": ["dragging", "dragging"],
        "construction": ["construction", "construction"],
        "machine_noise": ["machine_noise", "machine_noise"],
        "media_talking": ["media_talking", "media_talking"],
        "water_toilet": ["water_toilet", "water_toilet"],
        "water_shower": ["water_shower", "water_shower"],
        "dog_bark": ["dog_bark", "dog_bark"],
        "others": ["others", "others"],
    }

    for normalized, aliases in label_map.items():
        if label in aliases:
            return normalized

    print(f"[normalize_label Warning] Unknown label encountered: '{label}' -> fallback to original.")
    return label
"""

# [변경] 하이픈/공백 등 경미한 표기 변형을 흡수하고, 별칭 집합을 확장
def normalize_label(label: str) -> str:
    """
    주어진 라벨 문자열을 표준화된 라벨로 정규화.
    - 대소문자, 공백, 하이픈 변형 흡수
    - 예상 라벨 외에는 원문을 유지(침습 최소화)

    예: 'Heavy-Impact' -> 'heavy_impact'
    """
    s = (label or "").lower().strip()
    s = s.replace("-", "_").replace(" ", "_")

    aliases = {
        "heavy_impact": {"heavy_impact", "heavyimpact"},
        "dragging": {"dragging", "drag"},
        "construction": {"construction", "construct"},
        "machine_noise": {"machine_noise", "machinenoise", "appliance_noise"},
        "media_talking": {"media_talking", "talking", "speech", "tv"},
        "water_toilet": {"water_toilet", "toilet", "flush", "flushing"},
        "water_shower": {"water_shower", "shower"},
        "dog_bark": {"dog_bark", "dog", "bark", "dogbark"},
        "others": {"others", "other"},
    }

    for norm, aset in aliases.items():
        if s in aset:
            return norm

    print(f"[normalize_label Warning] Unknown label encountered: '{label}' -> fallback to original.")
    return s
    