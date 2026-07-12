"""
shared_vild/prompt_bank.py
Mark4.x / Mark5.0 공통 Prompt Bank 모듈.
prompt_bank.json에서 템플릿과 클래스 동의어를 로드하고,
각 클래스에 대한 prompt 텍스트 목록을 반환한다.
"""

import json
import os

_DEFAULT_TEMPLATES = [
    "a sound of {label} in the room",
    "the audio of {label}",
    "an indoor sound that resembles {label}",
    "a recording of {label}",
]

_DEFAULT_SYNONYMS = {
    "heavy_impact": ["heavy impact", "strong thud", "impact on floor"],
    "dragging": ["dragging", "scraping drag", "object dragging on floor"],
    "construction": ["construction", "construction work", "renovation noise"],
    "machine_noise": ["machine noise", "mechanical humming", "appliance machine sound"],
    "media_talking": ["media talking", "tv speech", "speaker talking audio"],
    "water_toilet": ["toilet water", "toilet flush", "bathroom flush sound"],
    "water_shower": ["shower water", "shower running", "bathroom shower sound"],
    "dog_bark": ["dog bark", "barking dog", "canine bark"],
    "others": ["other sound", "background noise", "non target sound"],
    "dummy_label": ["other sound", "background noise", "non target sound"],
}

_cache = {}


def _load_bank(path):
    if path in _cache:
        return _cache[path]
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _cache[path] = data
        return data
    _cache[path] = {}
    return {}


def get_prompt_templates(path=None):
    data = _load_bank(path)
    return data.get("prompt_templates", _DEFAULT_TEMPLATES)


def get_class_synonyms(path=None):
    data = _load_bank(path)
    return data.get("class_synonyms", _DEFAULT_SYNONYMS)


def get_prompt_texts_for_class(class_name, path=None):
    templates = get_prompt_templates(path)
    synonyms_map = get_class_synonyms(path)
    synonyms = synonyms_map.get(class_name, [class_name.replace("_", " ")])
    prompts = []
    for synonym in synonyms:
        for template in templates:
            prompts.append(template.format(label=synonym))
    return prompts
