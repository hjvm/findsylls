import re
from textgrid import TextGrid
from typing import Union, List, Tuple
from ..config.constants import SYLLABIC

def parse_textgrid_intervals(textgrid_input: Union[str, TextGrid], tier_idx: int) -> List[Tuple[float, float, str]]:
    if isinstance(textgrid_input, str):
        tg = TextGrid(strict=False); tg.read(textgrid_input)
    else:
        tg = textgrid_input
    tier = tg.tiers[tier_idx]
    return [(i.minTime, i.maxTime, i.mark) for i in tier.intervals]

def extract_vocalic_intervals(textgrid_input: Union[str, TextGrid], tier_idx: int) -> List[Tuple[float, float]]:
    intervals = parse_textgrid_intervals(textgrid_input, tier_idx)
    return [(start, end) for start, end, label in intervals if re.sub(r'[^a-zA-Z]', '', label.upper()) in SYLLABIC]

def extract_syllable_intervals(textgrid_input: Union[str, TextGrid], tier_idx: int, exclude_markers: List[str] = ["h#", "sil", "sp", "\{SIL\}", "_unknown"]) -> List[Tuple[float, float]]:
    intervals = parse_textgrid_intervals(textgrid_input, tier_idx)
    included = []; deleted = []
    for start, end, label in intervals:
        if label.strip() and not any(sub in label for sub in exclude_markers):
            included.append((start, end))
        else:
            deleted.append((start, end))
    return {"intervals": included, "deleted": deleted}

def generate_syllable_intervals(textgrid_path: str, phone_tier: Union[str, int]) -> List[Tuple[float, float]]:
    return []
