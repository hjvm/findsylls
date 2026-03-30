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

def extract_vocalic_intervals(
    textgrid_input: Union[str, TextGrid], 
    tier_idx: int,
    syllabic_set: set = None
) -> List[Tuple[float, float]]:
    """Extract vocalic intervals from a TextGrid tier.
    
    Parameters
    ----------
    textgrid_input : str or TextGrid
        Path to TextGrid file or TextGrid object.
    tier_idx : int
        Zero-based tier index.
    syllabic_set : set, optional
        Set of syllabic phone symbols (vowels and syllabic consonants) to match.
        If None, uses the default SYLLABIC constant (ARPABET + Spanish vowels).
        Phone labels are normalized by stripping non-alphabetic characters and
        converting to uppercase before matching.
    
    Returns
    -------
    list of (start, end)
        Vocalic intervals in seconds.
    """
    if syllabic_set is None:
        syllabic_set = SYLLABIC
    intervals = parse_textgrid_intervals(textgrid_input, tier_idx)
    
    # Match logic: try exact match first (for IPA), then normalized (for ARPABET)
    result = []
    for start, end, label in intervals:
        label_stripped = label.strip()
        if not label_stripped:
            continue
        
        # Try exact match (case-sensitive) - for IPA symbols like ɔ, ɛ
        if label_stripped in syllabic_set:
            result.append((start, end))
            continue
        
        # Try lowercase exact match
        if label_stripped.lower() in syllabic_set:
            result.append((start, end))
            continue
        
        # Try uppercase exact match
        if label_stripped.upper() in syllabic_set:
            result.append((start, end))
            continue
        
        # Try normalized (strip non-ASCII-letters) - for ARPABET like "AH0" → "AH"
        normalized = re.sub(r'[^a-zA-Z]', '', label_stripped.upper())
        if normalized and normalized in syllabic_set:
            result.append((start, end))
    
    return result

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
