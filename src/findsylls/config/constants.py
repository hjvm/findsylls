## Evaluation method types (used for validation and filtering).
# Note: Actual evaluation keys are generated dynamically based on tier names,
# e.g., "syllable_boundaries", "word_spans", "phone_boundaries", etc.
EVAL_METHOD_TYPES = [
    "nuclei",      # Always uses phone tier for vocalic intervals
    "boundaries",  # Generic boundary evaluation (works on any tier)
    "spans",       # Generic span evaluation (works on any tier)
]

## Default tolerance for evaluation.
DEFAULT_TOLERANCE = 0.05  # seconds

## Vowel and syllabic consonant sets.
VOWELS = {
    # ARPABET vowels
    "AA", "AE", "AH", "AO", "AW", "AX", "AY",
    "EH", "ER", "EY",
    "IH", "IX", "IY",
    "OW", "OY",
    "UH", "UW",
    # Spanish vowels
    "A", "E", "I", "O", "U",
}
SYLLABIC_CONSONANTS = {"EL", "EM", "EN", "ENG"}
SYLLABIC = VOWELS.union(SYLLABIC_CONSONANTS)

## Language-specific vowel sets
# Kono (Mande language spoken in Sierra Leone)
# Duplicates are different Unicode representations of the same character...
KONO_VOWELS = {
    "a","aa","à","àà","àá","á","áà","áá","ã","ã́",
    "à","àà","àá","á","áá","ã","ɑ̀",
    "e","è","é","éè","è","èè","é",
    "i","ì","ìì","ìí","í","ì","ìí","í","íí","ɪ́",
    "ḿ","ń","ɱ́",
    "o","ò","ò̃","ó","õ","õ̀","ṍ","ò","ó","õ","õ̀",
    "u","ù","ú","u̟","ù","ùù","ú","ʉ́","ʌ́",
    "ɔ","ɔɔ","ɔ̀","ɔ́","ɔ́ɔ̀","ɔ́ɔ́","ɔ̟́",
    "ɛ","ɛ̀","ɛ̝̀","ɛ́","ɛ́ɛ́","ɛ̝́ɛ̝́","ɛ̝","ɛ̝̀",
}