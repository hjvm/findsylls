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
