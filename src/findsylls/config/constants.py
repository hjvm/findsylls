## Evaluation methods.
EVAL_METHODS = [
    "nuclei",
    "syll_boundaries",
    "syll_spans",
    "word_boundaries",
    "word_spans",
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
