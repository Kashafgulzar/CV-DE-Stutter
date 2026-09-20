"""
phoneme_inventory.py

Single source of truth for phoneme classification.
"""

STRESS_CHARS = set('ˈˌ')

VOWELS = set("""a aɪ aɪə aʊ aː eə eɪ eː i iə iː o oː u uː yː øː œ ɑ ɑː ɑ̃ ɒ ɔ ɔø ɔɪ ɔː ə əʊ ɛ ɛɪ ɛː ɜ ɜː ɨ ɪ ʊ ʊɐ ʊə ʌ ʏ""".split())

# Documented in the stuttering-prolongation literature AND a steady-state
# continuant a TTS engine can lengthen just by extending duration, without
# the result turning into a different or unnatural sound.
#   - excludes the tap ɾ (single instantaneous contact, no "middle" to hold)
#   - excludes the trill r (rapid discrete contacts; stretching one duration
#     frame doesn't produce more trill cycles -> tends to buzz)
#   - excludes the glide j (too weak a constriction; stretching it usually
#     just sounds like an extended vowel, not a recognizable disfluency)
ELIGIBLE_CONSONANTS = set("m n ŋ ɲ l ɹ ʁ s f v z ʃ ʒ ç x h".split())
ELIGIBLE = VOWELS | ELIGIBLE_CONSONANTS

# Never prolongable: stops/affricates (zero duration to stretch), the tap,
# the trill, and the glide (see notes above). Kept here only so the
# tokenizer still recognizes and correctly segments these symbols.
OTHER_CONSONANTS = set("p b t d k ɡ g ɾ r j ʔ pf ts tʃ dʒ".split())

ALL_SYMBOLS = sorted(VOWELS | ELIGIBLE_CONSONANTS | OTHER_CONSONANTS, key=len, reverse=True)


def tokenize_phonemes(text):
    """Return list of {symbol,start,end} for real phonemes in `text`
    (words or a whole utterance - spaces and unrecognized characters come
    back as single-character 'unknown' entries so the index bookkeeping
    stays exact). Stress marks are skipped (left untouched in the string,
    not counted as phonemes)."""
    phonemes = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] in STRESS_CHARS:
            i += 1
            continue
        matched = None
        for sym in ALL_SYMBOLS:
            L = len(sym)
            if text[i:i + L] == sym:
                matched = sym
                break
        if matched:
            phonemes.append({'symbol': matched, 'start': i, 'end': i + len(matched)})
            i += len(matched)
        else:
            phonemes.append({'symbol': text[i], 'start': i, 'end': i + 1, 'unknown': True})
            i += 1
    return phonemes


def build_char_symbol_map(text):
    """Map each character index in `text` to the full phoneme symbol it
    belongs to (e.g. both characters of 'tʃ' map to 'tʃ', not 't' and 'ʃ'
    separately). Only real, recognized phonemes are included - spaces and
    unknown characters are absent from the map. This lets classification
    (continuant vs. not) operate on whole phoneme symbols instead of
    single characters, which matters for multi-character symbols like
    affricates and diphthongs."""
    m = {}
    for ph in tokenize_phonemes(text):
        if ph.get('unknown'):
            continue
        for ci in range(ph['start'], ph['end']):
            m[ci] = ph['symbol']
    return m