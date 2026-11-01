"""Phoneme inventory and grapheme-to-phoneme conversion.

Follows the convention of the Willett et al. (2023) speech neuroprosthesis release:
39 stress-less ARPAbet phonemes plus an inter-word ``SIL`` token, encoded as
``PHONE_DEF_SIL.index(p) + 1`` so that id 0 is free for the CTC blank (and padding).
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

PHONE_DEF = [
    "AA", "AE", "AH", "AO", "AW",
    "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G",
    "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW",
    "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V",
    "W", "Y", "Z", "ZH",
]
PHONE_DEF_SIL = PHONE_DEF + ["SIL"]

BLANK_ID = 0
N_PHONEMES = len(PHONE_DEF_SIL)   # 40 phoneme classes
N_CTC_CLASSES = N_PHONEMES + 1    # 41 = phonemes + CTC blank (the paper's d_bc)


def clean_transcript(text: str) -> str:
    """Normalise a prompt the way the phoneme labels were produced (letters, ' and - only)."""
    text = re.sub(r"[^a-zA-Z\- \']", "", str(text).strip())
    return text.replace("--", "").lower()


def text_to_phonemes(text: str, g2p=None) -> List[str]:
    """Sentence -> ARPAbet phonemes with ``SIL`` after every word (needs ``g2p_en``)."""
    if g2p is None:
        from g2p_en import G2p

        g2p = G2p()
    phonemes: List[str] = []
    for p in g2p(clean_transcript(text)):
        if p == " ":
            phonemes.append("SIL")
        p = re.sub(r"[0-9]", "", p)  # drop stress markers
        if re.match(r"[A-Z]+", p):
            phonemes.append(p)
    phonemes.append("SIL")  # so every word, including the last, ends with SIL
    return phonemes


def phonemes_to_ids(phonemes: Iterable[str]) -> List[int]:
    return [PHONE_DEF_SIL.index(p) + 1 for p in phonemes]


def ids_to_phonemes(ids: Sequence[int], length: Optional[int] = None) -> List[str]:
    """Inverse of :func:`phonemes_to_ids`; 0 (blank / padding) is skipped."""
    ids = ids[:length] if length is not None else ids
    return [PHONE_DEF_SIL[int(i) - 1] for i in ids if int(i) != BLANK_ID]
