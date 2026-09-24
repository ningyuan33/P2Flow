"""Standard TIMIT 39-phone mapping.

This project uses the standard TIMIT 39-phone target space:

    61 original TIMIT phones -> 39 total labels

Important:
    The 39 labels already include `sil`.
    So this is NOT 39 phones + silence.
    It is 39 total labels including silence.

Common convention:
    - Collapse allophonic and closure labels.
    - Map silence-like labels to `sil`.
    - Keep `dx` as a final phone class.
    - Discard `q`.
"""

from __future__ import annotations

# Standard TIMIT 39-phone set, including silence.
# Note: dx is included. This gives exactly 39 total labels.
PHONES_39 = [
    "aa", "ae", "ah", "aw", "ay",
    "b", "ch", "d", "dh", "dx",
    "eh", "er", "ey", "f", "g",
    "hh", "ih", "iy", "jh", "k",
    "l", "m", "n", "ng", "ow",
    "oy", "p", "r", "s", "sh",
    "sil", "t", "th", "uh", "uw",
    "v", "w", "y", "z",
]

PHONE_TO_ID_39 = {p: i for i, p in enumerate(PHONES_39)}
ID_TO_PHONE_39 = {i: p for p, i in PHONE_TO_ID_39.items()}

SIL_ID = PHONE_TO_ID_39["sil"]

# PyTorch CrossEntropyLoss ignore_index.
PAD_ID = -100

# Standard TIMIT 61 -> 39 mapping.
# `q` is discarded by mapping it to None.
TIMIT_TO_39: dict[str, str | None] = {
    # Stops
    "b": "b",
    "d": "d",
    "g": "g",
    "p": "p",
    "t": "t",
    "k": "k",

    # Flap
    "dx": "dx",

    # Affricates
    "jh": "jh",
    "ch": "ch",

    # Fricatives
    "s": "s",
    "sh": "sh",
    "z": "z",
    "zh": "sh",
    "f": "f",
    "th": "th",
    "v": "v",
    "dh": "dh",

    # Nasals and nasal variants
    "m": "m",
    "n": "n",
    "ng": "ng",
    "em": "m",
    "en": "n",
    "eng": "ng",
    "nx": "n",

    # Semivowels / liquids / glides
    "l": "l",
    "r": "r",
    "w": "w",
    "y": "y",
    "hh": "hh",
    "hv": "hh",
    "el": "l",

    # Vowels and vowel variants
    "iy": "iy",
    "ih": "ih",
    "eh": "eh",
    "ey": "ey",
    "ae": "ae",
    "aa": "aa",
    "ao": "aa",
    "aw": "aw",
    "ay": "ay",
    "ah": "ah",
    "oy": "oy",
    "ow": "ow",
    "uh": "uh",
    "uw": "uw",
    "ux": "uw",
    "er": "er",
    "ax": "ah",
    "ix": "ih",
    "axr": "er",
    "ax-h": "ah",

    # Closures / silence / special labels
    "bcl": "sil",
    "dcl": "sil",
    "gcl": "sil",
    "pcl": "sil",
    "tcl": "sil",
    "kcl": "sil",
    "epi": "sil",
    "pau": "sil",
    "h#": "sil",

    # Glottal stop is discarded in standard TIMIT evaluation.
    "q": None,
}


def validate_mapping() -> None:
    values = {v for v in TIMIT_TO_39.values() if v is not None}
    expected = set(PHONES_39)

    missing = expected - values
    extra = values - expected

    if extra:
        raise ValueError(f"Mapping produced phones outside PHONES_39: {sorted(extra)}")

    if missing:
        print(
            "Warning: these phones are in PHONES_39 but not produced by mapping: "
            f"{sorted(missing)}"
        )


def map_timit_phone(phone: str) -> str | None:
    """Map a raw TIMIT phone to the standard 39-phone set.

    Returns:
        mapped phone string, or None if the phone should be discarded.
    """
    phone = phone.lower()

    if phone not in TIMIT_TO_39:
        raise ValueError(f"Unknown TIMIT phone: {phone}")

    return TIMIT_TO_39[phone]