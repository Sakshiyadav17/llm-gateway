"""Regex-based PII masking for Indian data. Aadhaar is validated with the Verhoeff checksum."""
import re

# ---- Verhoeff checksum (used by Aadhaar) ----
_D = [
    [0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],
    [3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],
    [9,8,7,6,5,4,3,2,1,0],
]
_P = [
    [0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],
    [8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8],
]


def verhoeff_valid(num: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def _aadhaar_ok(m: str) -> bool:
    digits = re.sub(r"\D", "", m)
    return len(digits) == 12 and digits[0] in "23456789" and verhoeff_valid(digits)


# ---- Patterns (order matters: more specific first) ----
PATTERNS = [
    ("EMAIL",   re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), None),
    ("GST",     re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b"), None),
    ("PAN",     re.compile(r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b"), None),
    ("IFSC",    re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), None),
    ("AADHAAR", re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)"), _aadhaar_ok),
    ("PHONE",   re.compile(r"(?<!\d)(?:\+91[\s-]?|91[\s-]?|0)?\(?[6-9]\d{4}\)?[\s-]?\d{5}(?!\d)"), None),
]


def mask(text: str):
    """Return (masked_text, mapping). mapping: placeholder -> original value."""
    mapping, counters, spans = {}, {}, []
    for label, rx, validator in PATTERNS:
        for m in rx.finditer(text):
            s, e = m.span()
            if any(s < pe and e > ps for ps, pe, _, _ in spans):
                continue  # overlaps something already matched
            if validator and not validator(m.group()):
                continue
            spans.append((s, e, label, m.group()))
    spans.sort()
    out, last = [], 0
    for s, e, label, val in spans:
        counters[label] = counters.get(label, 0) + 1
        ph = f"<{label}_{counters[label]}>"
        mapping[ph] = val
        out.append(text[last:s]); out.append(ph)
        last = e
    out.append(text[last:])
    return "".join(out), mapping


def restore(text: str, mapping: dict) -> str:
    for ph, val in mapping.items():
        text = text.replace(ph, val)
    return text


if __name__ == "__main__":
    t = "Ravi's PAN is ABCDE1234F, mail ravi@x.com, phone +91 98765 43210"
    m, mp = mask(t)
    print(m); print(mp); print(restore(m, mp))