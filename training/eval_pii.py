"""Evaluate PII masking on synthetic sentences: precision/recall per type."""
import random
import string
from gateway.pii import mask, _D, _P

random.seed(42)


def make_aadhaar():
    while True:
        base = str(random.randint(2, 9)) + "".join(random.choices(string.digits, k=10))
        c = 0
        for i, ch in enumerate(reversed(base + "0")):
            c = _D[c][_P[i % 8][int(ch)]]
        for chk in range(10):
            from gateway.pii import verhoeff_valid
            if verhoeff_valid(base + str(chk)):
                n = base + str(chk)
                return f"{n[:4]} {n[4:8]} {n[8:]}"


def make_pan():
    return ("".join(random.choices(string.ascii_uppercase, k=5)) +
            "".join(random.choices(string.digits, k=4)) + random.choice(string.ascii_uppercase))


def make_phone():
    n = random.choice("6789") + "".join(random.choices(string.digits, k=9))
    return random.choice([n, f"+91 {n[:5]} {n[5:]}", f"+91-{n}"])


def make_email():
    return "".join(random.choices(string.ascii_lowercase, k=7)) + random.choice(["@gmail.com", "@company.in", "@mail.co.in"])


def make_ifsc():
    return "".join(random.choices(string.ascii_uppercase, k=4)) + "0" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def make_gst():
    return (f"{random.randint(1, 37):02d}" + make_pan() + random.choice("123456789") + "Z" + random.choice("ABC123"))


MAKERS = {"AADHAAR": make_aadhaar, "PAN": make_pan, "PHONE": make_phone,
          "EMAIL": make_email, "IFSC": make_ifsc, "GST": make_gst}
TEMPLATES = [
    "Please update the record, the value is {x} for the customer.",
    "My details: {x}. Kindly verify.",
    "Customer shared {x} over the call yesterday.",
    "Mera number {x} hai, please check.",
    "Send the invoice reference {x} to accounts.",
]
DECOYS = ["The order total is 1234567890123 units.", "Invoice 2024-11-0987 was paid.",
          "Employee ID 123456 joined in 2021.", "Call extension 4455 for support.",
          "Pincode 110001 and ticket 998877665544 are open.",
          "Revenue was 500000 rupees in Q3."]

tp = {k: 0 for k in MAKERS}; fn = {k: 0 for k in MAKERS}; fp = {k: 0 for k in MAKERS}
N = 25
for label, fn_make in MAKERS.items():
    for _ in range(N):
        val = fn_make()
        text = random.choice(TEMPLATES).format(x=val)
        masked, mp = mask(text)
        hit = [ph for ph, v in mp.items() if v == val and ph.startswith(f"<{label}_")]
        if hit:
            tp[label] += 1
        else:
            fn[label] += 1
        for ph in mp:  # wrong-type labels count as false positives
            if not ph.startswith(f"<{label}_") and mp[ph] == val:
                fp[ph.split("_")[0][1:]] += 1

decoy_fp = 0
for d in DECOYS * 5:
    _, mp = mask(d)
    for ph in mp:
        fp[ph.split("_")[0][1:]] += 1
        decoy_fp += 1

print(f"{'Type':<9}{'TP':>4}{'FN':>4}{'FP':>4}{'Precision':>11}{'Recall':>8}")
for k in MAKERS:
    p = tp[k] / (tp[k] + fp[k]) if tp[k] + fp[k] else 0
    r = tp[k] / (tp[k] + fn[k]) if tp[k] + fn[k] else 0
    print(f"{k:<9}{tp[k]:>4}{fn[k]:>4}{fp[k]:>4}{p:>11.2f}{r:>8.2f}")
print(f"\nDecoy sentences wrongly masked (false positives): {decoy_fp} of {len(DECOYS) * 5}")