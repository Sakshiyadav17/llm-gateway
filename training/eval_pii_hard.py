"""Hand-written hard cases. Each: (text, expected list of (type, value))."""
from gateway.pii import mask

CASES = [
    ("pan is abcde1234f", [("PAN", "abcde1234f")]),                 # lowercase
    ("Aadhaar 234567890123 given", []),                              # no-gap, likely invalid checksum
    ("call me at +91 (98765) 43210", [("PHONE", "+91 (98765) 43210")]),   # odd format
    ("number 98765-43210 and 9876543210", [("PHONE", "98765-43210"), ("PHONE", "9876543210")]),
    ("tracking 9876543210123", []),                                  # longer number, not a phone
    ("mail: first.last+tag@sub.company.co.in.", [("EMAIL", "first.last+tag@sub.company.co.in")]),
    ("GSTIN 27ABCDE1234F1Z5 and PAN ABCDE1234F", [("GST", "27ABCDE1234F1Z5"), ("PAN", "ABCDE1234F")]),
    ("मेरा नंबर 9876543210 है", [("PHONE", "9876543210")]),            # Hindi context
    ("IFSC SBIN0001234 branch", [("IFSC", "SBIN0001234")]),
    ("order ABCDE1234F-77", [("PAN", "ABCDE1234F")]),               # looks like PAN, is it?
        ("order number 234567890123 shipped", []),                       # 12 digits, likely fails checksum
    ("tracking id 7123456789012 delivered", []),                     # 13 digits, not a phone
    ("the word ABCDEfghij1234 is a code", []),                       # not a PAN
]

ok = 0
for text, expected in CASES:
    _, mp = mask(text)
    found = sorted((ph.split("_")[0][1:], v) for ph, v in mp.items())
    status = "PASS" if found == sorted(expected) else "FAIL"
    ok += status == "PASS"
    print(f"{status} | {text}\n       expected={expected}\n       found   ={found}")
print(f"\n{ok}/{len(CASES)} passed")
