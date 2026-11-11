from gateway.pii import mask, restore, verhoeff_valid


def test_pan_masked():
    m, mp = mask("Ravi's PAN is ABCDE1234F")
    assert "<PAN_1>" in m and "ABCDE1234F" not in m


def test_restore_roundtrip():
    t = "mail a@b.com or call +91 98765 43210"
    m, mp = mask(t)
    assert restore(m, mp) == t


def test_any_12_digit_id_is_masked():
    m, _ = mask("number 2345 6789 0123")
    assert "<AADHAAR_1>" in m


def test_verhoeff_known_valid():
    assert verhoeff_valid("2363")  # classic Verhoeff test vector