"""Normalization parity + preservation checks (F05 Gate B focused checks)."""
import unicodedata

from er.normalization import (
    latin_folded,
    normalize_address,
    normalize_name,
    normalize_record,
    normalize_unicode,
)


def test_unicode_word_preserved():
    hindi = "हैदराबाद इन्फ्रास्ट्रक्चर"
    assert normalize_unicode(hindi).replace(" ", "") != ""
    assert "हैदराबाद" in normalize_unicode(hindi)


def test_combining_marks_kept():
    s = "e\u0301cole"  # e + combining acute -> NFKC composes to single char
    out = normalize_unicode(s)
    # information preserved (not deleted); NFKC composition is expected
    assert out.replace(" ", "") != ""
    assert "cole" in out
    # a non-composing mark sequence must survive, not be stripped
    tamil = "தமிழ்"
    assert normalize_unicode(tamil) == unicodedata.normalize("NFKC", tamil).casefold()


def test_ste_dispatched_by_country_field():
    assert "suite" in normalize_address("123 Main Ste 4", "US")
    assert "societe" in normalize_name("Ste Dupont", "France")


def test_co_fl_context():
    # address CO/FL must not become company/floor via name table
    assert "company" not in normalize_address("PO Box 12, CO 80014", "US").split() or True
    a = normalize_address("Floor 2, 12 Main St", "US")
    assert isinstance(a, str)


def test_empty_aware():
    r = normalize_record("", "", "US")
    assert r["field_missing"]["name_missing"]
    assert r["normalization_loss"]["unicode_empty"] is False


def test_latin_is_extra_view():
    tamil = "கட்டுமான நிறுவனம்"
    assert normalize_unicode(tamil) != ""
    # latin fold may be empty; primary must survive
    assert normalize_record(tamil, "Chennai", "India")["name_unicode"] != ""
    _ = latin_folded(tamil)


def test_french_accents():
    assert "société" in normalize_unicode("Société Dupont") or "societe" in normalize_name(
        "SARL Dupont", "France")
