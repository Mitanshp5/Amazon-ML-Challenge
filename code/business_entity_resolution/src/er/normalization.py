"""Unicode-preserving, field- and country-aware normalization (F05 Phase B).

P0 defects fixed vs notebooks/entity_resolution_local.ipynb cell 10:
- normalize_ascii(encode-ignore) deleted Indic names -> primary views are
  Unicode (NFKC + casefold, keep L/M/N). Latin folding is an ADDITIONAL view.
- [^\\w\\s] removed combining marks -> we keep Unicode categories L/M/N and
  collapse separators without breaking graphemes.
- Merged ABBR dict overrode US 'ste'->suite with FR 'ste'->societe, and
  CO->company / FL->floor regardless of field -> dispatch by country+field.
- LEGAL_SUFFIX=set(ABBR) mixed road/state abbreviations with legal forms ->
  separate legal-form vs address-abbreviation tables.

Views per record (see plan 9.1):
  name_unicode / address_unicode : primary, information-preserving
  name_latin / address_latin     : accent-folded extra view (never replaces primary)
  name_core                      : legal-suffix-reduced view (feature, not filter)
  compact/domain/acronym/number parses retained alongside raw strings.
"""
from __future__ import annotations

import re
import unicodedata

# --- legal forms (name field only) ---
LEGAL_BY_COUNTRY = {
    "US": {
        "corp": "corporation", "inc": "incorporated", "corp.": "corporation",
        "llc": "limited liability company", "llp": "limited liability partnership",
        "co": "company", "ltd": "limited", "pvt": "private",
        "mfg": "manufacturing", "ent": "enterprises",
        "dba": "doing business as",
    },
    "India": {
        "pvt": "private", "ltd": "limited",
        "mfg": "manufacturing", "ent": "enterprises",
        "co": "company",
        "dba": "doing business as",
    },
    "France": {
        "sarl": "societe responsabilite limitee",
        "sas": "societe actions simplifiee",
        "sasu": "societe actions simplifiee unipersonnelle",
        "sa": "societe anonyme", "eurl": "entreprise unipersonnelle",
        "ets": "etablissements", "ste": "societe",
    },
    "__default__": {
        "pvt": "private", "ltd": "limited", "co": "company",
        "corp": "corporation", "inc": "incorporated",
        "dba": "doing business as",
    },
}

# --- street/unit abbreviations (address field only) ---
ADDR_ABBR_BY_COUNTRY = {
    "US": {
        "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
        "ste": "suite", "apt": "apartment", "dr": "drive", "hwy": "highway",
        "bldg": "building",
    },
    "India": {
        "rd": "road", "st": "street", "nagar": "nagar", "marg": "marg",
        "bldg": "building",
    },
    "France": {
        "rue": "rue", "bd": "boulevard", "av": "avenue", "pl": "place",
        "imp": "impasse", "cedex": "cedex", "ste": "suite",
    },
    "__default__": {"rd": "road", "st": "street", "ave": "avenue", "bldg": "building"},
}

LEGAL_SUFFIX_TOKENS = {
    "corporation", "incorporated", "private", "limited", "company",
    "limited liability company", "limited liability partnership",
    "societe", "societe responsabilite limitee", "societe actions simplifiee",
    "societe anonyme", "entreprise unipersonnelle", "etablissements",
    "gmbh",
}

_MISSING_LITERALS = {"", "nan", "none", "null", "-", "n/a", "na"}
_PIN_RE = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")
_HOUSE_RE = re.compile(r"\b\d{1,6}[A-Za-z]?\b")
_UNIT_RE = re.compile(r"\b(?:suite|ste|apt|unit|floor|fl|bldg|building)\s+([A-Za-z0-9\-/]+)", re.IGNORECASE)


def _keep_lmn_space(s: str) -> str:
    """Keep letters (L), marks (M), numbers (N); collapse the rest to spaces.

    Preserves combining marks (M*) that the old [^\\w\\s] pattern deleted.
    """
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "M", "N") or ch == " ":
            out.append(ch)
        elif ch in ("&",):
            out.append(" and ")
        else:
            out.append(" ")
    return "".join(out)


def _clean_unicode_raw(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = s.casefold()
    s = s.replace("&", " and ")
    # M/s, DBA, dotted initialisms: normalize without deleting content
    s = re.sub(r"^(m\s*/\s*s|m\s*\.\s*s)\.?\s+", "", s)
    s = re.sub(r"\bd\s*/\s*b\s*/\s*a\b", "dba", s)
    s = re.sub(r"\b([a-z])\.(?=[a-z]\b|\s|[a-z]\.|$)", r"\1", s)
    s = _keep_lmn_space(s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_unicode(s: str) -> str:
    """Primary Unicode view. Never empty a non-empty Indic/French string."""
    return _clean_unicode_raw(s)


def latin_folded(s: str) -> str:
    """Additional Latin view: NFKD accent strip. Must NOT replace primary."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.encode("ascii", "ignore").decode()
    return _clean_unicode_raw(s)


def expand_tokens(text: str, table: dict) -> str:
    return " ".join(table.get(t, t) for t in text.split())


def normalize_name(s: str, country: str = "__default__") -> str:
    base = normalize_unicode(s)
    table = LEGAL_BY_COUNTRY.get(country, LEGAL_BY_COUNTRY["__default__"])
    return expand_tokens(base, table)


def normalize_address(s: str, country: str = "__default__") -> str:
    base = normalize_unicode(s)
    table = ADDR_ABBR_BY_COUNTRY.get(country, ADDR_ABBR_BY_COUNTRY["__default__"])
    return expand_tokens(base, table)


def normalize_name_latin(s: str, country: str = "__default__") -> str:
    base = latin_folded(s)
    table = LEGAL_BY_COUNTRY.get(country, LEGAL_BY_COUNTRY["__default__"])
    return expand_tokens(base, table)


def normalize_address_latin(s: str, country: str = "__default__") -> str:
    base = latin_folded(s)
    table = ADDR_ABBR_BY_COUNTRY.get(country, ADDR_ABBR_BY_COUNTRY["__default__"])
    return expand_tokens(base, table)


def name_core(name_unicode_norm: str) -> str:
    """Suffix-reduced view: strip trailing legal-form tokens. Feature only."""
    toks = name_unicode_norm.split()
    changed = True
    while changed and toks:
        changed = False
        for n in (4, 3, 2, 1):
            if len(toks) >= n and " ".join(toks[-n:]) in LEGAL_SUFFIX_TOKENS:
                toks = toks[:-n]
                changed = True
                break
    return " ".join(toks)


def is_missing_field(raw: str) -> bool:
    return str(raw).strip().casefold() in _MISSING_LITERALS


def normalized_empty(view: str) -> bool:
    return len(view.strip()) == 0


def script_flags(text: str) -> dict:
    has_latin = has_nonlatin = has_mark = False
    for ch in text:
        cat = unicodedata.category(ch)
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = ""
        if cat[0] == "L":
            if "LATIN" in name:
                has_latin = True
            else:
                has_nonlatin = True
        if cat[0] == "M":
            has_mark = True
    return {"has_latin": has_latin, "has_nonlatin": has_nonlatin, "has_mark": has_mark}


def parse_numbers(address_raw: str) -> dict:
    addr = str(address_raw)
    pins = _PIN_RE.findall(addr)
    houses = _HOUSE_RE.findall(addr)
    units = _UNIT_RE.findall(addr)
    return {
        "postal_candidates": pins,
        "pin_confident": len(pins) == 1,
        "house_tokens": houses,
        "unit_tokens": units,
        "raw": addr,
    }


def normalize_record(name_raw: str, address_raw: str, country_raw: str) -> dict:
    country_key = str(country_raw).strip() or "__default__"
    if country_key not in ("US", "India", "France"):
        country_key = "__default__" if country_key == "__default__" else country_key
        policy_country = "__default__"
    else:
        policy_country = country_key
    nu = normalize_name(name_raw, policy_country)
    au = normalize_address(address_raw, policy_country)
    nl = normalize_name_latin(name_raw, policy_country)
    al = normalize_address_latin(address_raw, policy_country)
    return {
        "name_raw": str(name_raw),
        "address_raw": str(address_raw),
        "country_raw": str(country_raw),
        "country_key": country_key,
        "name_unicode": nu,
        "address_unicode": au,
        "name_latin": nl,
        "address_latin": al,
        "name_core": name_core(nu),
        "name_compact": nu.replace(" ", ""),
        "field_missing": {
            "address_missing": is_missing_field(address_raw),
            "name_missing": is_missing_field(name_raw),
        },
        "normalization_loss": {
            "unicode_empty": normalized_empty(nu) and not is_missing_field(name_raw),
            "latin_empty": normalized_empty(nl) and bool(nu.strip()),
        },
        "script": {
            "name": script_flags(str(name_raw)),
            "address": script_flags(str(address_raw)),
        },
        "numbers": parse_numbers(address_raw),
    }
