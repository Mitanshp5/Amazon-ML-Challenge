"""Pair features v2: schema-pinned, missingness/conflict aware (F05 12.2).

Fixes cell-24 P1s:
- _char3('') -> {''} made every empty name identical -> empty/short-aware
  ngrams + explicit availability flags; empty equality is NOT positive evidence.
- number intersection treated any shared postal/unit/house alike -> parse
  house/unit/postal separately with equality AND conflict features.
- Column_0.. names -> real feature names asserted at inference.

Country-match inside same-country partitions is constant: kept as a
diagnostic, must not be the transfer signal.
"""
from __future__ import annotations

import re

import numpy as np
from rapidfuzz import fuzz

_HOUSE_RE = re.compile(r"\b\d{1,6}[A-Za-z]?\b")

FEATURES = [
    "tfidf_max", "n_channels",
    "name_wratio", "name_set", "name_sort", "name_partial",
    "name_char_jac", "name_len_ratio", "name_exact_nonempty",
    "name_core_jac",
    "addr_sort", "addr_set", "addr_partial", "addr_word_jac",
    "addr_present_both",
    "house_equal", "house_conflict", "unit_equal", "unit_conflict",
    "pin_equal", "pin_conflict",
    "source_is_s2",
    "rrf_best",
]


def char_ngrams(s: str, n: int = 3) -> set:
    s = s.replace(" ", "")
    if len(s) < n:
        return set()  # short/empty aware: no fake {''} token
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _jac(a: set, b: set) -> float:
    if not a and not b:
        return 0.0  # empty-empty is not positive evidence
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _house_unit_nums(addr: str) -> set:
    return set(_HOUSE_RE.findall(addr or ""))


def pair_feature_row(q: dict, t: dict, chmap: dict, rrf: float,
                     src_is_s2: bool) -> list:
    qn, tn = q["name_unicode"], t["name_unicode"]
    qa, ta = q["address_unicode"], t["address_unicode"]
    scores = [s for s, _ in chmap.values()] if chmap else [0.0]
    la, lb = len(qn), len(tn)
    out = [
        max(scores),
        float(len(chmap)),
        fuzz.WRatio(qn, tn) / 100.0,
        fuzz.token_set_ratio(qn, tn) / 100.0,
        fuzz.token_sort_ratio(qn, tn) / 100.0,
        fuzz.partial_ratio(qn, tn) / 100.0,
        _jac(char_ngrams(qn), char_ngrams(tn)),
        (min(la, lb) / max(la, lb)) if max(la, lb) else 0.0,
        1.0 if (qn and qn == tn) else 0.0,
        _jac(set(q.get("name_core", "").split()), set(t.get("name_core", "").split())),
        fuzz.token_sort_ratio(qa, ta) / 100.0,
        fuzz.token_set_ratio(qa, ta) / 100.0,
        fuzz.partial_ratio(qa, ta) / 100.0,
        _jac(set(qa.split()), set(ta.split())),
        1.0 if (qa.strip() and ta.strip()) else 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # numeric slots filled below
        1.0 if src_is_s2 else 0.0,
        float(rrf),
    ]
    qn_nums = q.get("numbers", {})
    tn_nums = t.get("numbers", {})
    qh, th = set(qn_nums.get("house_tokens", [])), set(tn_nums.get("house_tokens", []))
    qu, tu = set(qn_nums.get("unit_tokens", [])), set(tn_nums.get("unit_tokens", []))
    qp = (qn_nums.get("postal_candidates", []) or [None])[0]
    tp = (tn_nums.get("postal_candidates", []) or [None])[0]
    out[15] = 1.0 if (qh & th) else 0.0
    out[16] = 1.0 if (qh and th and not (qh & th)) else 0.0
    out[17] = 1.0 if (qu & tu) else 0.0
    out[18] = 1.0 if (qu and tu and not (qu & tu)) else 0.0
    out[19] = 1.0 if (qp and qp == tp) else 0.0
    out[20] = 1.0 if (qp and tp and qp != tp) else 0.0
    return out


def rows_to_matrix(rows: list) -> np.ndarray:
    return np.array(rows, dtype=np.float32)
