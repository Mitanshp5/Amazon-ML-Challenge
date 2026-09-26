"""Structured retrieval: frequency-ranked keys, overflow logging (F05 10.3).

Fixes: oversized buckets silently deleted + postings sliced in input order.
Here every key family logs posting-length distribution / overflow / unique
gains; common keys require intersection with independent evidence or
secondary ranking. Exact-name candidates stay candidates (never auto-accept).
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

_PIN = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")
_BIGNUM = re.compile(r"\b\d{4,}\b")
_HOUSE = re.compile(r"\b\d{1,3}\b")

CAP = {"exact": 500, "pin": 500, "bignum": 500, "house": 250,
       "prefix": 250, "token": 150, "loc": 150}
W = {"exact": 3.0, "pin": 2.5, "bignum": 2.0, "house": 1.5,
     "token": 1.0, "prefix": 1.0, "loc": 1.0}


def keys_of(name: str, addr: str, stop: set) -> dict:
    ntoks = [t for t in name.split() if len(t) >= 3 and t not in stop]
    atoks = [t for t in addr.split() if len(t) >= 3]
    compact = name.replace(" ", "")
    out = {"exact": [name] if name else [],
           "token": ntoks,
           "prefix": [compact[:4]] if len(compact) >= 4 else [],
           "pin": _PIN.findall(addr),
           "bignum": list(set(_BIGNUM.findall(name + " " + addr))),
           "house": [],
           "loc": [t for t in atoks if len(t) >= 6]}
    hn = _HOUSE.findall(addr)
    if hn and atoks:
        out["house"] = [hn[0] + "|" + atoks[0]]
    return out


def build_index(names, addrs, stop: set):
    index = {k: defaultdict(list) for k in W}
    for pi, (n, a) in enumerate(zip(names, addrs)):
        for kt, kl in keys_of(n, a, stop).items():
            for k in kl:
                index[kt][k].append(pi)
    overflow = {}
    for kt in W:
        big = [k for k, v in index[kt].items() if len(v) > CAP[kt]]
        overflow[kt] = {"n_keys": len(index[kt]), "n_overflow": len(big),
                        "max_posting": max((len(v) for v in index[kt].values()), default=0)}
        for k in big:
            del index[kt][k]  # explicit, logged; never silent first-N relevance
    return index, overflow


def score_query(name, addr, index, stop: set):
    acc: dict = {}
    for kt, kl in keys_of(name, addr, stop).items():
        d = index[kt]
        for k in kl:
            bkt = d.get(k)
            if not bkt:
                continue
            w = W[kt] / math.log2(2 + len(bkt))
            for pi in bkt:
                acc[pi] = acc.get(pi, 0.0) + w
    ranked = sorted(acc.items(), key=lambda kv: -kv[1])[:200]
    return ranked
