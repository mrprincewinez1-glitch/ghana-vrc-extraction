"""Recompute the paper's tables from the per-field results CSV (VRC documents only).

Usage: python3 recompute_tables.py [path/to/benchmark_results.csv]
Reads: timestamp, document_type, document, method, field, ground_truth,
       extracted, match, edit_distance, ...   (written by evaluate_benchmark.py)
The Ghana Card rows (document_type != 'vrc') are excluded: they are a single
informational document and are reported separately in the paper.
"""
import csv
import sys
from collections import defaultdict
from math import comb

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/benchmark_results.csv"
NULLS = {"", "none", "null", "nan"}
LONG_FIELDS = {"chassis_number", "vehicle_registration_number"}


def is_null(s):
    return str(s).strip().lower() in NULLS


def is_match(s):
    return str(s).strip().lower() in {"true", "1", "yes", "pass"}


def method_of(s):
    return "Tesseract" if "tess" in s.lower() else "Gemini"


def _binom_cdf(k, n, p):
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))


def clopper_pearson(k, n, alpha=0.05):
    """Exact two-sided binomial interval, by bisection (no SciPy needed)."""
    if k == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, 1.0  # smallest p with P(X >= k) >= alpha/2
        for _ in range(80):
            mid = (lo + hi) / 2
            if 1 - _binom_cdf(k - 1, n, mid) < alpha / 2:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2
    if k == n:
        upper = 1.0
    else:
        lo, hi = 0.0, 1.0  # largest p with P(X <= k) >= alpha/2
        for _ in range(80):
            mid = (lo + hi) / 2
            if _binom_cdf(k, n, mid) > alpha / 2:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2
    return 100 * lower, 100 * upper


def fmt_ci(k, n):
    lo, hi = clopper_pearson(k, n)
    return f"{lo:.2f}% to {hi:.2f}%"


def doc_cer(rows):
    num = sum(float(r["edit_distance"]) for r in rows)
    den = sum(max(len(r["ground_truth"]), 1) for r in rows)
    return 100 * num / den


rows = [r for r in csv.DictReader(open(PATH)) if r["document_type"].strip().lower() == "vrc"]
by_method = defaultdict(list)
for r in rows:
    by_method[method_of(r["method"])].append(r)

print(f"File: {PATH}  |  VRC rows: {len(rows)}")
print("\n=== TABLE 1: Overall results (VRC documents only) ===")
for m in ("Tesseract", "Gemini"):
    rs = by_method[m]
    docs = defaultdict(list)
    for r in rs:
        docs[r["document"]].append(r)
    vb = [r for r in rs if not is_null(r["ground_truth"])]
    sub = [r for r in vb if r["field"] in LONG_FIELDS]
    full = sum(all(is_match(x["match"]) for x in d) for d in docs.values())
    cer_all = sum(doc_cer(d) for d in docs.values()) / len(docs)
    docs_vb = {k: [x for x in d if not is_null(x["ground_truth"])] for k, d in docs.items()}
    cer_vb = sum(doc_cer(d) for d in docs_vb.values()) / len(docs_vb)
    em_all = sum(is_match(r["match"]) for r in rs)
    em_vb = sum(is_match(r["match"]) for r in vb)
    em_sub = sum(is_match(r["match"]) for r in sub)
    nl = [r for r in rs if is_null(r["ground_truth"])]
    print(f"\n{m}  ({len(docs)} documents)")
    print(f"  Field EM, all fields           : {100*em_all/len(rs):.2f}% ({em_all}/{len(rs)})  95% CI {fmt_ci(em_all, len(rs))}")
    print(f"  Field EM, value-bearing fields : {100*em_vb/len(vb):.2f}% ({em_vb}/{len(vb)})  95% CI {fmt_ci(em_vb, len(vb))}")
    print(f"  EM, chassis + registration     : {100*em_sub/len(sub):.2f}% ({em_sub}/{len(sub)})  95% CI {fmt_ci(em_sub, len(sub))}")
    print(f"  CER all fields (mean of docs)  : {cer_all:.2f}%")
    print(f"  CER value-bearing (mean docs)  : {cer_vb:.2f}%")
    print(f"  Documents with all fields right: {full}/{len(docs)} ({100*full/len(docs):.2f}%)  95% CI {fmt_ci(full, len(docs))}")
    print(f"  Null-label fields scored right : {sum(is_match(r['match']) for r in nl)}/{len(nl)}")

print("\n=== TABLE 2: Per-field results, value-bearing labels (correct / empty / wrong) ===")
fields = sorted({r["field"] for r in rows})
print(f"{'Field':30} {'n':>3}  {'Tesseract':>14}  {'Gemini':>12}")
for f in fields:
    cells = []
    for m in ("Tesseract", "Gemini"):
        rs = [r for r in by_method[m] if r["field"] == f and not is_null(r["ground_truth"])]
        c = sum(is_match(r["match"]) for r in rs)
        e = sum((not is_match(r["match"])) and is_null(r["extracted"]) for r in rs)
        w = len(rs) - c - e
        cells.append((len(rs), f"{c} / {e} / {w}"))
    print(f"{f:30} {cells[0][0]:>3}  {cells[0][1]:>14}  {cells[1][1]:>12}")
