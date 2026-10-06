import pandas as pd
"""
Benchmark runner. This file does NOT extract anything itself — it only
calls the real extraction modules and scores their output against
ground truth. If you improve extraction logic, fix it in vrc_ocr.py,
ocr_pipeline.py, or gemini_vision_ocr.py — this script will automatically
reflect the change on the next run.
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from Levenshtein import distance as levenshtein_distance

from dotenv import load_dotenv
load_dotenv()  # lets genai.Client() find GEMINI_API_KEY from .env

# Real extraction modules — the single source of truth for each method.
from src.vrc_ocr import extract_vrc                 # Method A, VRC documents
from src.ocr_pipeline import run_ocr_pipeline           # Method A, Ghana Card
from src.gemini_vision_ocr import extract_vrc_data      # Method B, both document types


VRC_GROUND_TRUTH_PATH = Path("ground_truth_vrc.json")
GHANA_CARD_GROUND_TRUTH_PATH = Path("ground_truth_ghana_card.json")
RESULTS_CSV = Path("results") / "benchmark_results.csv"

VRC_FIELDS = ["vehicle_registration_number", "chassis_number", "make", "model", "colour"]
GHANA_CARD_FIELDS = ["client_name", "ghana_card_id"]


# =====================================================================
# NORMALIZATION — each method names fields differently; map every
# method's raw output onto the SAME canonical keys used in ground truth
# before comparing anything.
# =====================================================================

def normalize_value(v) -> str:
    if v is None:
        return ""
    return str(v).strip().upper()


def normalize_field(field: str, v) -> str:
    """Field-aware normalization applied identically to ground truth AND
    predictions, so formatting differences (spaces, hyphens, casing) are
    never scored as extraction errors."""
    s = normalize_value(v)
    if field in ("vehicle_registration_number", "chassis_number"):
        return s.replace(" ", "").replace("-", "")
    return " ".join(s.split())


def normalize_tesseract_vrc(raw: dict) -> dict:
    # extract_vrc() already uses matching field names.
    return {f: normalize_value(raw.get(f)) for f in VRC_FIELDS}


def normalize_tesseract_ghana_card(raw: dict) -> dict:
    return {
        "client_name": normalize_value(raw.get("client_name")),
        "ghana_card_id": normalize_value(raw.get("ghana_card_number")),
    }


def normalize_gemini_vrc(raw: dict) -> dict:
    # gemini_vision_ocr's prompt schema already matches VRC field names.
    return {f: normalize_value(raw.get(f)) for f in VRC_FIELDS}


def normalize_gemini_ghana_card(raw: dict) -> dict:
    surname = raw.get("surname") or ""
    first_names = raw.get("first_names") or ""
    full_name = f"{surname} {first_names}".strip()
    return {
        "client_name": normalize_value(full_name),
        "ghana_card_id": normalize_value(raw.get("pin_number")),
    }


# =====================================================================
# METHOD RUNNERS — thin wrappers around the real functions. All error
# handling lives here so a single bad document/API call doesn't crash
# the whole batch.
# =====================================================================

def run_method_a(image_path: str, doc_kind: str) -> dict:
    """doc_kind is 'vrc' or 'ghana_card'."""
    try:
        if doc_kind == "vrc":
            raw = extract_vrc(image_path)
            return normalize_tesseract_vrc(raw)
        else:
            raw = run_ocr_pipeline(image_path)
            return normalize_tesseract_ghana_card(raw)
    except Exception as e:
        print(f"  [Method A Error] {image_path}: {e}")
        fields = VRC_FIELDS if doc_kind == "vrc" else GHANA_CARD_FIELDS
        return {f: "" for f in fields}


def run_method_b(image_path: str, doc_kind: str) -> dict:
    fields = VRC_FIELDS if doc_kind == "vrc" else GHANA_CARD_FIELDS
    try:
        cached = Path("outputs") / f"{Path(image_path).stem}_extracted.json"
        if cached.exists():
            print(f"  [Method B] using saved extraction: {cached.name}")
            output_path = str(cached)
        else:
            output_path = extract_vrc_data(image_path)  # returns a file path
        with open(output_path, "r") as f:
            raw = json.load(f)
        if doc_kind == "vrc":
            return normalize_gemini_vrc(raw)
        else:
            return normalize_gemini_ghana_card(raw)
    except Exception as e:
        print(f"  [Method B Error] {image_path}: {e}")
        return {f: "" for f in fields}


# =====================================================================
# METRICS
# =====================================================================

def compute_field_metrics(extracted: dict, truth: dict, fields: list) -> dict:
    exact_matches = 0
    total_edit_distance = 0
    total_chars = 0
    field_results = {}

    for field in fields:
        true_val = normalize_field(field, truth.get(field))
        pred_val = normalize_field(field, extracted.get(field))
        pred_null = pd.isna(pred_val) or str(pred_val).strip().lower() in ["nan", "none", ""]

        true_null = pd.isna(true_val) or str(true_val).strip().lower() in ["nan", "none", ""]


        if pred_null and true_null:

            is_match = True

            edit_dist = 0

        elif pred_null or true_null:

            is_match = False

            non_null_str = str(pred_val if not pred_null else true_val).strip()

            edit_dist = len(non_null_str)

        else:

            p_str = str(pred_val).strip().upper()

            t_str = str(true_val).strip().upper()

            is_match = (p_str == t_str)

            edit_dist = levenshtein_distance(p_str, t_str)


        if is_match:

            exact_matches += 1


        total_edit_distance += edit_dist
        total_chars += max(len(true_val), 1)

        field_results[field] = {
            "ground_truth": true_val,
            "extracted": pred_val,
            "match": is_match,
            "edit_distance": edit_dist,
        }

    total_fields = len(fields)
    exact_match_acc = (exact_matches / total_fields) * 100 if total_fields else 0.0
    cer = (total_edit_distance / total_chars) * 100 if total_chars else 0.0

    return {
        "exact_match_acc": exact_match_acc,
        "cer": cer,
        "total_edit_distance": total_edit_distance,
        "full_pass": exact_matches == total_fields,
        "field_results": field_results,
    }


# =====================================================================
# CSV LOGGING
# =====================================================================

def init_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    f = open(path, "a", newline="")
    writer = csv.writer(f)
    if is_new:
        writer.writerow([
            "timestamp", "document_type", "document", "method", "field",
            "ground_truth", "extracted", "match", "edit_distance",
            "field_exact_match_pct", "cer_pct", "full_pass",
        ])
    return f, writer


def log_result_rows(writer, timestamp, doc_type, doc_name, method, metrics):
    for field, fr in metrics["field_results"].items():
        writer.writerow([
            timestamp, doc_type, doc_name, method, field,
            fr["ground_truth"], fr["extracted"], fr["match"], fr["edit_distance"],
            round(metrics["exact_match_acc"], 2), round(metrics["cer"], 2),
            metrics["full_pass"],
        ])


def print_aggregate(label: str, results: dict, n: int, note: str = ""):
    print(f"\n{'=' * 70}")
    print(f"  {label}  (N = {n}){'  — ' + note if note else ''}")
    print("=" * 70)
    print(f"{'Metric':<35} | {'Tesseract':<15} | {'Gemini':<15}")
    print("-" * 70)
    for metric_name, key in [
        ("Field-Level Exact Match (EM) (%)", "exact_match_acc"),
        ("Character Error Rate (CER) (%)", "cer"),
    ]:
        a_vals = [r[key] for r in results["Tesseract"]]
        b_vals = [r[key] for r in results["Gemini"]]
        avg_a = sum(a_vals) / len(a_vals) if a_vals else 0.0
        avg_b = sum(b_vals) / len(b_vals) if b_vals else 0.0
        print(f"{metric_name:<35} | {avg_a:<15.2f} | {avg_b:<15.2f}")

    pass_a = sum(1 for r in results["Tesseract"] if r["full_pass"]) / n * 100 if n else 0
    pass_b = sum(1 for r in results["Gemini"] if r["full_pass"]) / n * 100 if n else 0
    print(f"{'Format-Level Pass Rate (100% EM)':<35} | {pass_a:<15.2f}% | {pass_b:<15.2f}%")
    print("=" * 70)


# =====================================================================
# MAIN
# =====================================================================

def main():
    if not VRC_GROUND_TRUTH_PATH.exists():
        print(f"Error: {VRC_GROUND_TRUTH_PATH} not found.")
        sys.exit(1)
    if not GHANA_CARD_GROUND_TRUTH_PATH.exists():
        print(f"Error: {GHANA_CARD_GROUND_TRUTH_PATH} not found.")
        sys.exit(1)

    with open(VRC_GROUND_TRUTH_PATH) as f:
        vrc_truth_list = json.load(f)
    with open(GHANA_CARD_GROUND_TRUTH_PATH) as f:
        ghana_card_truth = json.load(f)

    csv_file, csv_writer = init_csv(RESULTS_CSV)

    print("=====================================================================")
    print("      LOW-RESOURCE NLP DOCUMENT EXTRACTION BENCHMARK RUNNER")
    print(f"      VRC documents: {len(vrc_truth_list)}  |  Ghana Card: 1 (fixed)")
    print("=====================================================================")

    vrc_results = {"Tesseract": [], "Gemini": []}
    ghana_card_results = {"Tesseract": [], "Gemini": []}

    try:
        # ---- VRC documents: one ground-truth entry per real vehicle ----
        for entry in vrc_truth_list:
            img_path = entry["vrc_image_path"]
            if not Path(img_path).exists():
                print(f"Skipping {img_path} (file not found)")
                continue

            print(f"\nEvaluating VRC asset: {img_path}")
            timestamp = datetime.now().isoformat(timespec="seconds")

            pred_a = run_method_a(img_path, "vrc")
            metrics_a = compute_field_metrics(pred_a, entry, VRC_FIELDS)
            vrc_results["Tesseract"].append(metrics_a)
            log_result_rows(csv_writer, timestamp, "vrc", img_path, "Tesseract", metrics_a)

            pred_b = run_method_b(img_path, "vrc")
            metrics_b = compute_field_metrics(pred_b, entry, VRC_FIELDS)
            vrc_results["Gemini"].append(metrics_b)
            log_result_rows(csv_writer, timestamp, "vrc", img_path, "Gemini", metrics_b)

            csv_file.flush()

        # ---- Ghana Card: single fixed document, measured once ----
        gc_path = ghana_card_truth["ghana_card_image_path"]

        if not Path(gc_path).exists():
            print(f"\nSkipping Ghana Card evaluation — file not found: {gc_path}")
        else:
            print(f"\nEvaluating Ghana Card asset: {gc_path}")
            timestamp = datetime.now().isoformat(timespec="seconds")

            pred_a = run_method_a(gc_path, "ghana_card")
            metrics_a = compute_field_metrics(pred_a, ghana_card_truth, GHANA_CARD_FIELDS)
            ghana_card_results["Tesseract"].append(metrics_a)
            log_result_rows(csv_writer, timestamp, "ghana_card", gc_path, "Tesseract", metrics_a)

            pred_b = run_method_b(gc_path, "ghana_card")
            metrics_b = compute_field_metrics(pred_b, ghana_card_truth, GHANA_CARD_FIELDS)
            ghana_card_results["Gemini"].append(metrics_b)
            log_result_rows(csv_writer, timestamp, "ghana_card", gc_path, "Gemini", metrics_b)

            csv_file.flush()
    finally:
        csv_file.close()

    # ---- Reporting: kept fully separate, never averaged together ----
    print_aggregate("VRC EXTRACTION RESULTS", vrc_results, len(vrc_results["Tesseract"]))

    if ghana_card_results["Tesseract"]:
        print_aggregate(
            "GHANA CARD EXTRACTION RESULT", ghana_card_results, 1,
            note="single fixed document — not a statistically meaningful sample, informational only",
        )

    print(f"\nDetailed per-field results appended to: {RESULTS_CSV.resolve()}")


if __name__ == "__main__":
    main()