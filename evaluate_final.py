import csv, json, subprocess, sys
from datetime import datetime
from pathlib import Path

import evaluate_benchmark as eb

OUT_CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval_private/rehearsal.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

frozen = json.load(open("frozen_run10/ground_truth_vrc.json"))
docs = [(e, "pilot") for e in frozen]
for f in sorted(Path(".").glob("ground_truth_vrc_batch*.json")):
    if "BACKUP" not in f.name:
        docs += [(e, "new") for e in json.load(open(f))]

BLANK = {f: "" for f in eb.VRC_FIELDS}
errors = []

def run_tesseract(e, g):
    p = Path(e["vrc_image_path"])
    if g == "new":
        jpg = Path("tmp_jpg") / (p.stem + ".jpg")
        if not jpg.exists():
            Path("tmp_jpg").mkdir(exist_ok=True)
            subprocess.run(["sips", "-s", "format", "jpeg", str(p), "--out", str(jpg)],
                           check=True, capture_output=True)
        p = jpg
    try:
        return eb.normalize_tesseract_vrc(eb.extract_vrc(str(p)))
    except Exception as ex:
        errors.append(("Tesseract", p.name, repr(ex)[:80]))
        return dict(BLANK)

def run_gemini(e, g):
    stem = Path(e["vrc_image_path"]).stem
    f = Path("outputs" if g == "pilot" else "outputs_batch2") / (stem + "_extracted.json")
    try:
        return eb.normalize_gemini_vrc(json.load(open(f)))
    except Exception as ex:
        errors.append(("Gemini", f.name, repr(ex)[:80]))
        return dict(BLANK)

seen = {}
for e, g in docs:
    c = eb.normalize_field("chassis_number", e.get("chassis_number"))
    if c:
        seen.setdefault(c, []).append(Path(e["vrc_image_path"]).name)
print("duplicate chassis:", [v for v in seen.values() if len(v) > 1] or "none")

res = {g: {"Tesseract": [], "Gemini": []} for g in ("pilot", "new")}
now = datetime.now().isoformat(timespec="seconds")
cf = open(OUT_CSV, "w", newline="")
w = csv.writer(cf)
w.writerow(["timestamp", "document_type", "document", "method", "field", "ground_truth",
            "extracted", "match", "edit_distance", "field_exact_match_pct", "cer_pct", "full_pass"])
for e, g in docs:
    a = eb.compute_field_metrics(run_tesseract(e, g), e, eb.VRC_FIELDS)
    b = eb.compute_field_metrics(run_gemini(e, g), e, eb.VRC_FIELDS)
    res[g]["Tesseract"].append(a)
    res[g]["Gemini"].append(b)
    eb.log_result_rows(w, now, "vrc", e["vrc_image_path"], "Tesseract", a)
    eb.log_result_rows(w, now, "vrc", e["vrc_image_path"], "Gemini", b)
cf.close()

allr = {m: res["pilot"][m] + res["new"][m] for m in ("Tesseract", "Gemini")}
eb.print_aggregate("ALL DOCUMENTS (pooled)", allr, len(allr["Gemini"]))
eb.print_aggregate("FIRST 20 (Run 10 pilot set)", res["pilot"], len(res["pilot"]["Gemini"]))
eb.print_aggregate("NEW DOCUMENTS ONLY", res["new"], len(res["new"]["Gemini"]))
print("\nerrors:", len(errors))
for x in errors:
    print("  ", x)
print("CSV written to:", OUT_CSV)
