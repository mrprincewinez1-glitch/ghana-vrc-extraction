import json, subprocess
from pathlib import Path
from src.vrc_ocr import extract_vrc

rows = json.load(open("ground_truth_vrc_batch2.json"))
Path("tmp_jpg").mkdir(exist_ok=True)
ok, bad, r = 0, [], None
for e in rows:
    p = Path(e["vrc_image_path"])
    jpg = Path("tmp_jpg") / (p.stem + ".jpg")
    if not jpg.exists():
        subprocess.run(["sips", "-s", "format", "jpeg", str(p), "--out", str(jpg)], check=True, capture_output=True)
    try:
        r = extract_vrc(str(jpg))
        ok += 1
    except Exception as ex:
        bad.append((p.name, repr(ex)[:100]))
print("Tesseract ran on JPG copies:", ok, "of", len(rows))
for b in bad: print("  FAIL", b)
heic = next(e["vrc_image_path"] for e in rows if e["vrc_image_path"].endswith(".HEIC"))
try:
    extract_vrc(heic)
    print("Direct HEIC read: works")
except Exception as ex:
    print("Direct HEIC read: FAILS ->", repr(ex)[:120])
print("example output keys:", list(r.keys()) if isinstance(r, dict) else type(r))
