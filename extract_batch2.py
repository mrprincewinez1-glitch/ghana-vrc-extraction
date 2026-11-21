import json, subprocess, sys
from pathlib import Path

src = open("results/gemini_vision_ocr_run10_frozen.py").read()
a = 'models_to_try = ["gemini-3.6-flash", "gemini-1.5-flash", "gemini-1.5-pro"]'
b = 'Path("outputs")'
assert a in src and b in src, "frozen script has changed, stopping"
src = src.replace(a, 'models_to_try = ["gemini-3.6-flash"]').replace(b, 'Path("outputs_batch2")')
ns = {}
exec(compile(src, "frozen_patched", "exec"), ns)
extract = ns["extract_vrc_data"]

limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
rows = json.load(open("ground_truth_vrc_batch2.json"))
Path("tmp_jpg").mkdir(exist_ok=True)
Path("outputs_batch2").mkdir(exist_ok=True)

done = 0
for e in rows:
    p = Path(e["vrc_image_path"])
    if (Path("outputs_batch2") / (p.stem + "_extracted.json")).exists():
        continue
    if limit is not None and done >= limit:
        break
    jpg = Path("tmp_jpg") / (p.stem + ".jpg")
    if not jpg.exists():
        subprocess.run(["sips", "-s", "format", "jpeg", str(p), "--out", str(jpg)],
                       check=True, capture_output=True)
    extract(str(jpg))
    done += 1
print("extracted this run:", done)