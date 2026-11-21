import json, re, shutil, hashlib
from pathlib import Path

text = open("ground_truth_vrc.json").read()
gt = [json.loads(m) for m in re.findall(r"\{[^{}]*\}", text)]

run10 = {p.name.replace("_extracted.json", "") for p in Path("results/outputs_public").glob("*_extracted.json")}
frozen = [e for e in gt if Path(e["vrc_image_path"]).stem in run10]
matched = {Path(e["vrc_image_path"]).stem for e in frozen}

print(f"Entries parsed from file: {len(gt)}")
print(f"Run 10 extraction files: {len(run10)}")
print(f"Ground-truth entries matched: {len(frozen)}")
print(f"Unmatched extraction files: {sorted(run10 - matched)}")

if len(frozen) != 20 or run10 - matched:
    raise SystemExit("Stopping: expected exactly 20 matches. Nothing was written.")

dest = Path("frozen_run10")
dest.mkdir(exist_ok=True)
json.dump(frozen, open(dest / "ground_truth_vrc.json", "w"), indent=2)
for f in ["ground_truth_ghana_card.json", "run10.log"]:
    if Path(f).exists(): shutil.copy(f, dest)
for d in ["results", "outputs"]:
    if Path(d).exists(): shutil.copytree(d, dest / d, symlinks=True, dirs_exist_ok=True)
shutil.copy("ground_truth_vrc.json", dest / "ground_truth_vrc_ORIGINAL_EXPANDED_BACKUP.json")

with open(dest / "CHECKSUMS.txt", "w") as out:
    for p in sorted(dest.rglob("*")):
        if p.is_file() and p.name != "CHECKSUMS.txt":
            out.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p}\n")
print("Done. Frozen copy in frozen_run10/")
