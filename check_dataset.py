"""Audit YOLO labels and exact image overlap without extra dependencies."""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]


def main():
    report = {"splits": {}, "errors": [], "cross_split_duplicates": []}
    seen = {}
    source_names = {}
    for split in ("train", "valid", "test"):
        folder = ROOT / "Dataset" / split
        images = sorted(p for p in (folder / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        counts = Counter()
        source_names[split] = set()
        empty = 0
        for image in images:
            source = re.sub(r"\.rf\..*$", "", image.name)
            source = re.sub(r"^(rotation_\d+_|l_)", "", source)
            source_names[split].add(source)
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen and seen[digest][0] != split:
                report["cross_split_duplicates"].append([seen[digest][1], str(image.relative_to(ROOT))])
            else:
                seen[digest] = (split, str(image.relative_to(ROOT)))
            label = folder / "labels" / f"{image.stem}.txt"
            if not label.is_file():
                report["errors"].append(f"Missing label: {label}")
                continue
            lines = label.read_text(encoding="utf-8").splitlines()
            if not any(line.strip() for line in lines):
                empty += 1
            for number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError("expected class x y w h")
                    cls = int(fields[0])
                    x, y, w, h = map(float, fields[1:])
                    if cls not in range(6) or not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h)) or w <= 0 or h <= 0:
                        raise ValueError("invalid class or normalized box")
                    if min(x-w/2, y-h/2) < -0.001 or max(x+w/2, y+h/2) > 1.001:
                        raise ValueError("box outside image")
                    counts[NAMES[cls]] += 1
                except ValueError as exc:
                    report["errors"].append(f"{label.name}:{number}: {exc}")
        image_stems = {p.stem for p in images}
        for label in (folder / "labels").glob("*.txt"):
            if label.stem not in image_stems:
                report["errors"].append(f"Orphan label: {label}")
        report["splits"][split] = {"images": len(images), "empty_labels": empty, "boxes": dict(counts)}
    report["potential_shared_sources_by_filename"] = {
        f"{left}/{right}": len(source_names[left] & source_names[right])
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))
    }
    destination = ROOT / "dataset_audit.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["splits"], indent=2))
    print(f"Errors: {len(report['errors'])}; exact cross-split duplicates: {len(report['cross_split_duplicates'])}")
    print(f"Potential shared source names (heuristic): {report['potential_shared_sources_by_filename']}")
    print(f"Report: {destination}")
    raise SystemExit(1 if report["errors"] else 0)


if __name__ == "__main__":
    main()
