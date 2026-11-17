"""Check local documentation links, paper assets and recorded source hashes."""
from pathlib import Path
import hashlib
import json
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    count = 0
    for path in ROOT.rglob("*.md"):
        relative = path.relative_to(ROOT)
        if any(part.startswith(".") for part in relative.parts) or (relative.parts[0] == "data" and path.name != "README.md"):
            continue
        if relative.parts[0] == "data" and len(relative.parts) > 1:
            continue
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)\)", text):
            if ":" in target or target.startswith("#"):
                continue
            filename = unquote(target.split("#")[0])
            if filename and not (path.parent/filename).exists():
                errors.append(f"Broken link: {relative}: {target}")
            count += 1
    for directory in (ROOT/"experiments").iterdir():
        if not directory.is_dir():
            continue
        for path in directory.rglob("results.json"):
            result = json.loads(path.read_text(encoding="utf-8"))
            checks = result.get("implementation_sha256", result.get("code_sha256", {}))
            if isinstance(checks, str):
                checks = {"run_demo.py": checks}
            for name, expected in checks.items():
                source = directory/name
                if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                    errors.append(f"Source hash mismatch: {path.relative_to(ROOT)}: {name}")
    for path in (ROOT/"papers").glob("*/asset_manifest.json"):
        for name, expected in json.loads(path.read_text()).items():
            source = ROOT/name
            if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                errors.append(f"Paper input mismatch: {name}")
    pdfs = list((ROOT/"papers").glob("*/kernel_to_*.pdf"))
    if len(pdfs) != 2 or any(not p.read_bytes().startswith(b"%PDF") for p in pdfs):
        errors.append("Expected two built research-note PDFs")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Checked {count} local documentation links, recorded source hashes and both paper PDFs.")


if __name__ == "__main__":
    main()
