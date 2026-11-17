"""Rebuild all research notes from retained result arrays and compile twice."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PAPERS = (("01-kernel-to-equation", "kernel_to_equation"),
          ("02-kernel-to-magnetostatics", "kernel_to_magnetostatics"),
          ("03-maxwells-perspective", "maxwells_perspective"))


def main():
    latex = shutil.which("pdflatex")
    if latex is None:
        raise SystemExit("Install LaTeX with pdflatex before building the papers.")
    for folder, name in PAPERS:
        directory = ROOT/"papers"/folder
        print(f"Building {name}...", flush=True)
        subprocess.run([sys.executable, str(directory/"build_assets.py")], cwd=ROOT, check=True)
        for _ in range(2):
            result = subprocess.run([latex, "-interaction=batchmode", "-halt-on-error", name+".tex"],
                                    cwd=directory, capture_output=True, text=True)
            if result.returncode:
                raise SystemExit(f"LaTeX failed; see {directory/(name+'.log')}")
        log = (directory/(name+".log")).read_text(encoding="utf-8", errors="replace")
        if any(message in log for message in ("Overfull", "undefined", "Rerun to get")):
            raise SystemExit(f"Unresolved layout/reference issue: {directory/(name+'.log')}")
        print(directory/(name+".pdf"), flush=True)


if __name__ == "__main__":
    main()
