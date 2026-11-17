"""Download the pinned public Aalto multi-magnetometer recordings, unchanged."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
COMMIT = "b45da3bc7ab623fdef6e01247021daebf7983f29"
REPOSITORY = "https://github.com/AaltoML/magnetic-data"
RAW = f"https://raw.githubusercontent.com/AaltoML/magnetic-data/{COMMIT}"
SENSORS = ("invensense", "nexus", "trivisio")


def fetch(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    names = ["README.md"] + [
        f"data/{sensor}/{trial}-{kind}.csv"
        for sensor in SENSORS for trial in range(1, 10)
        for kind in ("loc", "mag", "time")
    ]
    old_manifest = destination / "provenance.json"
    old = json.loads(old_manifest.read_text()) if old_manifest.exists() else {}
    hashes = {f["path"]: f["sha256"] for f in old.get("files", [])}

    def one(name):
        path = destination / name
        if path.exists():
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if old.get("commit") == COMMIT and hashes.get(name) == digest:
                return {"path": name, "bytes": len(data), "sha256": digest}
        for attempt in range(3):
            try:
                request = Request(f"{RAW}/{name}", headers={"User-Agent": "blind-field-discovery"})
                with urlopen(request, timeout=45) as response:
                    data = response.read()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                return {"path": name, "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()}
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(one, names))
    manifest = {
        "repository": REPOSITORY, "commit": COMMIT,
        "paper": "https://doi.org/10.1109/TRO.2018.2830326",
        "citation": "Solin, Kok, Wahlstrom, Schon, and Sarkka (2018), Modeling and "
                    "interpolation of the ambient magnetic field by Gaussian processes.",
        "preprocessing_by_authors": ["magnetometer calibration", "orientation correction",
                                     "sensor position and alignment correction"],
        "licence_note": "No explicit licence file was present in the pinned upstream tree. "
                        "Upstream requests attribution; do not infer additional redistribution permissions.",
        "files": records,
    }
    old_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(records)} files ({sum(r['bytes'] for r in records):,} bytes) in {destination}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT.parents[1] / "data" / "aalto_magnetic")
    fetch(parser.parse_args().data)
