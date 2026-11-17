"""Download the public FIU MARG recordings and independent optical poses."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
COMMIT = "998c9ebbd243b3e13ba2dc8bbf94e958b6be1b8b"
REPOSITORY = "https://github.com/LABDSP/FIUMARGDB_marg_signals_and_reference_orientations"
RAW = f"https://raw.githubusercontent.com/LABDSP/FIUMARGDB_marg_signals_and_reference_orientations/{COMMIT}"


def fetch(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    names = ["README.md", "LICENSE", "readDBFile.m", "showTrioOut.m", "showDifTrioKF.m"]
    names += [f"REC_{start:02d}-{start+4:02d}.zip" for start in range(1, 31, 5)]
    manifest_path = destination / "provenance.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    hashes = {r["path"]: r["sha256"] for r in old.get("files", [])}

    def one(name):
        target = destination / name
        if target.exists() and old.get("commit") == COMMIT:
            data = target.read_bytes()
            if hashlib.sha256(data).hexdigest() == hashes.get(name):
                return name, data
        request = Request(f"{RAW}/{name}", headers={"User-Agent": "blind-field-discovery"})
        with urlopen(request, timeout=90) as response:
            data = response.read()
        target.write_bytes(data)
        return name, data

    records = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for name, data in pool.map(one, names):
            records.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            if name.endswith(".zip"):
                with ZipFile(BytesIO(data)) as archive:
                    for member in archive.infolist():
                        basename = Path(member.filename).name.lower()
                        if not basename.startswith("rec") or not basename.endswith(".csv"):
                            continue
                        # Only fixed basenames are extracted; archive paths cannot escape the data folder.
                        if basename not in {f"rec{i:02d}.csv" for i in range(1, 31)}:
                            raise ValueError(f"Unexpected recording name: {member.filename}")
                        content = archive.read(member)
                        target = destination / "recordings" / basename
                        target.parent.mkdir(exist_ok=True)
                        target.write_bytes(content)
                        records.append({"path": "recordings/" + basename, "bytes": len(content),
                                        "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {"repository": REPOSITORY, "commit": COMMIT,
                "paper": "https://doi.org/10.3390/s23083786", "files": records,
                "recording_type": "Measured MARG magnetic channels and independent optical position/orientation",
                "licence": "See the retained upstream LICENSE (GNU GPL v3) and paper attribution."}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    count = sum(r["path"].startswith("recordings/") for r in records)
    if count != 30:
        raise ValueError(f"Expected 30 recordings, extracted {count}")
    print(f"Verified {count} public measured recordings in {destination}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT.parents[1] / "data" / "fiu_marg")
    fetch(parser.parse_args().data)
