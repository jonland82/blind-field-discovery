"""Fetch a bounded, reproducible sample of public DICHASUS radio responses.

The archive files are 3–25 GB each. Validated HTTP byte ranges select whole
TFRecords at fixed indices; partial downloads never claim a whole-file MD5.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import struct
import threading
import time

import numpy as np
import requests

from records import read_record, tensor

ROOT = Path(__file__).resolve().parent
DATASET = "https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3831"
API = "https://darus.uni-stuttgart.de/api/"
FILES = {"dc01": (269445, 256), "dc02": (269446, 128),
         "dc04": (269447, 256), "dc13": (269449, 128)}
SPEC_ID = 269444
_local = threading.local()


def session():
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers["User-Agent"] = "blind-field-discovery-public-radio-study"
    return _local.session


def ranged(url, start, stop, expected_size):
    with session().get(url, headers={"Range": f"bytes={start}-{stop}"}, stream=True, timeout=45) as response:
        response.raise_for_status()
        expected = f"bytes {start}-{stop}/{expected_size}"
        if response.status_code != 206 or response.headers.get("Content-Range") != expected:
            raise ValueError("Server did not honor the exact byte range; refusing an unbounded download")
        content = response.raw.read(stop-start+2)
        if len(content) != stop-start+1:
            raise ValueError("Incomplete or oversized range response")
        return content, response.url


def fetch(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    response = session().get(API + "datasets/:persistentId/?persistentId=doi:10.18419/darus-3831", timeout=40)
    response.raise_for_status()
    metadata = response.json()
    version = metadata["data"]["latestVersion"]
    if (version["versionNumber"], version["versionMinorNumber"]) != (1, 0):
        raise ValueError("Dataset version changed; review the acquisition protocol before proceeding")
    (destination / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    entries = {entry["dataFile"]["id"]: entry["dataFile"] for entry in version["files"]}
    spec_response = session().get(API + f"access/datafile/{SPEC_ID}", timeout=40)
    spec_response.raise_for_status()
    spec_bytes = spec_response.content
    if hashlib.md5(spec_bytes).hexdigest() != entries[SPEC_ID]["checksum"]["value"]:
        raise ValueError("Specification MD5 does not match dataset metadata")
    (destination / "spec.json").write_bytes(spec_bytes)
    specification = json.loads(spec_bytes)
    old_path = destination / "provenance.json"
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    cached = {(r["file"], r["record_index"]): r for r in old.get("records", [])}
    records, groups = [], {}

    for name, (identifier, cap) in FILES.items():
        entry = entries[identifier]
        size = entry["filesize"]
        url = API + f"access/datafile/{identifier}"
        header, direct_url = ranged(url, 0, 11, size)
        length = struct.unpack("<Q", header[:8])[0]
        stride = length + 16
        if stride > 2_000_000 or size % stride != 0:
            raise ValueError("Record layout is not the audited fixed-size schema")
        count = size // stride
        indices = np.linspace(0, count-1, cap, dtype=int)
        groups[name] = {"datafile_id": identifier, "filename": entry["filename"], "bytes": size,
                        "whole_file_checksum": entry["checksum"], "whole_file_checksum_verified": False,
                        "record_bytes": stride, "total_records_in_file": count,
                        "selected_record_indices": indices.tolist()}
        folder = destination / "records" / name
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Fetching {name}: {cap}/{count} complete records, {cap*stride/1e6:.1f} MB...", flush=True)

        def one(index):
            index = int(index)
            path = folder / f"{index:06d}.tfrecord"
            previous = cached.get((name, index))
            content = path.read_bytes() if path.exists() else None
            if content is None or previous is None or hashlib.sha256(content).hexdigest() != previous["sha256"]:
                start = index * stride
                for attempt in range(3):
                    try:
                        content, _ = ranged(direct_url, start, start+stride-1, size)
                        break
                    except (requests.RequestException, OSError):
                        if attempt == 2:
                            raise
                        time.sleep(attempt+1)
                # Verify framing and CRCs before accepting a range as a record.
                parsed = read_record(content)
                if tensor(parsed["csi"]).shape != (64, 1024, 2):
                    raise ValueError("Unexpected CSI tensor layout")
                path.write_bytes(content)
            else:
                read_record(content)
            return {"file": name, "record_index": index, "byte_start": index*stride,
                    "bytes": len(content), "path": path.relative_to(destination).as_posix(),
                    "sha256": hashlib.sha256(content).hexdigest(), "tfrecord_crc32c_verified": True}

        with ThreadPoolExecutor(max_workers=8) as pool:
            records.extend(pool.map(one, indices))
        # Save completed file groups so a later interrupted download can resume.
        manifest = {"dataset": DATASET, "doi": "10.18419/DARUS-3831", "version": "1.0",
                    "licence": "CC BY 4.0", "source_specification_warnings": specification.get("warnings", []),
                    "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(), "files": groups, "records": records}
        old_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(records)} measured radio snapshots in {destination}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT.parents[1] / "data" / "dichasus")
    fetch(parser.parse_args().data)
