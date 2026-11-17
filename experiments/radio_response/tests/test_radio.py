"""Numerical, framing, causal-scope and leakage checks; fixtures are not evidence."""
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from records import masked_crc, read_record, tensor
from kernel_models import antenna_positions, fit_green_gain, green_quotient


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


radio = load_module("radio_run_for_tests", "run_experiment.py")
download = load_module("radio_download_for_tests", "fetch_data.py")


@pytest.fixture(autouse=True)
def one_thread():
    with threadpool_limits(limits=1):
        yield


def varint(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def message(field, content):
    return varint(field << 3 | 2) + varint(len(content)) + content


def integer(field, value):
    return varint(field << 3) + varint(value)


def record_fixture():
    values = np.array([[1., 2.], [3., 4.]], dtype="<f4")
    shape = b"".join(message(2, integer(1, size)) for size in values.shape)
    serialized_tensor = integer(1, 1) + message(2, shape) + message(4, values.tobytes())
    feature = message(1, message(1, serialized_tensor))
    entry = message(1, b"matrix") + message(2, feature)
    scalar = message(1, b"time") + message(2, message(2, message(1, struct.pack("<f", 1.25))))
    payload = message(1, message(1, entry) + message(1, scalar))
    length = struct.pack("<Q", len(payload))
    return length + struct.pack("<I", masked_crc(length)) + payload + struct.pack("<I", masked_crc(payload)), values


def test_real_format_decoder_and_crc_rejection():
    record, expected = record_fixture()
    parsed = read_record(record)
    np.testing.assert_array_equal(tensor(parsed["matrix"]), expected)
    assert parsed["time"] == 1.25
    corrupted = bytearray(record)
    corrupted[15] ^= 1
    with pytest.raises(ValueError, match="payload CRC"):
        read_record(bytes(corrupted))
    corrupted = bytearray(record)
    corrupted[8] ^= 1
    with pytest.raises(ValueError, match="header CRC"):
        read_record(bytes(corrupted))
    with pytest.raises(ValueError, match="length"):
        read_record(record[:-1])


def test_range_fetch_refuses_whole_file_fallback(monkeypatch):
    class Response:
        status_code = 200
        headers = {}
        raw = BytesIO(b"should never read this")
        url = "https://example.invalid/data"
        def raise_for_status(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    class Session:
        def get(self, *args, **kwargs):
            return Response()
    monkeypatch.setattr(download, "session", lambda: Session())
    with pytest.raises(ValueError, match="unbounded"):
        download.ranged("https://example.invalid/data", 0, 11, 10000000)


def test_common_clock_phase_and_delay_cancel_in_relative_operator():
    rng = np.random.default_rng(25)
    channels = rng.normal(size=(20, 6, 64)) + 1j*rng.normal(size=(20, 6, 64))
    frequencies = np.fft.fftfreq(64, d=1/50e6)
    phase = rng.uniform(-np.pi, np.pi, (20, 1))
    delay = rng.uniform(-1e-6, 1e-6, (20, 1))
    clock = np.exp(1j*phase - 2j*np.pi*delay*frequencies[None, :])
    shifted = channels*clock[:, None, :]
    np.testing.assert_allclose(shifted[:, 1:]/shifted[:, :1], channels[:, 1:]/channels[:, :1], atol=1e-12)


def test_geometric_gain_is_the_exact_complex_least_squares_fit():
    rng = np.random.default_rng(65)
    reference = rng.normal(size=(80, 10)) + 1j*rng.normal(size=(80, 10))
    propagation = rng.normal(size=(80, 4, 10)) + 1j*rng.normal(size=(80, 4, 10))
    actual = rng.normal(size=(80, 4, 10)) + 1j*rng.normal(size=(80, 4, 10))
    gain = fit_green_gain(propagation, reference, actual)
    for port in range(4):
        for frequency in range(10):
            design = (propagation[:, port, frequency]*reference[:, frequency])[:, None]
            expected = np.linalg.lstsq(design, actual[:, port, frequency], rcond=None)[0][0]
            assert gain[port, frequency] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_relative_delay_may_be_negative_without_noncausal_field_propagation():
    bandwidth, n = 50e6, 64
    frequencies = np.fft.fftfreq(n, d=1/bandwidth)
    relative_delay = -4/bandwidth
    spectrum = np.exp(-2j*np.pi*frequencies*relative_delay)
    impulse = np.fft.fftshift(np.fft.ifft(spectrum))
    assert np.argmax(abs(impulse)) == n//2-4


def fixture_spec():
    return {"frequency": 3438000000, "bandwidth": 50e6, "antennas": [
        {"assignments": [[0, 1, 2], [3, 4, 5]], "location": [-5., 0., 1.4],
         "direction": [1., 0., 0.], "spacingX": .04364, "spacingY": .04364, "name": "first"},
        {"assignments": [[6, 7, 8], [9, 10, 11]], "location": [8., 0., 1.4],
         "direction": [-1., 0., 0.], "spacingX": .04364, "spacingY": .04364, "name": "second"}]}


def test_geometry_rotation_basis_preserves_published_array_center():
    for array in fixture_spec()["antennas"]:
        positions = np.array(list(antenna_positions(array).values()))
        np.testing.assert_allclose(positions.mean(axis=0), array["location"], atol=1e-12)
        assert len(positions) == 6


def test_direct_path_fixture_and_no_test_label_leakage(tmp_path, monkeypatch):
    # Artificial direct-path arrays below validate software, never experimental claims.
    specification = fixture_spec()
    spec_bytes = json.dumps(specification).encode()
    (tmp_path / "spec.json").write_bytes(spec_bytes)
    import hashlib
    manifest = {"doi": "10.18419/DARUS-3831", "version": "1.0", "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(), "records": []}
    (tmp_path / "provenance.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(radio, "LENGTHS", (1.,))
    monkeypatch.setattr(radio, "RIDGES", (.1,))
    monkeypatch.setattr(radio, "plot_results", lambda result, output: None)
    monkeypatch.setattr(radio, "save_impulse_examples", lambda saved, bandwidth, output: None)
    offset, reads = [0.], []

    def load(data, manifest, name):
        reads.append(name)
        rng = np.random.default_rng({"dc01": 1, "dc02": 2, "dc04": 3, "dc13": 4}[name])
        n = 55 if name == "dc01" else 30
        x = np.column_stack((rng.uniform(-1, 1, n), rng.uniform(3, 6, n), np.full(n, .9)))
        frequencies = specification["frequency"] + np.fft.fftfreq(1024, d=1/50e6)[radio.BINS]
        csi = np.empty((n, 12, len(radio.BINS)), dtype=complex)
        configuration = {"horizontal_sign": 1, "vertical_sign": 1, "phase_sign": -1}
        for array in specification["antennas"]:
            ports = np.asarray(array["assignments"]).reshape(-1).tolist()
            reference = np.exp(1j*rng.uniform(-np.pi, np.pi, (n, len(radio.BINS))))
            ratio = green_quotient(x, array, ports[0], ports, frequencies, configuration)
            gains = np.linspace(.8, 1.2, len(ports))[None, :, None]
            csi[:, ports, :] = ratio*reference[:, None, :]*gains
            if name in ("dc04", "dc13"):
                csi[:, ports[1:], :] += offset[0]
        return radio.Snapshots(name, x, csi, np.full((n, 12), 15.), np.zeros(n), np.arange(n), np.arange(n))

    monkeypatch.setattr(radio, "load_group", load)
    first = radio.run(tmp_path, tmp_path / "first")
    assert all(row["scores"]["all"]["transfer_nmse"] < 1e-18 for row in first["evaluations"] if row["model"] == "geometric_green")
    offset[0] = 50.
    second = radio.run(tmp_path, tmp_path / "second")
    assert reads == ["dc01", "dc02", "dc04", "dc13"]*2
    assert first["model_selection"] == second["model_selection"]
    assert first["calibration"] == second["calibration"]
    assert first["split_audit"] == second["split_audit"]
    with np.load(tmp_path / "first/predictions.npz") as a, np.load(tmp_path / "second/predictions.npz") as b:
        for group in ("dc04", "dc13"):
            for array in (0, 1):
                for name in radio.MODELS:
                    np.testing.assert_array_equal(a[f"{group}_array{array}_{name}"], b[f"{group}_array{array}_{name}"])


def test_saved_measurement_scores_and_kernel_curves_are_reproducible():
    folder = ROOT / "results"
    if not all((folder / name).exists() for name in ("results.json", "predictions.npz", "impulse_examples.npz")):
        pytest.skip("Run the measured radio study before checking its artifacts")
    result = json.loads((folder / "results.json").read_text())
    assert result["synthetic_observations"] is False
    assert result["absolute_causality_or_light_speed_tested"] is False
    with np.load(folder / "predictions.npz") as saved, np.load(folder / "impulse_examples.npz") as impulse:
        for row in result["evaluations"]:
            prefix = f"{row['file']}_array{row['array']}"
            actual, reference = saved[f"{prefix}_actual"], saved[f"{prefix}_reference"]
            prediction = saved[f"{prefix}_{row['model']}"]*reference[:, None, :]
            expected = np.sum(abs(actual-prediction)**2)/np.sum(abs(actual)**2)
            assert row["scores"]["all"]["transfer_nmse"] == pytest.approx(expected, rel=1e-12)
        for array in (0, 1):
            prefix = f"dc04_array{array}"
            for name in ("learned_filter", "geometric_green", "green_envelope"):
                expected = np.fft.fftshift(np.fft.ifft(saved[f"{prefix}_{name}"][0, 0]))
                np.testing.assert_array_equal(expected, impulse[f"array{array}_{name}"])
