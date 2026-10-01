#!/usr/bin/env python3
"""imu: mount_rpy_deg in config.yaml (REQ-VER-036), python/replay.py side.

The board's attitude in the vehicle frame is composed onto the acc, gyr and
mag calibration matrices where the calibration is handed to the filter
(replay.mounted_calibration(), used by build_config()), never written back
into the configuration.

Runs under pytest or standalone:

    python3 python/tests/test_mount_rotation.py
"""

import math
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))

import yaml       # noqa: E402
import replay     # noqa: E402

M0 = (1.01, 0.002, -0.001, 0.003, 0.99, 0.0, 0.0, -0.002, 1.02)


def _load(extra_imu, mag_misalignment=None):
    cfg = {"name": "mount_test",
           "imu": dict({"gyr_psd": 1e-7, "acc_psd": 1e-5}, **extra_imu),
           "score": {"warmup_sec": 1, "min_epochs": 1}}
    if mag_misalignment is not None:
        cfg["mag"] = {"misalignment": list(mag_misalignment)}
    d = tempfile.mkdtemp()
    path = os.path.join(d, "config.yaml")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    spec, _ = replay.load_config(path)
    return spec, cfg


def _mul(M, v):
    return tuple(M[r] * v[0] + M[r + 3] * v[1] + M[r + 6] * v[2] for r in range(3))


def _close(a, b, tol=1e-9):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def test_mount_composes_all_three_sensors():
    spec, _ = _load({"acc_misalignment": list(M0), "mount_rpy_deg": [0.0, 0.0, -9.0]})
    acc, gyr, mag = replay.mounted_calibration(spec)
    # Sensor x 9 deg left of forward: the vehicle's forward axis seen in
    # sensor axes comes out as vehicle x (gyr and mag had no calibration).
    d = math.radians(9.0)
    fwd_s = (math.cos(d), math.sin(d), 0.0)
    assert _close(_mul(gyr, fwd_s), (1.0, 0.0, 0.0)), _mul(gyr, fwd_s)
    assert _close(_mul(mag, fwd_s), (1.0, 0.0, 0.0)), _mul(mag, fwd_s)
    # The accelerometer's own calibration is applied first, then the mount.
    x = (0.4, -2.0, 9.7)
    assert _close(_mul(acc, x), _mul(gyr, _mul(M0, x))), (_mul(acc, x),)
    # ZYX: the angles read back off the composed matrix are the ones given.
    spec, _ = _load({"mount_rpy_deg": [20.0, -15.0, 170.0]})
    R = replay.mounted_calibration(spec)[1]
    assert abs(math.degrees(math.atan2(R[5], R[8])) - 20.0) < 1e-9
    assert abs(math.degrees(-math.asin(R[2])) + 15.0) < 1e-9
    assert abs(math.degrees(math.atan2(R[1], R[0])) - 170.0) < 1e-9
    # And build_config() hands exactly these to the filter.
    spec, _ = _load({"acc_misalignment": list(M0), "mount_rpy_deg": [2.0, -3.0, -9.0]})
    acc, gyr, mag = replay.mounted_calibration(spec)
    cfg = replay.build_config(spec, None, 0, 0.8, 0.2, 100.0, None)
    assert _close(tuple(cfg.imu_acc_misalignment), acc, 1e-6)
    assert _close(tuple(cfg.imu_gyr_misalignment), gyr, 1e-6)
    assert _close(tuple(cfg.mag_misalignment), mag, 1e-6)


def test_zero_mount_is_a_noop():
    spec, _ = _load({"acc_misalignment": list(M0)}, mag_misalignment=M0)
    acc, gyr, mag = replay.mounted_calibration(spec)
    assert acc == M0 and mag == M0 and gyr == (0.0,) * 9


def test_mount_survives_a_config_roundtrip():
    # Loading, composing and saving the configuration again must not fold
    # the mounting into the matrices: a second load composes it exactly
    # once more, onto the unchanged calibration.
    spec, raw = _load({"acc_misalignment": list(M0), "mount_rpy_deg": [2.0, -3.0, -9.0]})
    first = replay.mounted_calibration(spec)
    assert tuple(spec["imu"]["acc_misalignment"]) == M0
    d = tempfile.mkdtemp()
    path = os.path.join(d, "config.yaml")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
    spec2, _ = replay.load_config(path)
    assert replay.mounted_calibration(spec2) == first


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:     # noqa: BLE001
                failed += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if failed else 0)
