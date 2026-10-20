#!/usr/bin/env python3
"""Dual-antenna GNSS heading in the replay (tools/replay.py).

Covers the config section and heading.csv the replay accepts, python/
replay.py's per-row gates and the baseline geometry (REQ-NAV-087) behind
heading_measurement().

Runs under pytest or standalone:

    python3 python/tests/test_heading_stream.py
"""

import math
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "datasets"))

import replay                              # noqa: E402


def test_heading_config_section_and_csv_are_accepted_by_the_replay():
    out = tempfile.mkdtemp(prefix="heading_cfg_")
    cfg = """imu:
  gyr_psd: 1.0e-4
  acc_psd: 1.0e-3
heading:
  enable: 1
  baseline_frd: [1.0, 0.0, 0.0]
  require_fixed: 1
"""
    csv = """# t_us, heading_deg, stddev_deg, carr_soln, length_m, itow_ms
5000000,90.00000,0.20000,2,1.1200,500000
"""
    for name, text in (("config.yaml", cfg), ("heading.csv", csv)):
        with open(os.path.join(out, name), "wb") as f:
            f.write(text.encode("utf-8"))
    spec, _ = replay.load_config(out)
    h = spec["heading"]
    assert int(h["enable"]) == 1
    assert [float(v) for v in h["baseline_frd"]] == [1.0, 0.0, 0.0]
    assert int(h["require_fixed"]) == 1
    rows = replay.load_heading(replay.input_path(out, spec, "heading"))
    assert rows == [(5_000_000, 90.0, 0.2, 2)], rows


def _cfg(**kw):
    c = dict(replay.DEFAULTS["heading"])
    c.update(kw)
    return c


def test_heading_measurement_gates_and_noise():
    row = (0, 30.0, 0.2, 2)
    why, yaw, sd = replay.heading_measurement(_cfg(), row, None)
    assert why == replay.HEADING_OK
    assert abs(math.degrees(yaw) - 30.0) < 1e-4
    assert abs(math.degrees(sd) - 0.2) < 1e-6

    float_row = (0, 30.0, 0.2, 1)
    assert replay.heading_measurement(_cfg(), float_row, None)[0] == \
        replay.HEADING_NOT_FIXED
    assert replay.heading_measurement(_cfg(require_fixed=0), float_row,
                                      None)[0] == replay.HEADING_OK

    _, _, sd = replay.heading_measurement(_cfg(stddev_scale=3.0), row, None)
    assert abs(math.degrees(sd) - 0.6) < 1e-6
    _, _, sd = replay.heading_measurement(_cfg(stddev_min_deg=1.0), row, None)
    assert abs(math.degrees(sd) - 1.0) < 1e-6

    assert replay.heading_measurement(_cfg(), (0, 30.0, 0.0, 2), None)[0] == \
        replay.HEADING_BAD_STDDEV
    assert replay.heading_measurement(
        _cfg(baseline_frd=[0.0, 0.0, 1.0]), row, None)[0] == \
        replay.HEADING_BAD_GEOMETRY


def test_heading_measurement_undoes_a_tilted_cross_baseline():
    # Antennas across the vehicle (base left, rover right): level, the
    # baseline points 90 deg right of the nose. Banked and pitched, its
    # azimuth moves by more than the yaw does, and the attitude handed in
    # is what takes that back out.
    roll, pitch, yaw = math.radians(25.0), math.radians(12.0), math.radians(-60.0)
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    b = (0.0, 1.0, 0.0)
    # R_b_to_n = Rz(yaw) Ry(pitch) Rx(roll), second column = R * e_y
    bn = (cy * sp * sr - sy * cr, sy * sp * sr + cy * cr)
    heading_deg = math.degrees(math.atan2(bn[1], bn[0]))
    why, y, _ = replay.heading_measurement(_cfg(baseline_frd=list(b)),
                                           (0, heading_deg, 0.2, 2),
                                           (roll, pitch, 0.0))
    assert why == replay.HEADING_OK
    assert abs(math.degrees(y - yaw)) < 1e-3, math.degrees(y)
    # Treated as level instead, the same row reads several degrees off.
    _, y_level, _ = replay.heading_measurement(_cfg(baseline_frd=list(b)),
                                               (0, heading_deg, 0.2, 2), None)
    assert abs(math.degrees(y_level - yaw)) > 2.0, math.degrees(y_level)


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
