#!/usr/bin/env python3
"""Dual-antenna GNSS heading stream (heading.csv) end to end.

Covers the three places the stream passes through before it reaches the
filter: tools/inslib_convert_ubx_to_csv.py turning a moving base
NAV-RELPOSNED into heading.csv rows that share their t_us with the gnss.csv
fix of the same epoch, the generated config.yaml section being one the
replay harness accepts, and python/replay.py's per-row gates plus the
baseline geometry (REQ-NAV-087) behind heading_measurement().

Runs under pytest or standalone:

    python3 python/tests/test_heading_stream.py
"""

import math
import os
import struct
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "datasets"))

import inslib_convert_ubx_to_csv as conv   # noqa: E402
import replay                              # noqa: E402
from inslib_ubx import (IMU_FMT, ID_IMU, build_imu_status,   # noqa: E402
                        ubx_frame)

IMU_DT_US = 1250


def _imu(t_us):
    payload = struct.pack(IMU_FMT, t_us, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0,
                          build_imu_status(25.0), 0)
    return ubx_frame(0x40, ID_IMU, payload)


def _nav_pvt(itow_ms, fix_type=3):
    # 92 B, only what the converter reads is non-zero: iTOW, a valid 2026
    # date, a 3D fix with gnssFixOK, a position and accuracies.
    p = bytearray(92)
    struct.pack_into("<I", p, 0, itow_ms)
    struct.pack_into("<HBBBBB", p, 4, 2026, 9, 24, 12, 0, 0)
    p[11] = 0x07                        # validDate/validTime/fullyResolved
    p[20] = fix_type
    p[21] = 0x01                        # gnssFixOK
    p[23] = 20
    struct.pack_into("<iiii", p, 24, 115000000, 481000000, 500000, 450000)
    struct.pack_into("<II", p, 40, 800, 1200)
    struct.pack_into("<I", p, 68, 50)
    return ubx_frame(0x01, 0x07, bytes(p))


def _relposned(itow_ms, heading_deg, acc_deg, carr_soln=2, moving=True,
               heading_valid=True, length_cm=112):
    flags = 0x01 | 0x02 | 0x04 | (carr_soln << 3)
    if moving:
        flags |= 1 << 5
    if heading_valid:
        flags |= 1 << 8
    p = struct.pack("<BBHIiiiii4sbbbbIIIII4sI", 1, 0, 0, itow_ms,
                    100, 50, 0, length_cm, int(round(heading_deg * 1e5)),
                    b"\0" * 4, 0, 0, 0, 0, 10, 10, 10, 10,
                    int(round(acc_deg * 1e5)), b"\0" * 4, flags)
    return ubx_frame(0x01, 0x3C, p)


def _read_rows(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip().split(",") for line in f
                if line.strip() and not line.startswith("#")]


def _convert(frames):
    tmp = tempfile.mkdtemp(prefix="heading_stream_")
    ubx = os.path.join(tmp, "capture.ubx")
    with open(ubx, "wb") as f:
        f.write(b"".join(frames))
    out = os.path.join(tmp, "out")
    n, _, _ = conv.convert(ubx, out)
    return n, out


def test_heading_row_shares_t_us_with_its_gnss_fix():
    t = 1_000_000
    frames = []
    for itow in (100_000, 100_100, 100_200):
        frames.append(_imu(t))
        frames.append(_nav_pvt(itow))
        # The rover solves the baseline only once the base's RTCM is in:
        # NAV-RELPOSNED lands after one more IMU sample, and must still
        # carry the t_us of its epoch's NAV-PVT, not its own arrival.
        t += IMU_DT_US
        frames.append(_imu(t))
        frames.append(_relposned(itow, 45.5 + itow * 1e-5, 0.3))
        t += 100_000
    n, out = _convert(frames)
    assert n["heading"] == 3, n
    gnss_t = [int(r[0]) for r in _read_rows(os.path.join(out, "gnss.csv"))]
    rows = _read_rows(os.path.join(out, "heading.csv"))
    assert [int(r[0]) for r in rows] == gnss_t, (rows, gnss_t)
    assert abs(float(rows[0][1]) - (45.5 + 1.0)) < 1e-4, rows[0]
    assert abs(float(rows[0][2]) - 0.3) < 1e-4, rows[0]
    assert int(rows[0][3]) == 2
    assert abs(float(rows[0][4]) - 1.12) < 1e-6, rows[0]
    assert int(rows[0][5]) == 100_000


def test_trailing_heading_is_placed_by_its_itow():
    # A NAV-RELPOSNED one epoch behind the latest NAV-PVT belongs 100 ms
    # before that fix, whichever IMU sample it arrived after.
    frames = [_imu(2_000_000), _nav_pvt(200_000),
              _imu(2_100_000), _nav_pvt(200_100),
              _imu(2_101_250), _relposned(200_000, 10.0, 0.2)]
    n, out = _convert(frames)
    assert n["heading"] == 1, n
    rows = _read_rows(os.path.join(out, "heading.csv"))
    assert int(rows[0][0]) == 2_100_000 - 100_000, rows


def test_unusable_relposned_epochs_are_dropped_and_counted():
    frames = [_imu(3_000_000), _nav_pvt(300_000),
              _relposned(300_000, 5.0, 0.2, moving=False),        # static base
              _relposned(300_000, 5.0, 0.2, heading_valid=False),
              _relposned(300_000, 5.0, 0.0),                      # no accHeading
              _relposned(305_000, 5.0, 0.2),                      # 5 s off
              _relposned(300_000, 5.0, 0.2, carr_soln=1)]         # float: kept
    n, out = _convert(frames)
    assert n["heading_not_moving_base"] == 1, n
    assert n["heading_invalid"] == 1, n
    assert n["heading_no_accuracy"] == 1, n
    assert n["heading_itow_mismatch"] == 1, n
    assert n["heading"] == 1, n
    rows = _read_rows(os.path.join(out, "heading.csv"))
    assert int(rows[0][3]) == 1, rows   # the replay's require_fixed decides


def test_no_relposned_writes_no_heading_file():
    n, out = _convert([_imu(4_000_000), _nav_pvt(400_000)])
    assert n["heading"] == 0
    assert not os.path.exists(os.path.join(out, "heading.csv"))


def test_generated_config_section_is_accepted_by_the_replay():
    frames = [_imu(5_000_000), _nav_pvt(500_000),
              _relposned(500_000, 90.0, 0.2)]
    n, out = _convert(frames)
    conv.write_cfg(os.path.join(out, "config.yaml"), "t", False,
                   (0.0, 0.0, 0.0), has_heading=n["heading"] > 0,
                   heading_length_m=n["heading_length_m"])
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
