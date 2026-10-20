#!/usr/bin/env python3
"""Range stream (ranges.csv) in the replay harnesses (REQ-VER-038).

The format helpers and the loader on their own, then a whole replay: the
simulated A_ideal flight with ranges to four anchors synthesized from its
reference and gnss.csv cut off part way, run through
tools/replay.py and, when it is built, tools/replay.c. Both have to fuse
the same ranges and keep the solution on the reference after the stop.

Runs under pytest or standalone:

    python3 python/tests/test_range_stream.py
"""

import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "datasets"))

import replay                 # noqa: E402
import replay_format as rf    # noqa: E402

A_IDEAL = os.path.join(REPO, "datasets", "simulated", "A_ideal")
C_REPLAY = os.path.join(REPO, "build", "replay")

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def llh_to_ecef(lat_deg, lon_deg, h):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    return ((n + h) * math.cos(lat) * math.cos(lon),
            (n + h) * math.cos(lat) * math.sin(lon),
            (n * (1.0 - WGS84_E2) + h) * math.sin(lat))


def test_range_row_and_loader_round_trip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ranges.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write(rf.RANGES_HEADER)
            f.write(rf.range_row(2_000_000, 11, (4.0e6, 6.0e5, 4.8e6), 12.345, 0.3, ("x", 7)))
            f.write(rf.range_row(1_000_000, 12, (4.0e6, 6.0e5, 4.8e6), 5.0, 0.2))
        rows = replay.load_ranges(path)
    # sorted by time, trailing producer columns ignored
    assert [r[0] for r in rows] == [1_000_000, 2_000_000], rows
    assert rows[1] == (2_000_000, 11, (4.0e6, 6.0e5, 4.8e6), 12.345, 0.3), rows


def test_loader_drops_unusable_rows():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ranges.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write(rf.RANGES_HEADER)
            f.write("1000,1,1,2,3,4.0,0.5\n")      # good
            f.write("2000,1,1,2,3,4.0,0.0\n")      # no 1-sigma
            f.write("3000,1,1,2,3,-1.0,0.5\n")     # negative range
            f.write("4000,70000,1,2,3,4.0,0.5\n")  # id beyond 16 bit
            f.write("5000,1,1,2,3,4.0\n")          # too few columns
        rows = replay.load_ranges(path)
    assert [r[0] for r in rows] == [1000], rows


def test_binding_takes_ranges_up_to_the_epoch_limit():
    """Navigator.range() fills the pending epoch up to INS_RANGE_MAX (4),
    refuses the next one, and the fused entries show in diag()."""
    from INSLIB import Ins, Config
    lat, lon, h = 48.783, 9.181, 300.0
    cfg = Config(lat_rad=math.radians(lat), lon_rad=math.radians(lon), h_m=h,
                 auto_init=False, allow_unlimited_deadreckoning=True)
    here = llh_to_ecef(lat, lon, h)
    anchors = [llh_to_ecef(lat + dn * 1e-4, lon + de * 1e-4, h)
               for dn, de in ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1))]
    with Ins(cfg) as nav:
        t = 0
        for k in range(200):
            t += 10_000
            nav.imu(t, 0.01, (0.0, 0.0, -9.81), (0.0, 0.0, 0.0))
            nav.zupt(True)
            if k == 150:
                took = [nav.range(a, math.dist(here, a), 0.2, 0, i + 1)
                        for i, a in enumerate(anchors)]
                assert took == [True, True, True, True, False], took
                nav.range_leverarm((0.0, 0.0, 0.0))
            nav.update()
        d = nav.diag()
    assert d["n_range_seen"] == 4 and d["n_range_used"] == 4, d
    assert abs(d["last_range_residual_m"]) < 0.05, d


def _dataset(d, stop_after_sec, with_ranges):
    """A_ideal plus ranges.csv to four anchors, 5 Hz each, round robin,
    and gnss.csv without the rows later than stop_after_sec after its
    first one. ref.csv stays whole, so the rest of the run still scores."""
    out = os.path.join(d, "a_ideal_ranges")
    shutil.copytree(A_IDEAL, out)
    gnss_path = os.path.join(out, next(n for n in os.listdir(out) if n.lower() == "gnss.csv"))
    with open(gnss_path, encoding="utf-8") as f:
        lines = f.readlines()
    t_fix = [int(ln.split(",")[0]) for ln in lines if not ln.startswith("#")]
    t_stop = t_fix[0] + int(stop_after_sec * 1e6)
    with open(gnss_path, "w", encoding="utf-8") as f:
        f.writelines(ln for ln in lines if ln.startswith("#") or int(ln.split(",")[0]) <= t_stop)
    ref = []
    with open(os.path.join(out, "ref.csv"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split(",")
            ref.append((int(p[0]), float(p[1]), float(p[2]), float(p[3])))
    lat0, lon0 = ref[0][1], ref[0][2]
    anchors = {}
    for aid, (dn, de, h) in {1: (150.0, 0.0, 5.0), 2: (0.0, 150.0, 20.0),
                             3: (-150.0, 0.0, 0.0), 4: (0.0, -150.0, 12.0)}.items():
        anchors[aid] = llh_to_ecef(lat0 + math.degrees(dn / 6378137.0),
                                   lon0 + math.degrees(de / (6378137.0 * math.cos(math.radians(lat0)))),
                                   h)
    with open(os.path.join(out, "ranges.csv"), "w", encoding="utf-8") as f:
        f.write(rf.RANGES_HEADER)
        k = 0
        for t, lat, lon, h in ref:
            if t % 50_000:
                continue                   # 20 Hz of rows, one anchor each
            aid = 1 + k % 4
            k += 1
            dist = math.dist(llh_to_ecef(lat, lon, h), anchors[aid])
            f.write(rf.range_row(t, aid, anchors[aid], dist, 0.1))
    with open(os.path.join(out, "config.yaml"), "a", encoding="utf-8") as f:
        f.write("\nranges:\n  enable: %d\n  leverarm_frd: [0, 0, 0]\n" % (1 if with_ranges else 0))
    return out


def _run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    return p.stdout + p.stderr


def _range_line(text):
    m = re.search(r"range aiding: (\d+) rows, (\d+) offered .*?, (\d+) fused, (\d+) rejected", text)
    assert m, text[-3000:]
    return tuple(int(g) for g in m.groups())


def _gnss_fusions(text):
    m = re.search(r"(\d+) gnss fusions", text)
    assert m, text[-3000:]
    return int(m.group(1))


def _pos_rms(text):
    m = re.search(r"pos error\s+mean\s+\S+\s+std\s+\S+\s+rms\s+(\S+)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"pos rms: (\S+) m", text)
    assert m, text[-3000:]
    return float(m.group(1))


def test_ranges_carry_the_replay_after_the_gnss_stops():
    with tempfile.TemporaryDirectory() as d:
        ds = _dataset(d, stop_after_sec=20.0, with_ranges=True)
        py = _run([sys.executable, os.path.join(REPO, "tools", "replay.py"), ds])
        rows, offered, fused, rejected = _range_line(py)
        assert rows == offered and fused > 0.8 * offered and rejected == 0, py[-2000:]
        # A_ideal has a GNSS fix every 0.5 s over 64 s, 20 s of them is ~40
        assert _gnss_fusions(py) < 50, py[-2000:]
        rms_ranges = _pos_rms(py)
        assert rms_ranges < 0.1, py[-2000:]
        if os.path.exists(C_REPLAY):
            c = _run([C_REPLAY, ds])
            assert _range_line(c) == (rows, offered, fused, rejected), c[-2000:]
            assert _gnss_fusions(c) == _gnss_fusions(py)
            assert _pos_rms(c) < 0.1, c[-2000:]
    # The same stop without ranges: dead reckoning alone drifts off by
    # metres over the remaining 44 s, so the ranges above did the work.
    with tempfile.TemporaryDirectory() as d:
        ds = _dataset(d, stop_after_sec=20.0, with_ranges=False)
        py = _run([sys.executable, os.path.join(REPO, "tools", "replay.py"), ds])
        assert "range aiding" not in py
        assert _pos_rms(py) > 10.0 * rms_ranges, py[-2000:]


if __name__ == "__main__":
    fails = 0
    for fn in (test_range_row_and_loader_round_trip, test_loader_drops_unusable_rows,
               test_binding_takes_ranges_up_to_the_epoch_limit,
               test_ranges_carry_the_replay_after_the_gnss_stops):
        try:
            fn()
            print("  ok    %s" % fn.__name__)
        except AssertionError as e:
            fails += 1
            print("  FAIL  %s\n%s" % (fn.__name__, e))
    print("==== %d failures ====" % fails)
    sys.exit(1 if fails else 0)
