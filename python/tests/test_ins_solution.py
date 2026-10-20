#!/usr/bin/env python3
"""Solution files (tools/ins_solution.py): replay.py --dump-solution and
inspostgui.py's Export solution.

* The GPS time model from timesync.csv: offset and rate of the recording
  clock, a pair on the UTC grid, a suspect pair, a single pair, the
  system clock guess without a file, GPST date strings.
* The RTKLIB .pos layout: column count, ENU sign convention of the
  covariance roots, Q from the GNSS fix quality and the dead reckoning
  time.
* The CSV is a ref.csv: tools/replay.py's own loader reads it back.
* One whole replay of the simulated A_ideal flight: --dump-solution as
  .csv and as .pos, with and without a timesync.csv, with --smooth also
  the _rts file, and the GUI's export of the same run.

Runs under pytest or standalone:

    python3 python/tests/test_ins_solution.py
"""

import datetime
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "datasets"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ins_solution           # noqa: E402
import replay                 # noqa: E402
import replay_format          # noqa: E402
import test_range_stream      # noqa: E402  (its A_ideal dataset)

WEEK = 2431
TOW0 = 323000.0   # GPS time of t_us = 0 in the synthetic timesync.csv
PPM = 40.0        # the synthetic recording clock runs this fast
LEAP = 18
GPS_VALID, UTC_BASE, TP_SUSPECT = 0x0001, 0x0002, 0x0010


def _gps_of(t_us):
    return WEEK * 604800 + TOW0 + t_us / 1e6 / (1.0 + PPM * 1e-6)


def _write_timesync(d, t0_us, t1_us, step_us=60_000_000, flags=GPS_VALID,
                    extra=()):
    """timesync.csv with a pair every step_us. extra: (t_us, gps_s, flags)
    rows as they are. A UTC_BASE row carries its time on the UTC grid."""
    rows = [(t, _gps_of(t), flags) for t in range(t0_us, t1_us, step_us)]
    rows += list(extra)
    with open(os.path.join(d, ins_solution.TIMESYNC_FILE), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(replay_format.TIMESYNC_HEADER)
        for i, (t, g, fl) in enumerate(sorted(rows)):
            if fl & UTC_BASE:
                g -= LEAP
            week = int(g // 604800)
            f.write(replay_format.timesync_row(t, week, g - week * 604800, fl, i + 1))


def test_gps_clock_fits_offset_and_rate():
    with tempfile.TemporaryDirectory() as d:
        _write_timesync(d, 5_000_000, 600_000_000)
        clock = ins_solution.load_gps_clock(d)
    assert clock is not None and clock.n == 10
    for t in (0, 123_456_789, 700_000_000):
        assert abs(clock.gps_s(t) - _gps_of(t)) < 1e-6
    assert abs(clock.ppm() - PPM) < 0.01
    assert clock.resid_rms_s < 1e-6
    assert "GPST from timesync.csv" in clock.describe()
    with tempfile.TemporaryDirectory() as d:
        assert ins_solution.load_gps_clock(d) is None


def test_gps_clock_utc_grid_suspect_and_single_pair():
    # On the UTC grid (UTC_BASE): taken as it is, off by the leap seconds,
    # and the header says so.
    with tempfile.TemporaryDirectory() as d:
        _write_timesync(d, 5_000_000, 600_000_000, flags=GPS_VALID | UTC_BASE)
        clock = ins_solution.load_gps_clock(d)
    assert abs(clock.gps_s(300_000_000) - (_gps_of(300_000_000) - LEAP)) < 1e-6
    assert clock.n_utc == 10 and "WARNING" in clock.describe()
    # A pair a whole second off: skipped by its TP_SUSPECT flag, or, when
    # nothing flagged it, by the refit.
    with tempfile.TemporaryDirectory() as d:
        _write_timesync(d, 5_000_000, 600_000_000, extra=[
            (30_000_000, _gps_of(30_000_000) + 1.0, GPS_VALID | TP_SUSPECT),
            (90_000_000, _gps_of(90_000_000) + 1.0, GPS_VALID)])
        clock = ins_solution.load_gps_clock(d)
    assert clock.n == 10 and clock.n_rejected == 1
    assert abs(clock.gps_s(300_000_000) - _gps_of(300_000_000)) < 1e-6
    # A single pair: the offset, at the nominal rate.
    with tempfile.TemporaryDirectory() as d:
        _write_timesync(d, 5_000_000, 6_000_000)
        clock = ins_solution.load_gps_clock(d)
    assert clock.n == 1 and "offset only" in clock.describe()
    assert abs(clock.gps_s(5_000_000) - _gps_of(5_000_000)) < 1e-6
    assert abs(clock.gps_s(65_000_000) - clock.gps_s(5_000_000) - 60.0) < 1e-9
    # A recorder without the flags column: taken as GPS time.
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, ins_solution.TIMESYNC_FILE), "w",
                  encoding="utf-8", newline="\n") as f:
            f.write("utc_iso,host_unix_s,imu_t_us,timesync_t_us,gps_week,"
                    "gps_tow_s,n_timesync_pairs,segment,mcu_restarts\n")
            f.write("2026-08-12T17:37:35+00:00,1.0,100,,,,0,0,0\n")
            f.write("2026-08-12T17:38:35+00:00,2.0,200,1000000,2431,323000.000000,3,0,0\n")
        clock = ins_solution.load_gps_clock(d)
    assert clock.n == 1 and abs(clock.gps_s(1_000_000) - (WEEK * 604800 + 323000.0)) < 1e-6


def test_gps_clock_refit_bad_rows_and_restart_segments():
    # Two slipped pairs: the fit ends on exactly the pairs it keeps.
    pairs = [(t, _gps_of(t)) for t in range(0, 600_000_000, 60_000_000)]
    pairs[2] = (pairs[2][0], pairs[2][1] + 1.0)
    pairs[5] = (pairs[5][0], pairs[5][1] - 1.0)
    clock = ins_solution.GpsClock(pairs)
    assert clock.n == 8 and clock.n_rejected == 2 and clock.resid_rms_s < 1e-6
    assert abs(clock.gps_s(300_000_000) - _gps_of(300_000_000)) < 1e-6
    with tempfile.TemporaryDirectory() as d:
        _write_timesync(d, 5_000_000, 600_000_000)
        path = os.path.join(d, ins_solution.TIMESYNC_FILE)
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            # a row whose flags do not parse: skipped, not fatal
            f.write(",,700000000,700000000,2431,323700.000000,11,0,0,GPS\n")
            # a second clock segment after an MCU restart, fewer pairs
            for i, t in enumerate((1_000_000, 61_000_000)):
                week = int(_gps_of(t) // 604800)
                f.write(replay_format.timesync_row(t + 7, week, 324000.0 + 60 * i,
                                                   0x0001, 20 + i, 1))
        clock = ins_solution.load_gps_clock(d)
    assert clock.n == 10 and clock.n_segments == 2
    assert "2 clock segments" in clock.describe()
    assert abs(clock.gps_s(300_000_000) - _gps_of(300_000_000)) < 1e-6


def test_guessed_clock_is_the_system_time():
    clock = ins_solution.guessed_clock(1_000_000)
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    gps_now = (now - ins_solution.GPS_EPOCH).total_seconds()
    assert abs(clock.gps_s(1_000_000) - gps_now) < 5.0
    assert clock.guessed and "GUESSED" in clock.describe()


def test_gpst_strings():
    # GPS week 0 starts 1980-01-06, no leap seconds on GPST.
    assert ins_solution.gpst_string(0.0) == "1980/01/06 00:00:00.000"
    assert ins_solution.gpst_string(2431 * 604800 + 323031.25) == "2026/08/12 17:43:51.250"


def _solution():
    pos_cov = [(4.0, 1.0, -0.5, 9.0, 0.25, 16.0)]   # nn ne nd ee ed dd
    vel_cov = [(0.01, 0.0, 0.0, 0.04, -0.01, 0.09)]
    return ins_solution.Solution(
        [1_000_000, 2_000_000], [(math.radians(48.9), math.radians(8.4), 160.0)] * 2,
        [(0.01, -0.02, 1.0)] * 2, [(1.0, 2.0, -0.5)] * 2, pos_cov * 2, vel_cov * 2,
        [(0.001, 0.002, 0.01)] * 2, [0.2, 30.0], [2, 2], [17, 17])


def _pos_rows(path):
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    head = [ln for ln in lines if ln.startswith("%")]
    rows = [ln.split() for ln in lines if not ln.startswith("%")]
    return head, rows


def test_pos_layout():
    sol = _solution()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.pos")
        ins_solution.write(path, sol)               # no clock: guessed
        head, rows = _pos_rows(path)
        assert any("GUESSED" in h for h in head)
        assert "latitude(deg)" in head[-1] and "sdvun" in head[-1]
        # date time lat lon h Q ns sdn sde sdu sdne sdeu sdun age ratio + 9 velocity
        assert all(len(r) == 2 + 3 + 2 + 6 + 2 + 9 for r in rows)
        r = rows[0]
        assert abs(float(r[2]) - 48.9) < 1e-9 and abs(float(r[4]) - 160.0) < 1e-9
        # aided: the fix's quality (2, float), dead reckoning: 5
        assert int(r[5]) == 2 and int(rows[1][5]) == ins_solution.Q_DEADRECKONING
        # ns: the fix's satellites while aided, 0 dead reckoning
        assert int(r[6]) == 17 and int(rows[1][6]) == 0
        sdn, sde, sdu, sdne, sdeu, sdun = map(float, r[7:13])
        assert (sdn, sde, sdu) == (2.0, 3.0, 4.0)
        # Signed roots, u = -d: cov(e,u) = -cov(e,d), cov(u,n) = -cov(d,n)
        assert sdne == 1.0 and sdeu == -0.5 and abs(sdun - math.sqrt(0.5)) < 1e-4
        assert float(rows[1][13]) == 30.0            # age = dead reckoning time
        vn, ve, vu = map(float, r[15:18])
        assert (vn, ve, vu) == (1.0, 2.0, 0.5)
        sdvn, sdve, sdvu, sdvne, sdveu = map(float, r[18:23])
        assert (sdvn, sdve, sdvu, sdvne) == (0.1, 0.2, 0.3, 0.0) and sdveu == 0.1

        clock = ins_solution.GpsClock([(0, WEEK * 604800 + TOW0),
                                       (10_000_000, WEEK * 604800 + TOW0 + 10.0)])
        ins_solution.write(path, sol, clock)
        head, rows = _pos_rows(path)
        assert not any("GUESSED" in h for h in head)
        assert rows[0][0] == "2026/08/12" and rows[0][1] == "17:43:21.000"
        # Q 1 where the fix quality is unknown
        sol.fix_q[:] = 0
        ins_solution.write(path, sol, clock)
        assert int(_pos_rows(path)[1][0][5]) == ins_solution.Q_AIDED


def test_csv_is_a_ref_csv():
    sol = _solution()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.csv")
        ins_solution.write(path, sol)
        ref = replay.load_ref(path)
        with open(path, encoding="utf-8") as f:
            data = [ln for ln in f if not ln.startswith("#")]
    assert len(ref) == 2 and all(len(ln.split(",")) == 19 for ln in data)
    assert ref[0]["t_us"] == 1_000_000
    assert abs(math.degrees(ref[0]["lat_rad"]) - 48.9) < 1e-9
    assert abs(ref[0]["yaw_rad"] - 1.0) < 1e-6 and ref[0]["vel_ned"] == (1.0, 2.0, -0.5)
    sd = [float(v) for v in data[0].split(",")[10:]]
    assert sd[0:3] == [2.0, 3.0, 4.0] and sd[3:6] == [0.1, 0.2, 0.3]


def _replay(ds, *args):
    p = subprocess.run([sys.executable, os.path.join(REPO, "tools", "replay.py"), ds,
                        *args], capture_output=True, text=True, cwd=REPO,
                       encoding="utf-8", errors="replace")
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    return p.stdout


def test_replay_py_dump_solution():
    with tempfile.TemporaryDirectory() as d:
        ds = test_range_stream._dataset(d, stop_after_sec=1e6, with_ranges=False)
        csv_path = os.path.join(d, "sol.csv")
        _replay(ds, "--dump-solution", csv_path, "--dump-solution-hz", "10")
        sol = replay.load_ref(csv_path)
        assert len(sol) > 100
        dt = np.diff([r["t_us"] for r in sol])
        assert abs(np.median(dt) - 100_000) < 5_000
        # .pos without timesync.csv: the time guessed from the system clock
        pos_path = os.path.join(d, "sol.pos")
        _replay(ds, "--dump-solution", pos_path, "--dump-solution-hz", "10")
        head, rows = _pos_rows(pos_path)
        assert any("GUESSED" in h for h in head) and len(rows) == len(sol)
        # with timesync.csv: GPST, the same epochs, and the smoother's file
        _write_timesync(ds, 0, 400_000_000)
        out = _replay(ds, "--dump-solution", pos_path, "--dump-solution-hz", "10",
                      "--smooth")
        assert "gps time: GPST from timesync.csv" in out
        head, rows = _pos_rows(pos_path)
        assert any("GPST from timesync.csv" in h for h in head)
        g = _gps_of(sol[0]["t_us"])
        assert rows[0][0] + " " + rows[0][1] == ins_solution.gpst_string(g)
        rts_path = os.path.join(d, "sol_rts.pos")
        assert os.path.exists(rts_path)
        _, rts_rows = _pos_rows(rts_path)
        assert len(rts_rows) > 100
        # the same position within the flight's error, not a shifted copy
        assert abs(float(rts_rows[50][2]) - float(rows[50][2])) < 1e-4


def test_gui_exports_the_solution():
    try:
        from PyQt6 import QtCore, QtWidgets
    except ImportError:
        print("  skip  PyQt6 not installed")
        return
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import inspostgui
    with tempfile.TemporaryDirectory() as d:
        ds = test_range_stream._dataset(d, stop_after_sec=1e6, with_ranges=False)
        _write_timesync(ds, 0, 400_000_000)
        raw, _cfg, data_dir = inspostgui.load_raw_config(ds)
        spec = inspostgui.merge_spec(raw)
        worker = inspostgui.ReplayWorker(spec, data_dir, smooth=True)
        done = {}
        worker.sig_finished.connect(lambda res: done.update(results=res))
        worker.sig_error.connect(lambda msg, det: done.update(error=det or msg))
        worker.run()
        assert "error" not in done, done.get("error")
        settings = QtCore.QSettings(os.path.join(d, "gui.ini"),
                                    QtCore.QSettings.Format.IniFormat)
        win = inspostgui.MainWindow(settings)
        win._on_finished(done["results"])
        assert win.sol_btn.isEnabled()
        written = win.write_solution(os.path.join(d, "out.pos"))
        assert [os.path.basename(p) for p in written] == ["out.pos", "out_rts.pos"]
        head, rows = _pos_rows(written[0])
        assert any("GPST from timesync.csv" in h for h in head) and len(rows) > 100
        written = win.write_solution(os.path.join(d, "out.csv"))
        assert len(replay.load_ref(written[1])) > 100
        # Without timesync.csv both files share one guessed time base.
        win.results["gps_clock"] = None
        written = win.write_solution(os.path.join(d, "guess.pos"))
        (_, f_rows), (_, r_rows) = (_pos_rows(p) for p in written)
        assert f_rows[0][:2] == r_rows[0][:2]
        # An empty filter solution: the guess comes from the smoother's.
        win.results["solution"] = ins_solution.Solution([], [], [], [], [], [], [], [])
        written = win.write_solution(os.path.join(d, "empty.pos"))
        head, r_rows = _pos_rows(written[1])
        assert any("GUESSED" in h for h in head) and len(r_rows) > 100
        win.close()
    del app


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name)
            fn()
    print("OK")
