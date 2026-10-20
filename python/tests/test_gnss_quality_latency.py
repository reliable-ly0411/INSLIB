#!/usr/bin/env python3
"""GNSS fix quality, transport latency, satellites and receiver integrity
in the replay's evaluation.

gnss.csv's optional quality/gps_week/gps_tow_s columns and the dataset's
timesync.csv give every fix its quality (RTKLIB numbering) and its
transport latency (arrival on the recorder's clock minus the receiver's
own GPS time of validity), the num_sv/jamming/spoofing/sig_auth/nav_auth
columns its satellite count and the receiver's interference and
authentication states. All of it shows up in the data quality summary,
as Dr. INS findings (the configured gnss: delay_ms against the measured
latency, an implausible latency as two clocks that do not belong
together, jamming/spoofing indications, failed authentication), on a page
of the --plot PDF and on the GUI's Plots tab.

Runs under pytest or standalone:

    python3 python/tests/test_gnss_quality_latency.py
"""

import io
import contextlib
import os
import re
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "datasets"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ins_solution           # noqa: E402
import replay                 # noqa: E402
import replay_core            # noqa: E402
import replay_format          # noqa: E402
import test_range_stream      # noqa: E402  (its A_ideal dataset)

WEEK, TOW0, PPM = 2431, 323000.0, 40.0
LATENCY_MS = 80.0


def _gps_of(t_us):
    return WEEK * 604800 + TOW0 + t_us / 1e6 / (1.0 + PPM * 1e-6)


def _dataset(d, latency_ms=LATENCY_MS, timesync_offset_s=0.0):
    """A_ideal with quality and GPS time in gnss.csv (every fifth fix
    single point, the rest RTK fixed, each fix valid latency_ms before it
    arrived), 18 satellites (12 on the single point fixes), jamming
    monitored (a warning on every 50th fix), spoofing monitored (none),
    the navigation message authenticated, signal authentication unknown,
    and a timesync.csv, its GPS time shifted by timesync_offset_s (a
    UTC-grid capture is -18 s off)."""
    ds = test_range_stream._dataset(d, stop_after_sec=1e6, with_ranges=False)
    # The file's own spelling (case matters on Linux), as test_range_stream
    # finds it.
    path = os.path.join(ds, next(n for n in os.listdir(ds) if n.lower() == "gnss.csv"))
    fixes = replay.load_gnss(path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(replay_format.GNSS_HEADER)
        for i, fx in enumerate(fixes):
            g = _gps_of(fx["t_us"] - latency_ms * 1000.0)
            week = int(g // 604800)
            q = replay_format.GNSS_Q_SINGLE if i % 5 == 0 else replay_format.GNSS_Q_FIX
            f.write(replay_format.gnss_row(
                fx["t_us"], *(v * 57.29577951308232 for v in (fx["lat_rad"], fx["lon_rad"])),
                fx["h_m"], fx["cov_pos"], fx["vel_ned"], fx["cov_vel"], fx["vel_ok"],
                q, week, g - week * 604800, 12 if i % 5 == 0 else 18,
                replay_format.GNSS_IND_WARNING if i % 50 == 0 else replay_format.GNSS_IND_NONE,
                replay_format.GNSS_IND_NONE, replay_format.GNSS_AUTH_UNKNOWN,
                replay_format.GNSS_AUTH_OK))
    with open(os.path.join(ds, ins_solution.TIMESYNC_FILE), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(replay_format.TIMESYNC_HEADER)
        for i, t in enumerate(range(0, 400_000_000, 60_000_000)):
            g = _gps_of(t) + timesync_offset_s
            week = int(g // 604800)
            f.write(replay_format.timesync_row(t, week, g - week * 604800, 0x0001, i + 1))
    return ds


def _run(ds, delay_ms=None):
    spec, data_dir = replay.load_config(ds)
    if delay_ms is not None:
        spec["gnss"]["delay_ms"] = delay_ms
    inp = replay_core.load_inputs(spec, data_dir, [], log=lambda *a, **k: None)
    nav = replay_core.make_navigator(spec, inp, log=lambda *a, **k: None)
    rec = replay_core.PlotRecorder(5.0)
    run = replay_core.Replay(spec, inp, nav, observers=(rec,))
    with contextlib.redirect_stdout(io.StringIO()):
        run.run()
    lines = []
    run.report(log=lines.append)
    findings = run.findings(rec.rec)
    rec.rec.update(run.plot_extras())
    nav.close()
    return run, lines, findings, rec.rec


def _finding(findings, needle):
    return [(sev, txt) for sev, txt in findings if needle in txt]


def test_quality_and_latency_stats():
    fixes = [{"t_us": 1_000_000 + i * 200_000, "quality": q, "gps_week": WEEK,
              "gps_tow_s": (_gps_of(1_000_000 + i * 200_000 - 50_000)) % 604800}
             for i, q in enumerate([1, 1, 2, 5, 0])]
    counts = replay.gnss_quality_stats(fixes)
    assert counts == {1: 2, 2: 1, 5: 1}
    assert replay.gnss_quality_text(counts) == "fix 50.0%, float 25.0%, single 25.0% (n=4)"
    assert replay.gnss_quality_stats([{"quality": 0}]) is None
    clock = ins_solution.GpsClock([(t, _gps_of(t)) for t in (0, 60_000_000, 120_000_000)])
    lat = replay.gnss_latency_stats(fixes, clock)
    assert lat["n"] == 5 and abs(lat["median"] - 50.0) < 0.01
    assert replay.gnss_latency_stats(fixes, None) is None
    assert replay.gnss_latency_stats([{"t_us": 0, "gps_week": 0}], clock) is None


def test_integrity_stats_and_short_rows():
    fixes = [{"num_sv": n, "jamming": j, "nav_auth": a}
             for n, j, a in ((20, 1, 2), (14, 2, 2), (0, 0, 3), (17, 1, 0))]
    st = replay.gnss_integrity_stats(fixes)
    assert st["num_sv"] == {"median": 17, "min": 14, "n": 3}
    assert st["jamming"] == {1: 2, 2: 1} and st["spoofing"] is None
    assert st["nav_auth"] == {2: 2, 3: 1} and st["sig_auth"] is None
    lines = replay.gnss_integrity_lines(st)
    assert lines[0] == "satellites used: median 17, min 14 (n=3)"
    assert any(ln.startswith("jamming: none 66.7%, warning 33.3%") for ln in lines)
    assert any("nav message authentication: authenticated 66.7%, failed 33.3%" in ln
               for ln in lines)
    assert replay.gnss_integrity_stats([{"quality": 1}]) is None
    # A gnss.csv that stops after num_sv, or after vel_ok: the rest reads
    # as unknown.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "gnss.csv")
        row = replay_format.gnss_row(1000, 49.0, 8.0, 100.0, [1] * 6, (0, 0, 0),
                                     [0.01] * 6, True, 1, WEEK, 1.5, 21, 1, 2, 3, 2)
        cols = row.strip().split(",")
        assert len(cols) == 28
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(replay_format.GNSS_HEADER)
            f.write(row)
            f.write(",".join(cols[:24]) + "\n")
            f.write(",".join(cols[:20]) + "\n")
        a, b, c = replay.load_gnss(path)
    assert (a["num_sv"], a["jamming"], a["spoofing"], a["sig_auth"],
            a["nav_auth"]) == (21, 1, 2, 3, 2)
    assert (b["num_sv"], b["jamming"], b["nav_auth"]) == (21, 0, 0)
    assert (c["quality"], c["num_sv"], c["spoofing"]) == (0, 0, 0)


def test_replay_reports_quality_and_latency():
    with tempfile.TemporaryDirectory() as d:
        ds = _dataset(d)
        run, lines, findings, rec = _run(ds, delay_ms=200)
    text = "\n".join(lines)
    assert re.search(r"gnss fix quality: fix 7\d\.\d%, single 2\d\.\d% \(n=\d+\)", text), text[-1500:]
    m = re.search(r"gnss transport latency: median (-?\d+) ms", text)
    assert m and abs(int(m.group(1)) - LATENCY_MS) <= 1, text[-1500:]
    sev, txt = _finding(findings, "transport latency")[0]
    assert sev == replay.SEV_OK and "configured delay 200 ms" in txt
    assert _finding(findings, "GNSS fix quality")[0][0] == replay.SEV_INFO
    assert len(rec["gnss_q"]) == len(rec["gnss_q_t"]) > 100
    assert len(rec["gnss_lat_ms"]) == len(rec["gnss_lat_t"]) > 100
    assert "gnss satellites used: median 18, min 12" in text, text[-1500:]
    assert re.search(r"gnss jamming: none 9\d\.\d%, warning \d\.\d%", text), text[-1500:]
    assert "gnss nav message authentication: authenticated 100.0%" in text
    assert "signal authentication" not in text
    sev, txt = _finding(findings, "indicated jamming")[0]
    assert sev == replay.SEV_WARN, txt
    assert _finding(findings, "indicated no spoofing")[0][0] == replay.SEV_OK
    assert _finding(findings, "navigation message authentication")[0][0] == replay.SEV_INFO
    assert _finding(findings, "satellites used")[0][0] == replay.SEV_INFO
    assert set(rec["gnss_sv"]) == {12, 18} and len(rec["gnss_sv_t"]) == len(rec["gnss_sv"])
    assert set(rec["gnss_ind"]) == {"jamming", "spoofing", "nav_auth"}
    assert len(rec["gnss_ind"]["jamming"]) == len(rec["gnss_ind_t"])


def test_delay_below_latency_and_implausible_latency_are_findings():
    with tempfile.TemporaryDirectory() as d:
        _run_ds = _dataset(d)
        _, _, findings, _ = _run(_run_ds, delay_ms=20)
    sev, txt = _finding(findings, "below the measured transport latency")[0]
    assert sev == replay.SEV_WARN and "delay_ms 20" in txt
    # A timesync.csv on the receiver's UTC grid: 18 s off.
    with tempfile.TemporaryDirectory() as d:
        _, _, findings, _ = _run(_dataset(d, timesync_offset_s=-18.0), delay_ms=200)
    sev, txt = _finding(findings, "latency implausible")[0]
    assert sev == replay.SEV_WARN and "UTC grid" in txt


def test_pdf_page_and_gui_plots():
    with tempfile.TemporaryDirectory() as d:
        _, _, _, rec = _run(_dataset(d), delay_ms=200)
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("  skip  PDF part, matplotlib not installed")
        else:
            import ins_plots
            fig = ins_plots._gnss_quality_page(rec, "t", 200.0)
            assert fig is not None and len(fig.axes) == 4
            labels = [t.get_text() for t in fig.axes[3].get_yticklabels()]
            assert labels == ["jamming", "spoofing", "nav msg auth"], labels
            assert ins_plots._gnss_quality_page({"t": [1.0]}, "t") is None
            # A UTC-grid capture (all latencies about -18 s) without a
            # configured delay: the axis stays upright and holds the data.
            neg = {"gnss_lat_t": [0.0, 1.0, 2.0],
                   "gnss_lat_ms": [-17940.0, -17930.0, -17920.0]}
            lo, hi = ins_plots._gnss_quality_page(neg, "t").axes[0].get_ylim()
            assert lo < -17940.0 and hi >= 0.0
        try:
            from PyQt6 import QtWidgets
        except ImportError:
            print("  skip  GUI part, PyQt6 not installed")
            return
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        import inspostgui
        import pyqtgraph as pg
        glw = pg.GraphicsLayoutWidget()
        inspostgui.populate_plots(glw, rec, 0.0)
        titles = [item.titleLabel.text for item in glw.ci.items
                  if hasattr(item, "titleLabel")]
        assert any("GNSS fix quality" in t for t in titles), titles
        assert any("GNSS transport latency" in t for t in titles), titles
        assert any("GNSS satellites used" in t for t in titles), titles
        assert any("GNSS integrity" in t for t in titles), titles
        del app


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name)
            fn()
    print("OK")
