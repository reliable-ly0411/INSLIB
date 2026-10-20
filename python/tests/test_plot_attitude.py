"""Self-test for ins_plots' attitude comparison (INSLIB / ARS / AHRS error
against ground truth and against INSLIB, RMS / mean / max statistics). Runs
under pytest or standalone

(c) Jan Zwiener (jan@zwiener.org)
(``python3 python/tests/test_plot_attitude.py``)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "tools"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import ins_plots as ip  # noqa: E402

NAN3 = [math.nan] * 3


def _rec(n=200, with_ref=True):
    t = [0.1 * i for i in range(n)]
    ref = [[2.0 * math.sin(0.1 * x), 1.0 * math.cos(0.1 * x), 10.0 + 5.0 * x]
           for x in t]
    rec = {"t": t, "ref_epoch_count": 5 if with_ref else 0,
           "ref_rpy_deg": ref if with_ref else [NAN3] * n}
    rec["rpy_deg"] = [[r[0] + 0.1, r[1] - 0.1, r[2] + 1.0] for r in ref]
    rec["ars_rpy_deg"] = [[r[0] + 0.5, r[1], math.nan] for r in ref]
    rec["ars_rpy_deg"][100] = [r + 5.0 for r in ref[100][:2]] + [math.nan]
    rec["ahrs_rpy_deg"] = [[r[0], r[1] + 0.2, r[2] - 359.0] for r in ref]
    return rec


def test_stats_vs_ground_truth_and_ins():
    rec = _rec()
    rows = {(r, l, a): s for r, l, a, s in ip._att_stat_rows(rec, 0.0)}
    # INSLIB: constant 0.1 deg roll offset, yaw error wrapped (+1 deg)
    s = rows[("ground truth", "INSLIB", 0)]
    assert abs(s["rms"] - 0.1) < 1e-9 and abs(s["max"] - 0.1) < 1e-9
    assert abs(rows[("ground truth", "INSLIB", 2)]["mean"] - 1.0) < 1e-9
    # AHRS yaw: -359 deg offset wraps to +1 deg
    assert abs(rows[("ground truth", "AHRS", 2)]["mean"] - 1.0) < 1e-9
    # ARS: roll spike of 5 deg at sample 100 is the max, and has no yaw
    s = rows[("ground truth", "ARS", 0)]
    assert abs(s["max"] - 5.0) < 1e-9 and abs(s["tmax"] - 10.0) < 1e-9
    assert ("ground truth", "ARS", 2) not in rows
    # ARS against INSLIB: roll 0.5 - 0.1 = 0.4 offset
    assert abs(rows[("INSLIB", "AHRS", 2)]["mean"] - 0.0) < 1e-9
    assert ("INSLIB", "ARS", 2) not in rows
    print("ok  attitude statistics vs ground truth and vs INSLIB")


def test_warmup_excludes_early_samples():
    rec = _rec()
    rows = {(r, l, a): s for r, l, a, s in ip._att_stat_rows(rec, 15.0)}
    assert abs(rows[("ground truth", "ARS", 0)]["rms"] - 0.5) < 1e-9
    print("ok  attitude statistics honour the warm-up")


def test_no_reference_falls_back_to_ins_only():
    rec = _rec(with_ref=False)
    groups = ip._att_error_series(rec)
    assert [g[0] for g in groups] == ["INSLIB"]
    # position-only truth padded with constant angles is no attitude truth
    rec = _rec()
    rec["ref_rpy_deg"] = [[0.0, 0.0, 90.0]] * len(rec["t"])
    assert ip._att_ref_axes(rec) == ()
    print("ok  padded reference attitude is not used as ground truth")


def test_page_renders():
    fig = ip._attitude_compare_page(_rec(), _rec()["t"], "x", 1.0)
    assert fig is not None
    assert ip._attitude_compare_page({"t": [0.0], "rpy_deg": [NAN3],
                                      "ref_rpy_deg": [NAN3]},
                                     [0.0], "x", 0.0) is None
    print("ok  attitude comparison page renders")


if __name__ == "__main__":
    test_stats_vs_ground_truth_and_ins()
    test_warmup_excludes_early_samples()
    test_no_reference_falls_back_to_ins_only()
    test_page_renders()
