#!/usr/bin/env python3
"""tools/gnss_faults.py and the GNSS fault and single point replays of
datasets/kfgins (REQ-VER-050): the committed files are exactly what the
script writes, the fault model is the documented one, a step is scored
against the moved position, the SPP error is reported with its accuracy,
and the faults of spp_<type> add to the SPP error. The configs point at
their files and run with the chi2 gate on.

Runs under pytest or standalone:

    python3 python/tests/test_gnss_faults.py
"""

import functools
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import gnss_faults as gf  # noqa: E402
import replay             # noqa: E402

BASE = os.path.join(REPO, "datasets", "kfgins")


@functools.lru_cache(maxsize=None)
def data():
    rows = gf.read_rows(os.path.join(BASE, "gnss_vel.csv"))
    truth = gf.read_rows(os.path.join(BASE, "ref.csv"))
    return gf.fix_times_us(rows, truth), truth


def horizontal(m):
    return math.hypot(m[0], m[1])


def test_committed_files_reproduce():
    assert gf.write_all(BASE, check=True) == 0


def test_event_counts_and_spacing():
    fu, truth = data()
    t_first = gf.motion_start_s(truth) + gf.FAULT_START_AFTER_MOTION_S
    for typ in gf.FAULT_TYPES:
        evs = gf.fault_events(fu, truth, typ)
        steps = typ in ("step", "big_step")
        period = gf.FAULT_STEP_PERIOD_S if steps else gf.FAULT_PERIOD_S
        assert len(evs) >= 5, typ
        assert evs[0]["t0_s"] >= t_first - 1.0, typ
        # a gross event's first blunder lies anywhere in its window
        slack = gf.FAULT_GROSS_WINDOW_S if typ == "gross" else 0.0
        for a, b in zip(evs, evs[1:]):
            assert b["t0_s"] - a["t0_s"] > period - slack - 2.0, typ
        for ev in evs:
            mods = [ev["mods"][u] for u in sorted(ev["mods"])]
            if typ in ("spike", "vel_spike"):
                assert len(mods) == 3, typ
            if typ == "gross":
                # two of the three times can fall on the same fix
                assert len(mods) in (2, 3)
            if typ == "spike":
                assert all(5.0 <= horizontal(m) <= 50.0 and m[3] is None for m in mods)
            if typ == "gross":
                assert all(abs(horizontal(m) - gf.FAULT_GROSS_M) < 1e-6 for m in mods)
                assert ev["t1_s"] - ev["t0_s"] <= gf.FAULT_GROSS_WINDOW_S
            if typ == "vel_spike":
                assert all(m[:3] == (0.0, 0.0, 0.0) and 1.0 <= math.hypot(*m[4][:2]) <= 5.0
                           for m in mods)
            if typ in ("wrong_fix", "multipath"):
                assert len(set(m[:3] for m in mods)) == 1, typ
                assert ev["t1_s"] - ev["t0_s"] < gf.FAULT_DUR_S[1], typ
            if typ == "wrong_fix":
                assert all(0.5 <= horizontal(m) <= 3.0 and m[3] is None for m in mods)
            if typ == "multipath":
                assert all(3.0 <= horizontal(m) <= 15.0 and m[3] == (1.0, 1.0, 2.0) for m in mods)
            if typ == "degraded":
                assert all(m[3] == (1.0, 1.0, 2.0) for m in mods)
            # the vertical offset is the horizontal magnitude times U(-1, 1)
            assert all(abs(m[2]) <= horizontal(m) + 1e-9 for m in mods if typ != "degraded")


def test_step_is_its_own_correct_answer():
    fu, truth = data()
    for typ, (lo, hi) in (("step", gf.FAULT_STEP_M), ("big_step", gf.FAULT_BIG_STEP_M)):
        for ev in gf.fault_events(fu, truth, typ):
            assert ev["shift"] is not None
            assert lo <= horizontal(ev["shift"]) <= hi
            assert all(m[:3] == ev["shift"] and m[3] is None for m in ev["mods"].values())
            assert abs(ev["t1_s"] - ev["t0_s"] - gf.FAULT_STEP_S) < 2.0
    assert all(ev["shift"] is None for ev in gf.fault_events(fu, truth, "wrong_fix"))


def test_spp_accuracy_is_reported():
    fu, _ = data()
    mods = gf.spp_mods(fu)
    sd = (gf.SPP_SD_M[0], gf.SPP_SD_M[0], gf.SPP_SD_M[1])
    assert len(mods) == len(fu)
    assert all(m[3] == sd for m in mods.values())
    x = np.array([mods[int(u)][:3] for u in fu])
    for k in range(3):
        assert 0.6 * sd[k] < np.std(x[:, k]) < 1.4 * sd[k], k
        # 1 Hz fixes: the lag one correlation of a 30 s Gauss-Markov process
        c = np.corrcoef(x[:-1, k], x[1:, k])[0, 1]
        assert abs(c - math.exp(-1.0 / gf.SPP_TAU_S)) < 0.03, (k, c)
    # and the file carries it as the reported covariance
    rows = gf.read_rows(os.path.join(BASE, "gnss_spp.csv"))
    inside = set(int(u) for u in fu)
    for r in rows:
        if int(r[0]) in inside:
            assert np.allclose(r[4:10], [sd[0] ** 2, 0, 0, sd[1] ** 2, 0, sd[2] ** 2])


def test_faults_add_to_the_spp_error():
    fu, truth = data()
    spp = gf.spp_mods(fu)
    for typ in gf.SPP_FAULT_TYPES:
        both = gf.scenario_mods(fu, truth, True, typ)
        faults = {}
        for ev in gf.fault_events(fu, truth, typ):
            faults.update(ev["mods"])
        assert set(both) == set(spp), typ
        for u, m in both.items():
            f = faults.get(u, (0.0, 0.0, 0.0, None, None))
            assert np.allclose(m[:3], np.add(spp[u][:3], f[:3]), rtol=0, atol=1e-12), typ
            assert m[3] == spp[u][3], typ
            assert (m[4] is None) == (f[4] is None), typ


def test_configs_point_at_their_files():
    names = ["config_fault_%s.yaml" % t for t in gf.FAULT_TYPES] + ["config_spp.yaml"] + \
        ["config_spp_%s.yaml" % t for t in gf.SPP_FAULT_TYPES]
    files = [n for n, _, _ in gf.scenarios()]
    for cfg, gnss in zip(names, files):
        spec, _ = replay.load_config(os.path.join(BASE, cfg))
        assert os.path.basename(spec["inputs"]["gnss"]) == gnss, cfg
        assert not spec["chi2_disable"], cfg
        assert spec["score"]["lim_pos_rms_m"] > 0.0, cfg


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok  " + name)
