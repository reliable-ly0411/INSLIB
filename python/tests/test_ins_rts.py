#!/usr/bin/env python3
"""The offline RTS smoother (tools/ins_rts.py).

* The error-state convention: removing a dx from a nominal state and
  taking the difference again gives the same dx back, attitude included,
  small and large angles alike.
* The backward pass on a linear system it can be checked against: a
  fake navigator that is a plain error-state Kalman filter on a linear
  model, smoothed through the recorder, has to give exactly the textbook
  RTS solution of the same total-state filter. Throttled predictions and
  corrections at a non-propagated epoch are part of the sequence.
* A correction that grows a variance cuts the chain there.
* One whole replay of the simulated A_ideal flight with a GNSS outage:
  the smoother beats the filter, replay.py --smooth prints it and leaves
  the filter's own numbers alone, its --plot-out PDF gets the smoother's
  four pages, the KML/map track agrees with the local solution, and the
  GUI worker hands the same smoothed solution to the window (3D trail,
  plot series, Map tab, panel).

Runs under pytest or standalone:

    python3 python/tests/test_ins_rts.py
"""

import json
import math
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ins_rts                # noqa: E402
import replay                 # noqa: E402
import replay_core            # noqa: E402
import test_range_stream      # noqa: E402  (its A_ideal + ranges dataset)

OUTAGE = "25:20"


def _row(pos, vel, R, ba=(0, 0, 0), bg=(0, 0, 0), bm=(0, 0, 0)):
    row = np.empty(ins_rts._NCOL)
    row[0:3], row[3:6] = pos, vel
    row[6:15] = np.asarray(R).ravel()
    row[15:18], row[18:21], row[21:24] = ba, bg, bm
    return row


def test_state_minus_inverts_the_correction():
    rng = np.random.default_rng(1)
    R_nom = ins_rts.rotvec_to_mat(np.array([[0.3, -0.2, 2.0]]))[0]
    nominal = _row([10, -5, 2], [1, 2, 0.1], R_nom, [0.1] * 3, [0.01] * 3)
    for scale in (1e-4, 0.05, 1.5, 3.1):   # rad, up to near 180 deg
        dx = rng.normal(size=15)
        dx[6:9] *= scale / np.linalg.norm(dx[6:9])
        # The truth the filter's convention implies: nominal minus dx,
        # R_true = exp(-[psi x]) R_nominal.
        R_true = ins_rts.rotvec_to_mat(-dx[None, 6:9])[0] @ R_nom
        truth = _row(nominal[0:3] - dx[0:3], nominal[3:6] - dx[3:6], R_true,
                     nominal[15:18] - dx[9:12], nominal[18:21] - dx[12:15])
        back = ins_rts.state_minus(nominal, truth, 15)
        assert np.allclose(back, dx, atol=1e-9), (scale, back - dx)


class _FakeNav:
    """A linear error-state Kalman filter behind the navigator interface
    RtsRecorder drives: dx = nominal - truth, the nominal state propagated
    with the same Phi as the covariance, measurements of the position.
    Only pos/vel move, the attitude and bias states stay uncorrelated,
    so its estimate is exactly that of the total-state filter run next to
    it (self.kf_*)."""

    N = 15

    def __init__(self, rng, steps):
        self.rng = rng
        n = self.N
        self.x = np.zeros(n)
        self.x[0:6] = rng.normal(size=6)
        self.P = np.eye(n) * 0.01
        self.P[0:6, 0:6] = np.eye(6) * 4.0
        dt = 0.1
        self.Phi = np.eye(n)
        self.Phi[0:3, 3:6] = np.eye(3) * dt
        self.Q = np.zeros((n, n))
        self.Q[3:6, 3:6] = np.eye(3) * 0.02
        self.Q[0:3, 0:3] = np.eye(3) * 1e-4
        self.H = np.zeros((3, n))
        self.H[:, 0:3] = np.eye(3)
        self.Rm = np.eye(3) * 0.25
        # Truth and measurements up front.
        xt = self.x[0:6] + rng.normal(size=6) * 2.0
        self.truth, self.z = [], []
        for _ in range(steps):
            w = rng.multivariate_normal(np.zeros(6), self.Q[0:6, 0:6])
            xt = self.Phi[0:6, 0:6] @ xt + w
            self.truth.append(xt.copy())
            self.z.append(xt[0:3] + rng.multivariate_normal(np.zeros(3), self.Rm))
        self.k = -1
        self.inflate_at = None
        # Bookkeeping for the textbook reference.
        self.kf_xm, self.kf_Pm, self.kf_xp, self.kf_Pp, self.kf_phi = [], [], [], [], []

    def _propagates(self, k):
        return k % 3 != 2           # every third epoch the prediction is throttled

    def _measures(self, k):
        return k % 2 == 0

    def predict(self):
        self.k += 1
        k = self.k
        if not self._propagates(k) and k > 0:
            self.kf_phi.append(np.eye(self.N))
            self.kf_xm.append(self.x.copy())
            self.kf_Pm.append(self.P.copy())
            return None
        self.x = self.Phi @ self.x
        self.P = self.Phi @ self.P @ self.Phi.T + self.Q
        self.kf_phi.append(self.Phi.copy())
        self.kf_xm.append(self.x.copy())
        self.kf_Pm.append(self.P.copy())
        return self.Phi.tolist()

    def correct(self):
        k = self.k
        if self._measures(k):
            S = self.H @ self.P @ self.H.T + self.Rm
            K = self.P @ self.H.T @ np.linalg.inv(S)
            dx = K @ (self.H @ self.x - self.z[k])
            self.x = self.x - dx
            I_KH = np.eye(self.N) - K @ self.H
            self.P = I_KH @ self.P @ I_KH.T + K @ self.Rm @ K.T
        if self.inflate_at == k:
            self.P = self.P * 100.0
        self.kf_xp.append(self.x.copy())
        self.kf_Pp.append(self.P.copy())

    def covariance(self):
        return self.P.tolist()

    def position_local(self):
        return list(self.x[0:3])

    def velocity_ned(self):
        return list(self.x[3:6])

    def rotmat_b_to_n(self):
        return list(np.eye(3).ravel(order="F"))

    def bias_acc(self):
        return list(self.x[9:12])

    def bias_gyr(self):
        return list(self.x[12:15])

    def bias_mag(self):
        return None

    def position_llh(self):
        return None   # optional for the recorder, NaN then

    def deadreckoning_ms(self):
        return 0 if self._measures(self.k) else 100

    def origin_ecef(self):
        return [4e6, 6e5, 4.9e6]


class _Epoch:
    def __init__(self, t):
        self.t = t


def _textbook_rts(nav):
    """Plain RTS over the fake's own forward pass (float64)."""
    K = len(nav.kf_xp)
    xs, Ps = [None] * K, [None] * K
    xs[-1], Ps[-1] = nav.kf_xp[-1], nav.kf_Pp[-1]
    for k in range(K - 2, -1, -1):
        C = nav.kf_Pp[k] @ nav.kf_phi[k + 1].T @ np.linalg.inv(nav.kf_Pm[k + 1])
        xs[k] = nav.kf_xp[k] + C @ (xs[k + 1] - nav.kf_xm[k + 1])
        Ps[k] = nav.kf_Pp[k] + C @ (Ps[k + 1] - nav.kf_Pm[k + 1]) @ C.T
    return xs, Ps


def _run_fake(steps=120, inflate_at=None):
    nav = _FakeNav(np.random.default_rng(7), steps)
    nav.inflate_at = inflate_at
    rts = ins_rts.RtsRecorder(nav)
    for k in range(steps):
        rts.step(_Epoch(1_000_000 + k * 100_000))
    return nav, rts, rts.smooth()


def test_backward_pass_matches_the_textbook_rts():
    nav, rts, sm = _run_fake()
    assert rts.n_cuts == 0
    xs, Ps = _textbook_rts(nav)
    # Compared at the knots. In between, the recorder interpolates the
    # error over a nominal state the strapdown keeps moving, while this
    # fake holds its state still on a throttled epoch, so only the knots
    # are the same problem.
    ke = np.array(rts.k_epoch)
    assert 0 < len(ke) < len(xs)   # some epochs are no knot
    pos = np.array([xs[i][0:3] for i in ke])
    vel = np.array([xs[i][3:6] for i in ke])
    # float32 covariances in the recorder, so not to the last bit
    assert np.allclose(sm["pos"][ke], pos, atol=2e-4), np.abs(sm["pos"][ke] - pos).max()
    assert np.allclose(sm["vel"][ke], vel, atol=2e-4), np.abs(sm["vel"][ke] - vel).max()
    sd = np.array([np.sqrt(np.diag(Ps[i]))[0:6] for i in ke])
    assert np.allclose(sm["sd"][ke, 0:6], sd, rtol=1e-3, atol=1e-5)
    # And it is a smoother: closer to the truth than the filter.
    truth = np.array([t[0:3] for t in nav.truth])
    filt = np.array([x[0:3] for x in nav.kf_xp])
    assert (np.sqrt(np.mean((sm["pos"][ke] - truth[ke]) ** 2))
            < 0.8 * np.sqrt(np.mean((filt[ke] - truth[ke]) ** 2)))


def test_a_bias_only_correction_is_a_knot():
    # A measurement of an accelerometer bias state alone, uncorrelated with
    # position/velocity/attitude: the correction moves nothing but that
    # bias, and still has to be recorded (its P(k|k)) for the next step.
    steps = 30
    nav = _FakeNav(np.random.default_rng(3), steps)
    nav.H = np.zeros((3, nav.N))
    nav.H[:, 9:12] = np.eye(3)
    nav.z = [np.full(3, 0.05)] * steps
    nav._propagates = lambda k: k % 2 == 0     # odd epochs: no prediction
    nav._measures = lambda k: k % 2 == 1       # ... but a bias update
    rts = ins_rts.RtsRecorder(nav)
    for k in range(steps):
        rts.step(_Epoch(1_000_000 + k * 100_000))
    # Epochs 1, 3, 5, ...: no prediction, a bias update only: knots each.
    for k in range(1, steps, 2):
        assert rts.knot_of[k] >= 0 and rts.k_epoch[rts.knot_of[k]] == k, k
    # and the knot after it starts from that update's P(k|k)
    k1 = rts.knot_of[1]
    assert np.allclose(rts.k_Pm[k1 + 1], nav.kf_Pm[2], atol=1e-6)


def test_a_growing_variance_cuts_the_chain():
    cut = 60
    nav, rts, sm = _run_fake(inflate_at=cut)
    assert rts.n_cuts == 1
    # The knot of the inflating epoch starts a new segment, the one before
    # it ends the old one and keeps the filtered value.
    k_cut = rts.knot_of[cut]
    assert not rts.k_link[k_cut]
    i_end = rts.k_epoch[k_cut - 1]
    assert np.allclose(sm["dx"][i_end], 0.0)
    assert np.allclose(sm["pos"][i_end], nav.kf_xp[i_end][0:3])
    # Before the cut it is still smoothed.
    assert np.abs(sm["dx"][: i_end - 5, 0:3]).max() > 1e-3


def _dataset(d):
    return test_range_stream._dataset(d, stop_after_sec=1e6, with_ranges=False)


def test_smoother_beats_the_filter_through_an_outage():
    with tempfile.TemporaryDirectory() as d:
        ds = _dataset(d)
        spec, data_dir = replay.load_config(ds)
        inp = replay_core.load_inputs(spec, data_dir,
                                      replay_core.parse_gnss_outages([OUTAGE]),
                                      log=lambda *a, **k: None)
        nav = replay_core.make_navigator(spec, inp, log=lambda *a, **k: None)
        run = replay_core.Replay(spec, inp, nav)
        fs = ins_rts.EpochScore()
        run.observers.append(fs)
        run.run()
        nav.close()
        smr = ins_rts.run_smoother(spec, inp)
    f = ins_rts.score_filter(fs)
    s = ins_rts.score_rts(smr.score, spec["score"].get("leverarm_frd") or (0, 0, 0))
    assert f["n"] > 100 and s["n"] >= f["n"]
    # The filter's own score is the replay's.
    assert math.isclose(f["pos_rms"], run.e_pos.rms(), rel_tol=1e-9)
    assert s["pos_rms"] < f["pos_rms"], (s["pos_rms"], f["pos_rms"])
    assert s["pos_max"] < f["pos_max"], (s["pos_max"], f["pos_max"])
    # The smoother's own forward pass never stops dead reckoning, and the
    # dataset's config is left as it is.
    assert smr.replay.spec["allow_unlimited_deadreckoning"] == 1
    plain = dict(spec, allow_unlimited_deadreckoning=0)
    assert ins_rts.smoother_spec(plain)["allow_unlimited_deadreckoning"] == 1
    assert plain["allow_unlimited_deadreckoning"] == 0


def _have(module):
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def test_replay_py_smooth_keeps_the_filter_numbers():
    plot = _have("matplotlib")
    with tempfile.TemporaryDirectory() as d:
        ds = _dataset(d)
        out = {}
        for smooth in (False, True):
            js = os.path.join(d, f"s{int(smooth)}.json")
            cmd = [sys.executable, os.path.join(REPO, "tools", "replay.py"), ds,
                   "--gnss-outage", OUTAGE, "--summary-json", js]
            if plot:
                cmd += ["--plot-out", os.path.join(d, f"s{int(smooth)}.pdf")]
            if smooth:
                cmd.append("--smooth")
            p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                               encoding="utf-8", errors="replace")
            assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
            with open(js, encoding="utf-8") as f:
                out[smooth] = (json.load(f), p.stdout)
    plain, smoothed = out[False][0], out[True][0]
    assert smoothed["pos_rms_m"] == plain["pos_rms_m"]
    assert smoothed["scored_epochs"] == plain["scored_epochs"]
    assert "rts" not in plain
    assert smoothed["rts"]["rts"]["pos_rms"] < smoothed["rts"]["filter"]["pos_rms"]
    assert "RTS smoother vs. filter" in out[True][1]
    if plot:
        # The smoother's pages: trajectory, state history, error vs. the
        # reference, IMU bias.
        pages = [int(re.search(r"\.pdf \((\d+) pages\)", out[s][1]).group(1))
                 for s in (False, True)]
        assert pages[1] == pages[0] + 4, pages
    else:
        print("  skip  PDF part, matplotlib not installed")


def test_kml_track_is_the_smoothed_solution():
    with tempfile.TemporaryDirectory() as d:
        ds = _dataset(d)
        spec, data_dir = replay.load_config(ds)
        inp = replay_core.load_inputs(spec, data_dir,
                                      replay_core.parse_gnss_outages([OUTAGE]),
                                      log=lambda *a, **k: None)
        smr = ins_rts.run_smoother(spec, inp)
        r = smr.replay
        track = ins_rts.kml_track(smr, r.t0_us, rate_hz=2.0)
        assert len(track) > 50 and all(len(row) == 7 for row in track)
        # About 2 Hz and in time order.
        dts = np.diff([row[0] for row in track])
        assert np.all(dts > 0) and abs(np.median(dts) - 0.5) < 0.05
        # The geodetic track is the local smoothed solution (to ins's own
        # latlonh vs. pos_local bookkeeping, sub-millimetre on this
        # short flight).
        sm = smr.rts.smoothed
        pos = ins_rts.pos_in_frame(smr, r.origin_ecef, r.origin_lat, r.origin_lon)
        rows = np.nonzero(sm["valid"])[0][::200]
        for i in rows:
            lat, lon, h = sm["llh"][i]
            p = replay.ref_to_local_ned({"lat_rad": lat, "lon_rad": lon, "h_m": h},
                                        r.origin_ecef, r.origin_lat, r.origin_lon)
            assert np.linalg.norm(np.array(p) - pos[i]) < 0.01, (i, p, pos[i])
        if _have("simplekml"):
            import ins_kml
            out = os.path.join(d, "t.kmz")
            ins_kml.write_kml(out, track, [], [], "t", rts_track=track)
            assert os.path.getsize(out) > 0
        else:
            print("  skip  KML file, simplekml not installed")


def test_gui_worker_smooths_and_the_window_shows_it():
    try:
        from PyQt6 import QtCore, QtWidgets
    except ImportError:
        print("  skip  PyQt6 not installed")
        return
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import inspostgui
    with tempfile.TemporaryDirectory() as d:
        ds = _dataset(d)
        raw, _cfg, data_dir = inspostgui.load_raw_config(ds)
        spec = inspostgui.merge_spec(raw)
        worker = inspostgui.ReplayWorker(
            spec, data_dir, smooth=True,
            gnss_outages=replay_core.parse_gnss_outages([OUTAGE]))
        done = {}
        worker.sig_error.connect(lambda msg, details: done.update(error=details or msg))
        worker.sig_finished.connect(lambda res: done.update(results=res))
        worker.run()
        assert "error" not in done, done.get("error")
        res = done["results"]
        rts = res["rts"]
        assert rts is not None and len(rts["trail"]) > 100
        assert rts["rts"]["pos_rms"] < rts["filter"]["pos_rms"]
        assert "RTS smoother vs. filter" in res["text"]
        rec = res["rec"]
        for k in ("rts_pos", "rts_pos_err_ned", "rts_rpy_err_deg", "rts_vel"):
            assert len(rec[k]) == len(rec["t"]), k
        assert np.isfinite(np.asarray(rec["rts_pos_err_ned"], float)).any()

        settings = QtCore.QSettings(os.path.join(d, "gui.ini"),
                                    QtCore.QSettings.Format.IniFormat)
        win = inspostgui.MainWindow(settings)
        assert not win.chk_smooth.isChecked()   # costs run time, off by default
        assert "rts" not in win.legend._shown    # no smoothed trail yet
        win._on_finished(res)
        assert win.pos_view.has_rts()
        assert "rts" in win.legend._shown
        assert "RTS pos rms" in win.lbl_result.text()
        assert len(win.map_view.c_rts.getData()[0]) == len(rts["kml_est"]) > 0
        assert win.map_view.c_rts.isVisible()
        win.chk_rts.setChecked(False)
        assert "rts" not in win.legend._shown
        assert not win.map_view.c_rts.isVisible()
        for k in ins_rts.REC_SERIES + ("rts_track_ne", "rts_lines"):
            assert k in rec, k
        win.plots_widget.resize(1200, 900)
        assert not win.plots_widget.grab().isNull()
        win.close()
    del app


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name)
            fn()
    print("OK")
