#!/usr/bin/env python3
"""Offline Rauch-Tung-Striebel (RTS) smoother for the ins error-state filter.

An example general RTS MATLAB implementation is found here:

    https://github.com/jnz/KFCore/blob/main/matlab/kalman_rts.m

(Note: error state attitude requires extra attention)

The forward pass is the filter itself: RtsRecorder.step() replaces the
replay's nav.update() by its two halves, predict() and correct(), and
records what the backward pass needs at every epoch that moved the
covariance (a "knot"):

* Phi, the transition matrix of the covariance prediction this epoch
  (identity when the prediction was throttled or skipped),
* P(k|k-1) between the halves and P(k|k) after the correction,
* the nominal state before and after the correction.

Every IMU epoch additionally records its nominal state, so the smoothed
solution is available at the full IMU rate.

Error-state convention (src/ins.c, ins_apply_correction): dx = nominal -
truth, attitude as R_nominal = (I + [psi x]) R_true, state order pos_ned
vel_ned rpy acc_bias gyr_bias [mag_bias]. Position is INS's own local NED
frame (position_local).

Backward pass per knot k (with the knot after it, k+1):

    C_k    = P(k|k) Phi(k+1)' P(k+1|k)^-1
    dx_k   = C_k (dxm_k+1 + dx_k+1)
    Ps_k   = P(k|k) + C_k (Ps_k+1 - P(k+1|k)) C_k'

where dx is the smoothed error relative to the filtered (corrected)
nominal state and dxm_k+1 = nominal(k+1|k) [-] nominal(k+1|k+1) moves
the error of knot k+1 onto its predicted nominal state, the one Phi
propagates knot k's nominal state to. The smoothed state is the
filtered nominal state with dx removed, the way ins applies a correction
(attitude through the full rotation, not its small-angle form). Between
two knots the nominal state is the filter's strapdown and the error is
interpolated linearly in time.

Optionally (att_sd_max_deg) it is also cut while the filter's own
attitude 1-sigma is above a limit. The backward pass reuses the forward
pass's linearization, which can fail to carry an attitude correction of
tens of degrees back through a high-dynamics re-acquisition. It does not
always fail though: after a re-acquisition with an unknown heading on a
slow vehicle the smoother recovers the heading back to the re-acquisition
epoch, which is exactly what the limit would throw away. So it is off by
default.

References:

* H. Rauch, F. Tung, and C. Striebel. Maximum likelihood estimates of
  linear dynamic systems. AIAA Journal, 3(8):1445-1450, 1965.
* P. D. Groves, Principles of GNSS, Inertial, and Multisensor Integrated
  Navigation Systems, 2nd ed. Norwood, MA: Artech House, 2013.

(c) Jan Zwiener (jan@zwiener.org)
"""

import math

import numpy as np

from ins_solution import cov6 as _cov6

US_PER_SEC = 1_000_000

# Index blocks of the error state (src/ins.h, INS_IDX_*).
IDX_POS, IDX_VEL, IDX_RPY, IDX_ACC, IDX_GYR, IDX_MAG = 0, 3, 6, 9, 12, 15

# A correction may shrink a variance, never grow it. Above this relative
# growth (float32 round-off of a large update reaches a few 1e-3, a real
# re-initialization or re-acquisition inflates by orders of magnitude) the
# correction is no Bayesian update and cuts the chain.
_VAR_GROWTH_TOL = 0.05

# Per-epoch nominal state columns: pos(3) vel(3) R(9, row-major)
# acc_bias(3) gyr_bias(3) mag_bias(3), the geodetic position ins keeps
# next to pos (lat, lon [rad], h [m]) and the time since the last absolute
# position aiding [ms] (nav.deadreckoning_ms(), negative unknown), the
# quality and the satellite count of the replay's last GNSS fix (gnss.csv,
# 0 unknown).
_C_POS, _C_VEL, _C_R, _C_BA, _C_BG, _C_BM, _C_LLH, _C_DR, _C_Q, _C_NS, _NCOL = (
    0, 3, 6, 15, 18, 21, 24, 27, 28, 29, 30)

# WGS84, for moving a geodetic position by a local NED offset.
_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3


def _skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def rotvec_to_mat(v):
    """exp([v x]) for an (N, 3) array of rotation vectors -> (N, 3, 3)."""
    v = np.atleast_2d(v)
    a = np.linalg.norm(v, axis=1)
    K = np.zeros((len(v), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -v[:, 2], v[:, 1]
    K[:, 1, 0], K[:, 1, 2] = v[:, 2], -v[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -v[:, 1], v[:, 0]
    small = a < 1e-12
    a_safe = np.where(small, 1.0, a)
    s = np.where(small, 1.0, np.sin(a_safe) / a_safe)
    c = np.where(small, 0.5, (1.0 - np.cos(a_safe)) / (a_safe * a_safe))
    return np.eye(3)[None] + s[:, None, None] * K + c[:, None, None] * (K @ K)


def _mat_to_rotvec(R):
    """log(R) as a rotation vector."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    a = math.acos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if a < 1e-9:
        return 0.5 * w
    if math.pi - a < 1e-6:
        # Near 180 deg the antisymmetric part vanishes, take the axis
        # from the symmetric part instead.
        B = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(B), 0.0))
        i = int(np.argmax(axis))
        axis = B[:, i] / axis[i]
        return a * axis / np.linalg.norm(axis)
    return a / (2.0 * math.sin(a)) * w


def state_minus(a, b, n):
    """a [-] b in the filter's error convention: the dx for which b is
    the truth when a is the nominal state. a, b: nominal-state rows
    (_NCOL columns)."""
    dx = np.zeros(n)
    dx[IDX_POS:IDX_POS + 3] = a[_C_POS:_C_POS + 3] - b[_C_POS:_C_POS + 3]
    dx[IDX_VEL:IDX_VEL + 3] = a[_C_VEL:_C_VEL + 3] - b[_C_VEL:_C_VEL + 3]
    # R_a = (I + [psi x]) R_b
    Ra = a[_C_R:_C_R + 9].reshape(3, 3)
    Rb = b[_C_R:_C_R + 9].reshape(3, 3)
    dx[IDX_RPY:IDX_RPY + 3] = _mat_to_rotvec(Ra @ Rb.T)
    dx[IDX_ACC:IDX_ACC + 3] = a[_C_BA:_C_BA + 3] - b[_C_BA:_C_BA + 3]
    dx[IDX_GYR:IDX_GYR + 3] = a[_C_BG:_C_BG + 3] - b[_C_BG:_C_BG + 3]
    if n > IDX_MAG:
        dx[IDX_MAG:IDX_MAG + 3] = a[_C_BM:_C_BM + 3] - b[_C_BM:_C_BM + 3]
    return dx


def rpy_from_rotmat(R):
    """Tait-Bryan ZYX roll/pitch/yaw [rad] of R_b_to_n, (..., 3, 3) ->
    (..., 3)."""
    R = np.asarray(R)
    roll = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    pitch = -np.arcsin(np.clip(R[..., 2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    return np.stack([roll, pitch, yaw], axis=-1)


def ned_to_ecef_rot(lat, lon):
    """C_n^e (NED -> ECEF) as an array, replay.ned_to_ecef_rot()."""
    import replay  # noqa: E402  (tools/, lazy)
    return np.array(replay.ned_to_ecef_rot(lat, lon))


def wrap_pi(a):
    """replay.wrap_pi() for arrays: angles into [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class _Rows:
    """Growable float64 table (doubling), so the per-epoch state does not
    live in one Python object per value."""

    def __init__(self, ncol, cap=4096):
        self.a = np.empty((cap, ncol))
        self.n = 0

    def append(self, row):
        if self.n == len(self.a):
            self.a = np.concatenate([self.a, np.empty_like(self.a)])
        self.a[self.n] = row
        self.n += 1

    def view(self):
        return self.a[:self.n]


class RtsRecorder:
    """Forward-pass recorder. Hand it to replay_core.Replay as `stepper`,
    run the replay, then call smooth()."""

    def __init__(self, nav, att_sd_max_deg=None):
        self.nav = nav
        self.att_var_max = (math.radians(att_sd_max_deg) ** 2
                            if att_sd_max_deg is not None else None)
        # Per IMU epoch
        self.t = []                 # t_us
        self.x = _Rows(_NCOL)       # nominal state after the correction (NaN: no ins)
        self.knot_of = []           # last knot at or before this epoch, -1 none
        # Per knot. Matrices float32 (the filter's own precision), a knot
        # whose prediction was skipped shares the previous P(k|k) object
        # as its P(k|k-1) and the one identity as its Phi.
        self.k_epoch = []
        self.k_phi = []
        self.k_Pm = []
        self.k_Pp = []
        self.k_xm = []              # nominal state before the correction
        self.k_link = []            # chained to the previous knot
        self.k_origin = []          # index into self.origins
        self.origins = []           # ins's local origins (ECEF), in order
        self._eye = {}
        self._P_last = None
        self._origin = None
        self._n = None
        self._ready_run = False
        self.n_cuts = 0
        self.n_att_cuts = 0
        self.smoothed = None

    def _nominal(self, full=True):
        """The nominal state as a _NCOL row, None without ins. full=False
        leaves out what only the corrected state needs (geodetic position,
        dead reckoning time, NaN then), for the read before correct()."""
        nav = self.nav
        pos = nav.position_local()
        if pos is None:
            return None
        vel = nav.velocity_ned()
        r9 = nav.rotmat_b_to_n()
        ba = nav.bias_acc()
        bg = nav.bias_gyr()
        if vel is None or r9 is None or ba is None or bg is None:
            return None
        bm = (nav.bias_mag() if self._n is not None and self._n > IDX_MAG
              else None) or (0.0, 0.0, 0.0)
        llh = ((nav.position_llh() if full else None)
               or (math.nan, math.nan, math.nan))
        row = np.empty(_NCOL)
        row[_C_POS:_C_POS + 3] = pos
        row[_C_VEL:_C_VEL + 3] = vel
        # rotmat_b_to_n() is column-major, stored here row-major
        row[_C_R:_C_R + 9] = np.array(r9).reshape(3, 3, order="F").ravel()
        row[_C_BA:_C_BA + 3] = ba
        row[_C_BG:_C_BG + 3] = bg
        row[_C_BM:_C_BM + 3] = bm
        row[_C_LLH:_C_LLH + 3] = llh
        row[_C_DR] = nav.deadreckoning_ms() if full else math.nan
        return row

    def _identity(self, n):
        if n not in self._eye:
            self._eye[n] = np.eye(n, dtype=np.float32)
        return self._eye[n]

    def step(self, r):
        """One epoch: predict, record, correct, record. `r` is the Replay
        (only r.t is used)."""
        nav = self.nav
        phi = nav.predict()
        Pm = nav.covariance() if phi is not None else None
        xm = self._nominal(full=False)
        nav.correct()
        x = self._nominal()

        idx = len(self.t)
        self.t.append(r.t)
        if x is not None:
            fix = getattr(r, "last_fix", None)
            x[_C_Q] = int(fix.get("quality", 0)) if fix is not None else 0
            x[_C_NS] = int(fix.get("num_sv", 0)) if fix is not None else 0
        if x is None or xm is None:
            self.x.append(np.nan)
            self.knot_of.append(-1)
            self._ready_run = False
            self._P_last = None
            return
        self.x.append(x)

        # Any state the correction moved, the biases included: a bias-only
        # update still changes P and has to become a knot.
        changed = not np.array_equal(x[:_C_LLH], xm[:_C_LLH])
        if phi is None and not changed and self._P_last is not None:
            self.knot_of.append(self.knot_of[-1])
            return

        Pp = np.array(nav.covariance(), dtype=np.float32)
        n = Pp.shape[0]
        Pm = (np.array(Pm, dtype=np.float32) if Pm is not None
              else self._P_last if self._P_last is not None else Pp)
        origin = nav.origin_ecef()
        link = (self._ready_run and self._P_last is not None
                and n == self._n and origin == self._origin)
        # A correction that grows a variance is no Bayesian update.
        if np.any(np.diag(Pp) > np.diag(Pm) * (1.0 + _VAR_GROWTH_TOL) + 1e-12):
            link = False
        if (link and self.att_var_max is not None
                and np.max(np.diag(self._P_last)[IDX_RPY:IDX_RPY + 3]) > self.att_var_max):
            link = False
            self.n_att_cuts += 1
        if self.k_link and not link:
            self.n_cuts += 1
        if not self.origins or tuple(origin) != self.origins[-1]:
            self.origins.append(tuple(origin))
        self._n, self._origin = n, origin
        self.k_epoch.append(idx)
        self.k_phi.append(self._identity(n) if phi is None
                          else np.array(phi, dtype=np.float32))
        self.k_Pm.append(Pm)
        self.k_Pp.append(Pp)
        self.k_xm.append(xm)
        self.k_link.append(link)
        self.k_origin.append(len(self.origins) - 1)
        self.knot_of.append(len(self.k_epoch) - 1)
        self._P_last = Pp
        self._ready_run = True

    # ------------------------------------------------------------------
    def smooth(self):
        """Backward pass. Returns (and keeps in self.smoothed) a dict of
        per-IMU-epoch arrays (NaN where ins had no solution): t [us], pos
        (local NED), vel, R (b to n), rpy, acc_bias, gyr_bias, mag_bias,
        dx (the smoothed error removed from the filtered nominal state)
        sd (1-sigma of the smoothed covariance at the epoch's knot),
        origin (index into self.origins, the local frame pos is in, -1
        where invalid), llh (lat, lon [rad], ellipsoidal h [m]), pos_cov
        and vel_cov (the knot's smoothed NED covariance blocks, nn ne nd ee
        ed dd) and dr_s (seconds since the last absolute position aiding
        on the forward pass, NaN unknown), fix_q and fix_ns (the quality
        and the satellite count of the last GNSS fix there, 0 unknown)."""
        K = len(self.k_epoch)
        X = self.x.view()
        nmax = max((P.shape[0] for P in self.k_Pp), default=15)
        dxs = np.zeros((K, nmax))
        dxm = np.zeros((K, nmax))
        sds = np.zeros((K, nmax))
        # nn ne nd ee ed dd of the smoothed position / velocity covariance
        pcov = np.zeros((K, 6))
        vcov = np.zeros((K, 6))
        for k in range(K):
            n = self.k_Pp[k].shape[0]
            dxm[k, :n] = state_minus(self.k_xm[k], X[self.k_epoch[k]], n)
        Ps_next = None
        for k in range(K - 1, -1, -1):
            Pp = self.k_Pp[k].astype(float)
            n = Pp.shape[0]
            nxt = k + 1
            if nxt >= K or not self.k_link[nxt]:
                Ps = Pp
            else:
                Pm1 = self.k_Pm[nxt].astype(float)
                phi = self.k_phi[nxt].astype(float)
                # C = Pp Phi' Pm1^-1  <=>  C' = Pm1^-1 Phi Pp (Pm1 symmetric)
                try:
                    C = np.linalg.solve(Pm1, phi @ Pp).T
                except np.linalg.LinAlgError:
                    C = (np.linalg.pinv(Pm1) @ phi @ Pp).T
                dxs[k, :n] = C @ (dxm[nxt, :n] + dxs[nxt, :n])
                Ps = Pp + C @ (Ps_next - Pm1) @ C.T
            sds[k, :n] = np.sqrt(np.maximum(np.diag(Ps), 0.0))
            pcov[k] = _cov6(Ps, IDX_POS)
            vcov[k] = _cov6(Ps, IDX_VEL)
            Ps_next = Ps

        # Per IMU epoch: the error of its knot, or between knot k
        # (corrected) and a linked knot k+1 (predicted) interpolated.
        N = len(self.t)
        t = np.array(self.t, dtype=np.int64)
        kof = np.array(self.knot_of, dtype=np.int64)
        valid = (kof >= 0) & np.isfinite(X[:, 0])
        k = np.where(valid, kof, 0)
        k_epoch = np.array(self.k_epoch, dtype=np.int64)
        k_link = np.array(self.k_link + [False], dtype=bool)
        dx = dxs[k] if K else np.zeros((N, nmax))
        if K:
            nxt = np.minimum(k + 1, K - 1)
            between = (valid & (k_epoch[k] != np.arange(N)) & k_link[k + 1]
                       & (k + 1 < K))
            t0 = t[k_epoch[k]]
            t1 = t[k_epoch[nxt]]
            w = np.where(between & (t1 > t0),
                         (t - t0) / np.maximum(t1 - t0, 1), 0.0)[:, None]
            dx = np.where(between[:, None],
                          (1.0 - w) * dxs[k] + w * (dxm[nxt] + dxs[nxt]), dx)
        dx[~valid] = np.nan

        Rf = X[:, _C_R:_C_R + 9].reshape(N, 3, 3)
        R = rotvec_to_mat(-np.nan_to_num(dx[:, IDX_RPY:IDX_RPY + 3])) @ Rf
        R[~valid] = np.nan
        # The geodetic position moved by the same NED correction, with the
        # local radii of curvature, the way ins books a correction into
        # its own latlonh.
        lat, lon, h = (X[:, _C_LLH + i] for i in range(3))
        sl = np.sin(lat)
        den = 1.0 - _WGS84_E2 * sl * sl
        r_m = _WGS84_A * (1.0 - _WGS84_E2) / (den * np.sqrt(den))
        r_n = _WGS84_A / np.sqrt(den)
        llh = np.stack([lat - dx[:, IDX_POS] / (r_m + h),
                        lon - dx[:, IDX_POS + 1] / ((r_n + h) * np.cos(lat)),
                        h + dx[:, IDX_POS + 2]], axis=1)
        out = {
            "t": t,
            "valid": valid,
            "dx": dx,
            "pos": X[:, _C_POS:_C_POS + 3] - dx[:, IDX_POS:IDX_POS + 3],
            "vel": X[:, _C_VEL:_C_VEL + 3] - dx[:, IDX_VEL:IDX_VEL + 3],
            "R": R,
            "rpy": rpy_from_rotmat(R),
            "acc_bias": X[:, _C_BA:_C_BA + 3] - dx[:, IDX_ACC:IDX_ACC + 3],
            "gyr_bias": X[:, _C_BG:_C_BG + 3] - dx[:, IDX_GYR:IDX_GYR + 3],
            "mag_bias": (X[:, _C_BM:_C_BM + 3] - dx[:, IDX_MAG:IDX_MAG + 3]
                         if nmax > IDX_MAG else np.full((N, 3), np.nan)),
            "sd": np.where(valid[:, None], sds[k] if K else np.nan, np.nan),
            "origin": np.where(valid, np.array(self.k_origin + [-1])[k], -1),
            "llh": llh,
            "pos_cov": np.where(valid[:, None], pcov[k] if K else np.nan, np.nan),
            "vel_cov": np.where(valid[:, None], vcov[k] if K else np.nan, np.nan),
            "dr_s": np.where(valid & (X[:, _C_DR] >= 0), X[:, _C_DR] / 1e3, np.nan),
            "fix_q": np.where(valid, np.nan_to_num(X[:, _C_Q]), 0).astype(int),
            "fix_ns": np.where(valid, np.nan_to_num(X[:, _C_NS]), 0).astype(int),
        }
        self.smoothed = out
        return out

    def filtered(self):
        """The forward solution in the same layout as smooth()'s pos, vel,
        R, rpy (for side-by-side use)."""
        X = self.x.view()
        N = len(X)
        R = X[:, _C_R:_C_R + 9].reshape(N, 3, 3)
        return {"t": np.array(self.t, dtype=np.int64),
                "pos": X[:, _C_POS:_C_POS + 3], "vel": X[:, _C_VEL:_C_VEL + 3],
                "R": R, "rpy": rpy_from_rotmat(R)}

    def memory_mb(self):
        """Size of the recorded forward pass [MB]."""
        ids = set()
        b = self.x.a.nbytes
        for lst in (self.k_phi, self.k_Pm, self.k_Pp):
            for m in lst:
                if id(m) not in ids:
                    ids.add(id(m))
                    b += m.nbytes
        b += len(self.k_xm) * _NCOL * 8
        return b / 1e6


class EpochScore:
    """Replay observer that keeps every scored epoch (the epochs the
    replay's own position score uses) with what is needed to score it
    again afterwards: the reference, the replay's own position error
    (ECEF), the filter's velocity and attitude, and, with `rts`, the
    epoch's index in the recorder, so the smoothed solution can be scored
    at the same epochs (score_rts())."""

    def __init__(self, rts=None):
        self.rts = rts
        self.rows = []

    def on_epoch(self, r):
        pass

    def on_finish(self, r):
        pass

    def on_pos_error(self, r, err_ecef):
        nav = r.nav
        llh = nav.position_llh()
        r9 = nav.rotmat_b_to_n()
        vel = nav.velocity_ned()
        if llh is None or r9 is None or vel is None:
            return
        self.rows.append({
            "i": len(self.rts.t) - 1 if self.rts is not None else None,
            "ref": r.ref_now,
            "err": np.array(err_ecef, dtype=float),
            "vel": np.array(vel, dtype=float),
            "R": np.array(r9, dtype=float).reshape(3, 3, order="F"),
            "lat": llh[0], "lon": llh[1]})


def _ref_rpy(ref):
    return np.array([ref["roll_rad"], ref["pitch_rad"], ref["yaw_rad"]])


def _score_block(p, v, a):
    p, v, a = np.array(p), np.array(v), np.array(a).reshape(-1, 3)
    if not len(p):
        return {"n": 0}
    d = {"n": int(len(p)),
         "pos_rms": float(np.sqrt(np.mean(p ** 2))), "pos_max": float(p.max()),
         "vel_rms": float(np.sqrt(np.mean(v ** 2))), "vel_max": float(v.max())}
    for j, name in enumerate(("roll", "pitch", "yaw")):
        d[name] = (float(a[:, j].mean()), float(a[:, j].std()),
                   float(np.abs(a[:, j]).max()))
    return d


def score_filter(score):
    """Accuracy of the filter at an EpochScore's epochs: {n, pos_rms,
    pos_max [m], vel_rms, vel_max [m/s], roll/pitch/yaw: (mean, std,
    max_abs) [deg]}."""
    p, v, a = [], [], []
    for row in score.rows:
        ref = row["ref"]
        p.append(np.linalg.norm(row["err"]))
        v.append(np.linalg.norm(row["vel"] - np.array(ref["vel_ned"], dtype=float)))
        a.append(np.degrees(wrap_pi(rpy_from_rotmat(row["R"]) - _ref_rpy(ref))))
    return _score_block(p, v, a)


def score_rts(score, score_leverarm_frd=(0.0, 0.0, 0.0)):
    """The same accuracy figures as score_filter() for the smoothed
    solution, at the epochs of an EpochScore recorded on the smoother's
    own forward pass. The smoothed position error is the replay's error
    moved by the smoothing correction, both in ECEF, the score lever arm
    rotated with the smoothed instead of the filtered attitude."""
    rts = score.rts
    sm = rts.smoothed if rts.smoothed is not None else rts.smooth()
    la = np.array(score_leverarm_frd, dtype=float)
    p, v, a = [], [], []
    for row in score.rows:
        i = row["i"]
        if not sm["valid"][i]:
            continue
        ref = row["ref"]
        dx = sm["dx"][i]
        err = (row["err"]
               - ned_to_ecef_rot(row["lat"], row["lon"]) @ dx[IDX_POS:IDX_POS + 3]
               + ned_to_ecef_rot(ref["lat_rad"], ref["lon_rad"])
               @ ((sm["R"][i] - row["R"]) @ la))
        p.append(np.linalg.norm(err))
        v.append(np.linalg.norm(sm["vel"][i] - np.array(ref["vel_ned"], dtype=float)))
        a.append(np.degrees(wrap_pi(sm["rpy"][i] - _ref_rpy(ref))))
    return _score_block(p, v, a)


def smoother_spec(spec):
    """The config the smoother's forward pass runs with: the dataset's
    own, except that ins never gives up dead reckoning. The coasting
    window that makes the real-time filter stop (and re-acquire) after a
    long outage would leave nothing to smooth across the outage. Run
    through, the drift (however large) is what the backward pass removes
    by tying the outage to the fixes on both of its ends."""
    out = dict(spec)
    out["allow_unlimited_deadreckoning"] = 1
    return out


class SmootherRun:
    """Result of run_smoother(): the recorder (rts, with rts.smoothed),
    the forward-pass Replay (replay) and the scored epochs (score)."""

    def __init__(self, rts, replay, score):
        self.rts, self.replay, self.score = rts, replay, score


def run_smoother(spec, inp, eval_start=None, eval_end=None, progress=None,
                 before_epoch=None, att_sd_max_deg=None):
    """The smoother's own forward pass (smoother_spec()) over the loaded
    inputs, then the backward pass. Returns a SmootherRun, or None if
    before_epoch() aborted the forward pass."""
    import replay_core   # noqa: E402  (only needed here, keeps the module light)
    spec_s = smoother_spec(spec)
    nav = replay_core.make_navigator(spec_s, inp, log=lambda *a, **k: None)
    try:
        rts = RtsRecorder(nav, att_sd_max_deg=att_sd_max_deg)
        run = replay_core.Replay(spec_s, inp, nav, eval_start=eval_start,
                                 eval_end=eval_end, stepper=rts)
        score = EpochScore(rts)
        run.observers.append(score)
        run.run(progress=progress, before_epoch=before_epoch)
        if run.aborted:
            return None
        rts.smooth()
        return SmootherRun(rts, run, score)
    finally:
        nav.close()


def score_lines(f, s, n_cuts=None):
    """Summary lines, filter (score_filter()) next to RTS (score_rts())."""
    if not f.get("n") or not s.get("n"):
        return ["RTS smoother: no scored epochs"]
    head = f"RTS smoother vs. filter ({f['n']} / {s['n']} scored epochs"
    if n_cuts is not None:
        head += f", {n_cuts} chain cuts"
    lines = [head + "):",
             f"  {'':10s} {'filter':>12s} {'RTS':>12s}",
             f"  {'pos rms':10s} {f['pos_rms']:12.3f} {s['pos_rms']:12.3f}  m",
             f"  {'pos max':10s} {f['pos_max']:12.3f} {s['pos_max']:12.3f}  m",
             f"  {'vel rms':10s} {f['vel_rms']:12.3f} {s['vel_rms']:12.3f}  m/s",
             f"  {'vel max':10s} {f['vel_max']:12.3f} {s['vel_max']:12.3f}  m/s"]
    for name in ("roll", "pitch", "yaw"):
        lines.append(f"  {name + ' std':10s} {f[name][1]:12.3f} {s[name][1]:12.3f}  deg"
                     f"   (mean {f[name][0]:+.3f} / {s[name][0]:+.3f})")
    return lines


# ----------------------------------------------------------------------------
# Products for the front ends (replay.py, inspostgui.py)
# ----------------------------------------------------------------------------

# rts_* series added to a PlotRecorder history by add_rec_series(), each
# as long as rec["t"] and NaN where there is no smoothed solution.
REC_SERIES = (
    "rts_pos", "rts_pos_ref_pt", "rts_pos_sigma", "rts_vel", "rts_vel_sigma",
    "rts_rpy_deg", "rts_rpy_sigma_deg", "rts_acc_bias", "rts_acc_bias_sigma",
    "rts_gyr_bias", "rts_gyr_bias_sigma", "rts_mag_bias", "rts_mag_bias_sigma",
    "rts_pos_err_ned", "rts_rpy_err_deg")


def pos_in_frame(smr, origin_ecef, origin_lat, origin_lon):
    """The smoothed positions (N, 3) in the local NED frame of a replay
    (origin_* of replay_core.Replay, where its reference and trail are
    drawn): every local origin ins used on the smoother's pass moved the
    way replay_core.ins_pos_in_replay_frame() moves the live estimate."""
    import replay                       # noqa: E402  (tools/, lazy)
    from INSLIB import ecef_to_llh      # noqa: E402
    rts = smr.rts
    sm = rts.smoothed
    pos = sm["pos"].copy()
    if origin_ecef is None:
        return pos
    for o_id, o in enumerate(rts.origins):
        if max(abs(o[i] - origin_ecef[i]) for i in range(3)) < 1e-3:
            continue
        lat, lon, h = ecef_to_llh(*o)
        shift = replay.ref_to_local_ned({"lat_rad": lat, "lon_rad": lon, "h_m": h},
                                        origin_ecef, origin_lat, origin_lon)
        pos[sm["origin"] == o_id] += np.array(shift)
    return pos


def _sample(t_us, valid, t_query_us):
    """Index of the last smoothed epoch at or before each query time and
    whether there is a smoothed solution there."""
    idx = np.searchsorted(t_us, t_query_us, side="right") - 1
    ok = idx >= 0
    idx = np.where(ok, idx, 0)
    return idx, ok & valid[idx]


def add_rec_series(rec, smr, t0_us, origin_ecef, origin_lat, origin_lon,
                   score_leverarm_frd=(0.0, 0.0, 0.0), track_hz=50.0):
    """Add the smoothed solution to a replay_core.PlotRecorder history
    (`rec`, ticks rec["t"] in seconds from t0_us): the REC_SERIES at the
    recorder's ticks, in the replay's local frame, plus rec["rts_track_ne"],
    the North-East ground track at the reference point at track_hz (the
    map pages' own rate, see TrackRecorder). The position error is the
    smoothed reference point minus rec["ref_pos"], the attitude error is
    wrapped to +-180 deg."""
    sm = smr.rts.smoothed
    t_us, valid = sm["t"], sm["valid"]
    pos = pos_in_frame(smr, origin_ecef, origin_lat, origin_lon)
    la = np.array(score_leverarm_frd, dtype=float)
    pos_pt = pos + np.einsum("nij,j->ni", np.nan_to_num(sm["R"]), la)
    t_rec = np.asarray(rec["t"], dtype=float)
    idx, ok = _sample(t_us, valid, t0_us + np.round(t_rec * US_PER_SEC))
    nan3 = np.full((len(idx), 3), np.nan)

    def take(a):
        return np.where(ok[:, None], a[idx], nan3)

    sd = sm["sd"]
    nmax = sd.shape[1]
    out = {
        "rts_pos": take(pos),
        "rts_pos_ref_pt": take(pos_pt),
        "rts_pos_sigma": take(sd[:, IDX_POS:IDX_POS + 3]),
        "rts_vel": take(sm["vel"]),
        "rts_vel_sigma": take(sd[:, IDX_VEL:IDX_VEL + 3]),
        "rts_rpy_deg": np.degrees(take(sm["rpy"])),
        "rts_rpy_sigma_deg": np.degrees(take(sd[:, IDX_RPY:IDX_RPY + 3])),
        "rts_acc_bias": take(sm["acc_bias"]),
        "rts_acc_bias_sigma": take(sd[:, IDX_ACC:IDX_ACC + 3]),
        "rts_gyr_bias": take(sm["gyr_bias"]),
        "rts_gyr_bias_sigma": take(sd[:, IDX_GYR:IDX_GYR + 3]),
        "rts_mag_bias": take(sm["mag_bias"]),
        "rts_mag_bias_sigma": (take(sd[:, IDX_MAG:IDX_MAG + 3]) if nmax > IDX_MAG
                               else nan3),
    }
    out["rts_pos_err_ned"] = out["rts_pos_ref_pt"] - np.asarray(rec["ref_pos"], float)
    ref_rpy = np.asarray(rec["ref_rpy_deg"], dtype=float)
    out["rts_rpy_err_deg"] = (out["rts_rpy_deg"] - ref_rpy + 180.0) % 360.0 - 180.0
    for k, v in out.items():
        rec[k] = v.tolist()
    # North-East track at its own rate, NaN pairs break the line.
    period = US_PER_SEC / max(track_hz, 1e-3)
    tq = np.arange(t_us[0], t_us[-1] + 1, period) if len(t_us) else np.empty(0)
    ti, tok = _sample(t_us, valid, tq)
    rec["rts_track_ne"] = [(float(pos_pt[i, 0]), float(pos_pt[i, 1])) if o
                           else (math.nan, math.nan) for i, o in zip(ti, tok)]


def _pick(t_us, valid, period_us):
    """Indices of the smoothed epochs to keep at one per period_us of sim
    time (all valid ones for period_us <= 0): the first valid epoch at or
    after each point of a period_us grid."""
    vi = np.nonzero(valid)[0]
    if not len(vi) or period_us <= 0:
        return vi
    tv = t_us[vi]
    grid = np.arange(tv[0], tv[-1] + 1, period_us)
    k = np.unique(np.searchsorted(tv, grid))
    return vi[k[k < len(vi)]]


def trail(smr, origin_ecef, origin_lat, origin_lon, period_us=40_000):
    """(NED points, breaks) of the smoothed IMU position in the replay's
    frame for a 3D view, one point per period_us of sim time, a break
    (True) where the line starts again after epochs without a smoothed
    solution."""
    sm = smr.rts.smoothed
    pos = pos_in_frame(smr, origin_ecef, origin_lat, origin_lon)
    keep = _pick(sm["t"], sm["valid"], period_us)
    if not len(keep):
        return [], []
    # A break wherever an invalid epoch lies between two kept ones.
    n_invalid = np.cumsum(~sm["valid"])
    gap = np.empty(len(keep), dtype=bool)
    gap[0] = True
    gap[1:] = n_invalid[keep[1:]] > n_invalid[keep[:-1]]
    return [tuple(map(float, p)) for p in pos[keep]], gap.tolist()


def kml_track(smr, t0_us, rate_hz=2.0):
    """The smoothed solution in replay_core.KmlRecorder's estimate layout
    (t_rel_sec, lat_deg, lon_deg, h_m, roll_deg, pitch_deg, yaw_deg) at
    rate_hz, for ins_kml.write_kml(rts_track=...) and the map views."""
    sm = smr.rts.smoothed
    keep = _pick(sm["t"], sm["valid"], US_PER_SEC / max(rate_hz, 1e-3))
    t = (sm["t"][keep] - t0_us) / US_PER_SEC
    llh = sm["llh"][keep]
    rpy = np.degrees(sm["rpy"][keep])
    return [(float(t[j]), math.degrees(llh[j, 0]), math.degrees(llh[j, 1]),
             float(llh[j, 2]), float(rpy[j, 0]), float(rpy[j, 1]), float(rpy[j, 2]))
            for j in range(len(keep))]


def solution(smr, rate_hz=25.0):
    """The smoothed trajectory as an ins_solution.Solution at rate_hz of
    sim time (0: every IMU epoch), for a solution file."""
    import ins_solution  # noqa: E402  (tools/, lazy)
    sm = smr.rts.smoothed
    keep = _pick(sm["t"], sm["valid"], US_PER_SEC / rate_hz if rate_hz > 0 else 0)
    return ins_solution.Solution(
        sm["t"][keep], sm["llh"][keep], sm["rpy"][keep], sm["vel"][keep],
        sm["pos_cov"][keep], sm["vel_cov"][keep],
        sm["sd"][keep, IDX_RPY:IDX_RPY + 3], sm["dr_s"][keep], sm["fix_q"][keep],
        sm["fix_ns"][keep])
