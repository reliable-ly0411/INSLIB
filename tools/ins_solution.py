#!/usr/bin/env python3
"""Solution files of a replay: the filter's (or the RTS smoother's)
trajectory as a file, written by tools/replay.py --dump-solution and
inspostgui.py's Export solution.

Two formats, selected by the file extension:

* .csv: the dataset ref.csv format (t_us, lat, lon,
  h, roll/pitch/yaw, v_ned, see datasets/replay_format.py REF_HEADER)
  plus the 1-sigma of position, velocity and attitude as trailing
  columns. Both replay tools (CLI and GUI) read the first ten columns only, so
  the file is usable as another run's ref.csv as it is (the smoothed solution
  of a dataset without ground truth, say).
* .pos: the RTKLIB solution format (rnx2rtkp/RTKPOST output, read by
  RTKPLOT): GPST, latitude/longitude/ellipsoidal height, Q, ns, the
  position 1-sigma and the signed square roots of its covariances in
  ENU, age, ratio, then the velocity columns of the RTKLIB demo5 output
  (ENU velocity, its 1-sigma and signed covariance roots). How the RTKLIB
  fields are used for an inertial solution:
    Q      while the position is aided (no more than AIDED_MAX_S since the
           last absolute position aiding) the quality of the last GNSS fix
           (gnss.csv's quality column, the same RTKLIB numbering), Q_AIDED
           where that is unknown, Q_DEADRECKONING while dead reckoning
    ns     while aided the satellite count of the last GNSS fix
           (gnss.csv's num_sv column), 0 where that is unknown and while
           dead reckoning
    age    seconds since the last absolute position aiding
    ratio  0

The time of a .pos line is GPS time. A dataset's t_us is the recording's
own clock (the sensor board's microsecond timer). timesync.csv next to the
dataset's config.yaml maps it onto GPS time (see datasets/replay_format.py
TIMESYNC_HEADER): pairs of a pulse-per-second edge's t_us and the GNSS
time of that edge, fitted with a line here, offset plus rate (a MCU
crystal is tens of ppm off), or the offset alone from a single pair.
Without one the time is a guess: the first epoch is put at the system
clock of the moment the file is written (warning added to header).

(c) Jan Zwiener (jan@zwiener.org)
"""

import csv
import datetime
import math
import os

import numpy as np

US_PER_SEC = 1_000_000
GPS_SEC_PER_WEEK = 604800
GPS_EPOCH = datetime.datetime(1980, 1, 6)

# Q while aided with a fix of unknown quality, and while dead reckoning
# (RTKLIB: 1 fix, 2 float, 3 sbas, 4 dgps, 5 single, 6 ppp).
AIDED_MAX_S = 1.0
Q_AIDED, Q_DEADRECKONING = 1, 5

TIMESYNC_FILE = "timesync.csv"

# The board's 0x40/0x05 flags word (tools/inslib_protocol.md).
TS_UTC_BASE = 0x0002
TS_TP_SUSPECT = 0x0010

CSV_HEADER = ("# t_us, lat_deg, lon_deg, h_m, roll_deg, pitch_deg, yaw_deg,"
              " vn_mps, ve_mps, vd_mps, sd_n_m, sd_e_m, sd_d_m,"
              " sd_vn_mps, sd_ve_mps, sd_vd_mps,"
              " sd_roll_deg, sd_pitch_deg, sd_yaw_deg\n")


class Solution:
    """A trajectory as parallel arrays, N rows: t_us (int64), llh (lat,
    lon [rad], ellipsoidal h [m]), rpy [rad], vel_ned [m/s], pos_cov and
    vel_cov (NED, nn ne nd ee ed dd, [m^2], [(m/s)^2]), rpy_sd [rad],
    dr_s (seconds since the last absolute position aiding, NaN unknown),
    fix_q (RTKLIB quality of the last GNSS fix, 0 unknown) and fix_ns (its
    satellite count, 0 unknown). Any covariance or sigma may be NaN."""

    def __init__(self, t_us, llh, rpy, vel_ned, pos_cov, vel_cov, rpy_sd, dr_s,
                 fix_q=None, fix_ns=None):
        self.t_us = np.asarray(t_us, dtype=np.int64)
        self.llh = np.asarray(llh, dtype=float).reshape(-1, 3)
        self.rpy = np.asarray(rpy, dtype=float).reshape(-1, 3)
        self.vel_ned = np.asarray(vel_ned, dtype=float).reshape(-1, 3)
        self.pos_cov = np.asarray(pos_cov, dtype=float).reshape(-1, 6)
        self.vel_cov = np.asarray(vel_cov, dtype=float).reshape(-1, 6)
        self.rpy_sd = np.asarray(rpy_sd, dtype=float).reshape(-1, 3)
        self.dr_s = np.asarray(dr_s, dtype=float)
        self.fix_q = (np.zeros(len(self.t_us), dtype=int) if fix_q is None
                      else np.asarray(fix_q, dtype=int))
        self.fix_ns = (np.zeros(len(self.t_us), dtype=int) if fix_ns is None
                       else np.asarray(fix_ns, dtype=int))

    def __len__(self):
        return len(self.t_us)


def cov6(P, i0):
    """nn ne nd ee ed dd of the 3x3 block of P starting at index i0."""
    return (P[i0][i0], P[i0][i0 + 1], P[i0][i0 + 2],
            P[i0 + 1][i0 + 1], P[i0 + 1][i0 + 2], P[i0 + 2][i0 + 2])


def fix_quality(fix):
    """RTKLIB quality of a replay fix dict (replay.load_gnss), 0 unknown."""
    return int(fix.get("quality", 0)) if fix is not None else 0


def fix_num_sv(fix):
    """Satellites used in a replay fix dict (replay.load_gnss), 0 unknown."""
    return int(fix.get("num_sv", 0)) if fix is not None else 0


class SolutionRecorder:
    """replay_core observer: the filter's solution at rate_hz of sim time,
    the rows a solution file is written from (solution())."""

    def __init__(self, rate_hz):
        self.period_us = US_PER_SEC / rate_hz if rate_hz > 0 else 0.0
        self._last_us = None
        self.rows = []

    def on_epoch(self, r):
        if self._last_us is not None and (r.t - self._last_us) < self.period_us:
            return
        nav = r.nav
        llh = nav.position_llh()
        rpy = nav.rpy_ins()
        vel = nav.velocity_ned()
        P = nav.covariance()
        if llh is None or rpy is None or vel is None or P is None:
            return
        self._last_us = r.t
        dr = nav.deadreckoning_ms()
        self.rows.append((r.t, llh, rpy, vel, cov6(P, 0), cov6(P, 3),
                          [math.sqrt(max(P[i][i], 0.0)) for i in (6, 7, 8)],
                          dr / 1e3 if dr >= 0 else math.nan,
                          fix_quality(r.last_fix), fix_num_sv(r.last_fix)))

    def on_pos_error(self, r, err):
        pass

    def on_finish(self, r):
        pass

    def solution(self):
        if not self.rows:
            return Solution([], [], [], [], [], [], [], [])
        cols = list(zip(*self.rows))
        return Solution(*cols)


# ----------------------------------------------------------------------------
# GPS time
# ----------------------------------------------------------------------------

class GpsClock:
    """gps_s = a + b * t_us (continuous GPS seconds since 1980-01-06),
    fitted to (t_us, gps_s) pulse pairs: offset and rate from two or more,
    the offset alone (rate 1) from one. A pair a whole second off the
    others (a slipped pairing the flags did not catch) is dropped and the
    line fitted again. Raises ValueError without a pair."""

    def __init__(self, pairs, source=""):
        if not pairs:
            raise ValueError("no time sync pair")
        self.source = source
        t = np.array([p[0] for p in pairs], dtype=float)
        g = np.array([p[1] for p in pairs], dtype=float)
        keep = np.ones(len(t), dtype=bool)
        # Refit until the kept set stops changing (each round can only
        # change it while it differs from the last, a handful of rounds
        # covers any real file), the line always fitted to the set kept.
        for _ in range(5):
            self._fit(t[keep], g[keep])
            new_keep = np.abs(self.a + self.b * t - g) < 0.5
            if new_keep.sum() == 0 or np.array_equal(new_keep, keep):
                break
            keep = new_keep
        else:
            self._fit(t[keep], g[keep])
        self.n = int(keep.sum())
        self.n_rejected = len(t) - self.n
        self.resid_rms_s = float(np.sqrt(np.mean(
            (self.a + self.b * t[keep] - g[keep]) ** 2)))
        self.guessed = False
        self.n_utc = 0
        self.n_segments = 1

    def _fit(self, t, g):
        if len(t) < 2 or np.ptp(t) <= 0:
            self.b = 1e-6
            self.a = float(np.mean(g - self.b * t))
            return
        # About the means: t_us is ~1e9 while the span is ~1e3 s.
        tm, gm = t.mean(), g.mean()
        self.b = float(np.sum((t - tm) * (g - gm)) / np.sum((t - tm) ** 2))
        self.a = float(gm - self.b * tm)

    def gps_s(self, t_us):
        return self.a + self.b * np.asarray(t_us, dtype=float)

    def ppm(self):
        """Rate error of the recording clock against GPS time [ppm],
        positive = fast. 0 when only the offset is known."""
        return (1.0 / (self.b * 1e6) - 1.0) * 1e6

    def describe(self):
        if self.guessed:
            return ("GUESSED, no timesync.csv: the first epoch is put at the "
                    "system clock of the export, only the spacing is real")
        rate = (f", clock {self.ppm():+.1f} ppm" if self.n > 1
                else ", one pair: offset only, no clock rate")
        utc = (f" WARNING: {self.n_utc} pair(s) on the receiver's UTC grid,"
               f" off GPS time by the leap seconds" if self.n_utc else "")
        seg = (f" WARNING: {self.n_segments} clock segments (MCU restarts),"
               f" only the one with the most pairs used" if self.n_segments > 1
               else "")
        return (f"GPST from {self.source} ({self.n} pulse pairs"
                + (f", {self.n_rejected} rejected" if self.n_rejected else "")
                + f", residual {self.resid_rms_s * 1e6:.1f} us{rate}){utc}{seg}")


def guessed_clock(t0_us):
    """A GpsClock that puts t0_us at the current system time with the
    nominal rate, flagged as a guess (the system's UTC taken as GPST, the
    leap seconds are lost in the guess anyway)."""
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    clock = GpsClock([(int(t0_us), (now - GPS_EPOCH).total_seconds())])
    clock.guessed = True
    return clock


def load_gps_clock(data_dir):
    """GpsClock of a dataset directory's timesync.csv (TIMESYNC_HEADER's
    columns by name: timesync_t_us, gps_week, gps_tow_s, flags), None
    without one. TP_SUSPECT pairs and rows without a GPS time are skipped,
    repeated pairs (the recorder writes its last pair once a minute) count
    once, a row whose flags do not parse is skipped. The time is taken as
    GPS time: a pair on the receiver's UTC grid (UTC_BASE, the receiver
    default before tools/ublox_*_config.py set GPS) is off by the leap
    seconds, which is counted and named in the file header but not
    corrected. A recorder that restarted (the mcu_restarts column) starts
    its t_us anew, pairs of different restarts share no clock: only the
    segment with the most pairs is fitted, and the header says so."""
    path = os.path.join(data_dir, TIMESYNC_FILE)
    if not os.path.exists(path):
        return None
    segments = {}   # mcu_restarts -> {t_us: gps_s}
    n_utc = 0
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                t_us = int(row["timesync_t_us"])
                gps_s = int(row["gps_week"]) * GPS_SEC_PER_WEEK + float(row["gps_tow_s"])
            except (KeyError, TypeError, ValueError):
                continue
            try:
                flags = int(row.get("flags") or "0", 0)
                segment = int(row.get("mcu_restarts") or "0")
            except ValueError:
                continue
            if flags & TS_TP_SUSPECT:
                continue
            pairs = segments.setdefault(segment, {})
            if flags & TS_UTC_BASE and t_us not in pairs:
                n_utc += 1
            pairs[t_us] = gps_s
    if not segments:
        return None
    pairs = max(segments.values(), key=len)
    try:
        clock = GpsClock(sorted(pairs.items()), source=TIMESYNC_FILE)
    except ValueError:
        return None
    clock.n_utc = n_utc
    clock.n_segments = len(segments)
    return clock


def gpst_string(gps_s):
    """'yyyy/mm/dd hh:mm:ss.sss' of continuous GPS seconds (GPST, no leap
    seconds), RTKLIB's default time format."""
    ms = int(round(gps_s * 1000.0))
    t = GPS_EPOCH + datetime.timedelta(milliseconds=ms)
    return t.strftime("%Y/%m/%d %H:%M:%S.") + "%03d" % (ms % 1000)


# ----------------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------------

def _sd(cov6_rows, i):
    return np.sqrt(np.maximum(cov6_rows[:, i], 0.0))


def write_csv(path, sol, comment=""):
    """ref.csv layout plus 1-sigma columns (CSV_HEADER)."""
    sd_p = np.stack([_sd(sol.pos_cov, 0), _sd(sol.pos_cov, 3), _sd(sol.pos_cov, 5)], 1)
    sd_v = np.stack([_sd(sol.vel_cov, 0), _sd(sol.vel_cov, 3), _sd(sol.vel_cov, 5)], 1)
    rpy = np.degrees(sol.rpy)
    sd_a = np.degrees(sol.rpy_sd)
    # newline="\n" so the dataset CSVs stay LF on every platform
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(CSV_HEADER)
        if comment:
            f.write(f"# {comment}\n")
        for i in range(len(sol)):
            f.write("%d,%.9f,%.9f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                    "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" % (
                        sol.t_us[i], math.degrees(sol.llh[i, 0]),
                        math.degrees(sol.llh[i, 1]), sol.llh[i, 2],
                        rpy[i, 0], rpy[i, 1], rpy[i, 2],
                        sol.vel_ned[i, 0], sol.vel_ned[i, 1], sol.vel_ned[i, 2],
                        sd_p[i, 0], sd_p[i, 1], sd_p[i, 2],
                        sd_v[i, 0], sd_v[i, 1], sd_v[i, 2],
                        sd_a[i, 0], sd_a[i, 1], sd_a[i, 2]))


def _signed_root(c):
    return np.sign(c) * np.sqrt(np.abs(c))


def _enu(cov):
    """RTKLIB's ENU 1-sigma and signed covariance roots of a NED cov6:
    sd_n, sd_e, sd_u, sd_ne, sd_eu, sd_un."""
    nn, ne, nd, ee, ed, dd = (cov[:, i] for i in range(6))
    # u = -d: cov(e,u) = -cov(e,d), cov(u,n) = -cov(d,n)
    return (np.sqrt(np.maximum(nn, 0.0)), np.sqrt(np.maximum(ee, 0.0)),
            np.sqrt(np.maximum(dd, 0.0)), _signed_root(ne), _signed_root(-ed),
            _signed_root(-nd))


def _aided(sol):
    return np.isfinite(sol.dr_s) & (sol.dr_s <= AIDED_MAX_S)


def quality(sol):
    """The .pos Q per row (see the module docstring)."""
    return np.where(_aided(sol), np.where(sol.fix_q > 0, sol.fix_q, Q_AIDED),
                    Q_DEADRECKONING)


def num_sv(sol):
    """The .pos ns per row (see the module docstring)."""
    return np.where(_aided(sol), np.maximum(sol.fix_ns, 0), 0)


def write_pos(path, sol, clock=None, program="INSLIB", comment=""):
    """RTKLIB .pos (see the module docstring). clock: GpsClock, or None
    for a guessed one (guessed_clock(), the header says so)."""
    if clock is None:
        clock = guessed_clock(sol.t_us[0] if len(sol) else 0)
    times = [gpst_string(g) for g in clock.gps_s(sol.t_us)]
    pn, pe, pu, pne, peu, pun = _enu(sol.pos_cov)
    svn, sve, svu, svne, sveu, svun = _enu(sol.vel_cov)
    q = quality(sol)
    ns = num_sv(sol)
    age = np.where(np.isfinite(sol.dr_s), sol.dr_s, 0.0)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"% program   : {program}\n")
        if comment:
            f.write(f"% {comment}\n")
        if len(sol):
            f.write("%% obs start : %s GPST\n" % times[0])
            f.write("%% obs end   : %s GPST\n" % times[-1])
        f.write("%% time      : %s\n" % clock.describe())
        f.write("%% pos mode  : inertial (Q = quality of the last GNSS fix "
                "while aided, %d if unknown, %d dead reckoning, age = s since "
                "the last position aiding)\n" % (Q_AIDED, Q_DEADRECKONING))
        f.write("% (lat/lon/height=WGS84/ellipsoidal,Q=1:fix,2:float,3:sbas,"
                "4:dgps,5:single,6:ppp,ns=# of satellites)\n")
        f.write("%  GPST                   latitude(deg) longitude(deg)  height(m)"
                "   Q  ns   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m)"
                " age(s)  ratio    vn(m/s)    ve(m/s)    vu(m/s)      sdvn"
                "     sdve     sdvu    sdvne    sdveu    sdvun\n")
        for i in range(len(sol)):
            f.write("%s %14.9f %14.9f %10.4f %3d %3d %8.4f %8.4f %8.4f %8.4f "
                    "%8.4f %8.4f %6.2f %6.1f %10.5f %10.5f %10.5f %9.5f "
                    "%8.5f %8.5f %8.5f %8.5f %8.5f\n" % (
                        times[i], math.degrees(sol.llh[i, 0]),
                        math.degrees(sol.llh[i, 1]), sol.llh[i, 2], q[i], ns[i],
                        pn[i], pe[i], pu[i], pne[i], peu[i], pun[i], age[i], 0.0,
                        sol.vel_ned[i, 0], sol.vel_ned[i, 1], -sol.vel_ned[i, 2],
                        svn[i], sve[i], svu[i], svne[i], sveu[i], svun[i]))


def write(path, sol, clock=None, comment=""):
    """write_pos() for a .pos path, write_csv() for anything else."""
    if path.lower().endswith(".pos"):
        write_pos(path, sol, clock, comment=comment)
    else:
        write_csv(path, sol, comment=comment)


def companion_path(path, suffix):
    """path with suffix before the extension: out.pos -> out_rts.pos."""
    stem, ext = os.path.splitext(path)
    return f"{stem}{suffix}{ext}"
