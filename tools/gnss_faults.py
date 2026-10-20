#!/usr/bin/env python3
"""Synthetic GNSS faults and L1 single point errors for replay GNSS CSVs.

    python3 tools/gnss_faults.py DIR [--check]

Reads DIR/gnss_vel.csv (fixes with a GNSS velocity, the replay GNSS CSV
format) and DIR/ref.csv (the truth, for the time the platform first moves)
and writes one file per scenario next to them:

    gnss_fault_<type>.csv  the fixes with the faults of one FAULT_TYPES entry
    gnss_spp.csv           every fix at L1 single point accuracy
    gnss_spp_<type>.csv    gnss_spp.csv with the faults of one SPP_FAULT_TYPES
                           entry on top

--check writes nothing and fails if a file differs from what this script
would write (REQ-VER-050). The model and its seeds are fixed, so every run
writes the same files. The fault model:

    spike      3 single fixes, 3 s apart, offset U(5, 50) m, accuracy kept
    wrong_fix  U(5, 30) s with one offset U(0.5, 3) m, accuracy kept (a
               wrong RTK ambiguity fix)
    multipath  U(5, 30) s with one offset U(3, 15) m, reported 1 m
               horizontal, 2 m vertical
    degraded   30 s of white noise 1 m horizontal, 2 m vertical, reported
               exactly so
    vel_spike  3 single velocities, 3 s apart, U(1, 5) m/s off
    step       150 s with one offset U(0.5, 2) m, accuracy kept (a change of
               the RTK base or the datum, a receiver reset)
    big_step   as step, U(5, 10) m
    gross      3 times U(0, 30) s, the fix at or after each 500 m off (2 fixes
               when two times fall on the same one)

Horizontal offsets point in a uniformly random direction, the vertical one
is the horizontal magnitude times U(-1, 1). The events start 120 s after
the platform first moves (faster than 0.3 m/s) and follow every 90 s (300 s
for the steps) until 90 s before the end of the truth. The SPP error is a
first-order Gauss-Markov process per axis, 1.5 m horizontal and 3 m
vertical (1 sigma), correlation time 30 s, reported with that accuracy. A
fault on top of it keeps the SPP accuracy as reported.
"""

import argparse
import math
import os
import sys

import numpy as np

FAULT_TYPES = ["spike", "wrong_fix", "multipath", "degraded", "vel_spike", "step", "big_step",
               "gross"]
SPP_FAULT_TYPES = ["spike", "multipath", "vel_spike", "step", "gross"]

STILL_SPEED_MPS = 0.3
FAULT_SEED = 20261016
FAULT_START_AFTER_MOTION_S = 120.0
FAULT_PERIOD_S = 90.0
FAULT_TAIL_S = 90.0
FAULT_SPIKE_DT_S = (0.0, 3.0, 6.0)
FAULT_SPIKE_M = (5.0, 50.0)
FAULT_WRONG_FIX_M = (0.5, 3.0)
FAULT_MULTIPATH_M = (3.0, 15.0)
FAULT_DUR_S = (5.0, 30.0)
FAULT_DEGRADED_S = 30.0
FAULT_SD_M = (1.0, 2.0)
FAULT_VEL_SPIKE_MPS = (1.0, 5.0)
FAULT_STEP_M = (0.5, 2.0)
FAULT_BIG_STEP_M = (5.0, 10.0)
FAULT_STEP_S = 150.0
FAULT_STEP_PERIOD_S = 300.0
FAULT_GROSS_M = 500.0
FAULT_GROSS_WINDOW_S = 30.0

SPP_SD_M = (1.5, 3.0)
SPP_TAU_S = 30.0
SPP_SEED = 20261017

WGS84_A = 6378137.0
WGS84_E2 = (1.0 / 298.257223563) * (2.0 - 1.0 / 298.257223563)


def read_rows(path, lines=None):
    """The data rows of a replay CSV as float lists. lines (a list) gets
    the text of each row."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            rows.append([float(x) for x in line.split(",")])
            if lines is not None:
                lines.append(line.rstrip("\n"))
    return rows


def read_header(path):
    with open(path, encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh if line.startswith("#")]


def motion_start_s(truth):
    """First truth epoch faster than STILL_SPEED_MPS [s], truth rows as
    read_rows of ref.csv (t_us, ..., vn, ve at columns 7, 8)."""
    for r in truth:
        if math.hypot(r[7], r[8]) > STILL_SPEED_MPS:
            return r[0] * 1e-6
    return truth[0][0] * 1e-6


def fix_times_us(rows, truth):
    """The fix times the model uses: the fixes within the truth."""
    t0, t1 = truth[0][0], truth[-1][0]
    return np.array([int(r[0]) for r in rows if t0 <= r[0] <= t1], dtype=np.int64)


def fault_events(fix_us, truth, typ):
    """The faults of one type, the same on every call: [{type, t0_s, t1_s,
    mods, shift}], t0_s/t1_s the first and last faulty fix, mods {fix time
    [us]: (dn, de, dd, pos_sd, dvel)} with the NED position offset [m], the
    reported position 1-sigma (n, e, d) or None for unchanged, and the
    velocity error [m/s] or None. shift: for a step the NED offset that is
    the correct answer from t0_s to t1_s, else None."""
    fix_s = fix_us * 1e-6
    rng = np.random.default_rng([FAULT_SEED, FAULT_TYPES.index(typ)])
    sd_h, sd_v = FAULT_SD_M
    steps = ("step", "big_step")

    def offset(lo, hi):
        m = rng.uniform(lo, hi)
        az = rng.uniform(0.0, 2.0 * math.pi)
        return np.array([m * math.cos(az), m * math.sin(az), m * rng.uniform(-1.0, 1.0)])

    def span(t, dur):
        return fix_us[(fix_s >= t) & (fix_s < t + dur)]

    events = []
    t = motion_start_s(truth) + FAULT_START_AFTER_MOTION_S
    t_end = truth[-1][0] * 1e-6 - FAULT_TAIL_S
    while t < t_end:
        mods = {}
        d = None
        if typ in ("spike", "vel_spike"):
            for dt in FAULT_SPIKE_DT_S:
                j = int(np.searchsorted(fix_s, t + dt))
                if j >= len(fix_us):
                    continue
                if typ == "spike":
                    mods[int(fix_us[j])] = tuple(offset(*FAULT_SPIKE_M)) + (None, None)
                else:
                    mods[int(fix_us[j])] = (0.0, 0.0, 0.0, None, offset(*FAULT_VEL_SPIKE_MPS))
        elif typ == "gross":
            for dt in sorted(rng.uniform(0.0, FAULT_GROSS_WINDOW_S, 3)):
                j = int(np.searchsorted(fix_s, t + dt))
                if j < len(fix_us):
                    mods[int(fix_us[j])] = tuple(offset(FAULT_GROSS_M, FAULT_GROSS_M)) + (None, None)
        elif typ in steps:
            d = offset(*(FAULT_STEP_M if typ == "step" else FAULT_BIG_STEP_M))
            for u in span(t, FAULT_STEP_S):
                mods[int(u)] = (d[0], d[1], d[2], None, None)
        elif typ in ("wrong_fix", "multipath"):
            dur = rng.uniform(*FAULT_DUR_S)
            d = offset(*(FAULT_WRONG_FIX_M if typ == "wrong_fix" else FAULT_MULTIPATH_M))
            sd = None if typ == "wrong_fix" else (sd_h, sd_h, sd_v)
            for u in span(t, dur):
                mods[int(u)] = (d[0], d[1], d[2], sd, None)
        else:   # degraded
            for u in span(t, FAULT_DEGRADED_S):
                n = rng.normal(0.0, 1.0, 3) * np.array([sd_h, sd_h, sd_v])
                mods[int(u)] = (n[0], n[1], n[2], (sd_h, sd_h, sd_v), None)
        if mods:
            us = sorted(mods)
            events.append(dict(type=typ, t0_s=us[0] * 1e-6, t1_s=us[-1] * 1e-6, mods=mods,
                               shift=tuple(float(x) for x in d) if typ in steps else None))
        t += FAULT_STEP_PERIOD_S if typ in steps else FAULT_PERIOD_S
    return events


def spp_mods(fix_us):
    """The SPP error of every fix, in the mods format of fault_events."""
    rng = np.random.default_rng(SPP_SEED)
    sd = np.array([SPP_SD_M[0], SPP_SD_M[0], SPP_SD_M[1]])
    x = rng.normal(0.0, 1.0, 3) * sd
    mods, t_prev = {}, None
    for u in fix_us:
        if t_prev is not None:
            a = math.exp(-(u - t_prev) * 1e-6 / SPP_TAU_S)
            x = a * x + math.sqrt(1.0 - a * a) * rng.normal(0.0, 1.0, 3) * sd
        t_prev = u
        mods[int(u)] = (x[0], x[1], x[2], tuple(sd), None)
    return mods


def scenario_mods(fix_us, truth, spp, typ):
    """Every modified fix of a scenario: the SPP error (spp) and the faults
    of typ (or None) on top, a fault keeping the SPP accuracy as reported.
    None for none."""
    mods = spp_mods(fix_us) if spp else {}
    if typ:
        for ev in fault_events(fix_us, truth, typ):
            for u, (dn, de, dd, sd, dv) in ev["mods"].items():
                b = mods.get(u)
                if b:
                    mods[u] = (b[0] + dn, b[1] + de, b[2] + dd, b[3], dv)
                else:
                    mods[u] = (dn, de, dd, sd, dv)
    return mods or None


def offset_llh(lat, lon, h, dn, de, dd):
    """lat/lon [deg], h [m] moved by a NED offset [m]."""
    sl = math.sin(math.radians(lat))
    rn = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sl * sl)
    rm = rn * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * sl * sl)
    return (lat + math.degrees(dn / (rm + h)),
            lon + math.degrees(de / ((rn + h) * math.cos(math.radians(lat)))), h - dd)


def apply_mod(r, m):
    """Row r (float list) with one mod of fault_events applied, a new list."""
    r = list(r)
    dn, de, dd, sd, dv = m
    r[1:4] = offset_llh(r[1], r[2], r[3], dn, de, dd)
    if sd:
        r[4:10] = [sd[0] ** 2, 0, 0, sd[1] ** 2, 0, sd[2] ** 2]
    if dv is not None:
        r[10:13] = [r[10] + dv[0], r[11] + dv[1], r[12] + dv[2]]
    return r


def format_row(r):
    return ",".join(("%d" % x) if j == 0 else ("%.15g" % x) for j, x in enumerate(r))


def scenarios():
    """(file name, spp, fault type or None) of every scenario."""
    out = [("gnss_fault_%s.csv" % t, False, t) for t in FAULT_TYPES]
    out.append(("gnss_spp.csv", True, None))
    out += [("gnss_spp_%s.csv" % t, True, t) for t in SPP_FAULT_TYPES]
    return out


def render(header, rows, lines, truth, spp, typ):
    """The content of one scenario file from gnss_vel.csv (its header, rows
    and their text) and the truth. A row without a mod is kept as it is."""
    mods = scenario_mods(fix_times_us(rows, truth), truth, spp, typ) or {}
    what = ("the L1 single point error" if spp else "") + (" plus " if spp and typ else "") + \
        ("the %s faults" % typ if typ else "")
    head = header + ["# SYNTHETIC: gnss_vel.csv with %s of tools/gnss_faults.py "
                     "(fixed seeds, REQ-VER-050)" % what]
    body = [format_row(apply_mod(r, mods[int(r[0])])) if int(r[0]) in mods else line
            for r, line in zip(rows, lines)]
    return "\n".join(head + body) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir", help="directory with gnss_vel.csv and ref.csv")
    ap.add_argument("--check", action="store_true",
                    help="write nothing, fail if a file differs")
    a = ap.parse_args()
    sys.exit(1 if write_all(a.dir, a.check) else 0)


def write_all(directory, check=False):
    """Writes every scenario file into directory, with check only compares
    them. Returns the number of files that differ (check) or 0."""
    src = os.path.join(directory, "gnss_vel.csv")
    lines = []
    rows = read_rows(src, lines)
    header = read_header(src)
    truth = read_rows(os.path.join(directory, "ref.csv"))
    bad = 0
    for name, spp, typ in scenarios():
        path = os.path.join(directory, name)
        text = render(header, rows, lines, truth, spp, typ)
        if check:
            old = open(path, encoding="utf-8").read() if os.path.exists(path) else None
            if old != text:
                print("differs: " + path)
                bad += 1
        else:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            print("wrote " + path)
    return bad


if __name__ == "__main__":
    main()
