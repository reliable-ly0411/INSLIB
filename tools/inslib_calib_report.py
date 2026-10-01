#!/usr/bin/env python3
"""Calibration certificate and calibration record (library, not a command).

Turns one solved session of tools/inslib_calib_gui.py into a LaTeX
document in two parts:

  * **Page 1, the certificate.** One self-contained sheet for the
    customer, meant to be printed and shipped with the unit: which unit,
    when, where, by whom, against which reference magnitudes, the
    resulting coefficients in summary, what the calibration achieved, the
    verdict and the signatures. It is numbered "Page 1 of 1" on its own.
  * **The calibration record** for the manufacturer, numbered separately:
    every coefficient in full precision, the uncertainty, the acceptance
    criteria, diagnostics plots, orientation coverage, noise, the solver's
    own notes, and long per-pose tables. The complete raw recording is
    written next to the .tex as replay-format CSV and attached to the PDF
    (embedfile), with its SHA-256 and the command that recomputes the
    calibration from it, so the result can be reproduced and checked
    long after the unit has left.

Plain pdflatex (pgfplots for the curves, no figure files), so

    pdflatex certificate.tex && pdflatex certificate.tex

or `make` with the Makefile written beside it gives the PDF. Two passes
because the record footer says "page x of y".

Nothing here computes a calibration: the numbers come from
inslib_imu_calib.py and friends, this module only formats them plus a
little statistics on top.

**Uncertainty.** The accelerometer parameters get a Type A standard
uncertainty from the fit itself: the residual of every static pose
against |g|, the Jacobian of that residual with respect to the parameters
at the solution, and cov = s^2 (J'J)^-1 with s^2 the residual variance per
degree of freedom. It is the statistical part only. A wrong |g|, a pose
that was not really at rest or a temperature drift over the session are
systematic and do not show up in it, which is why the reference
magnitudes and the temperature span are printed right next to it. The gyro
bias gets the white noise limit of averaging over the rest period,
sqrt(psd / T), which is a lower bound for the same reason.

(c) Jan Zwiener (jan@zwiener.org)
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inslib_imu_tk as imu_tk            # noqa: E402
import inslib_imu_calib as calib           # noqa: E402
import inslib_mag_calib as mag_calib       # noqa: E402
import inslib_frame_align as fa            # noqa: E402

DEG = 180.0 / math.pi
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Points per curve after decimation. Each bucket contributes its minimum
# and its maximum, so a spike survives the decimation, and pdflatex's
# memory stays far away from its limit with every plot in the document.
PLOT_BUCKETS = 450

# A body axis counts as visited when a pose had gravity within this angle
# of it (either sign). Six of six is a session that turned the unit over
# every face at least once.
AXIS_COVER_DEG = 30.0

# Acceptance limits. Each one is the threshold the calibration code
# itself already warns at, so the document and the tool cannot disagree.
LIM_ACC_RMS = 0.05                  # quality_warnings(): |g| rms after fit
LIM_GYR_DIR_RAD = 0.05              # quality_warnings(): gyro direction rms
LIM_TEMP_SPAN_C = 3.0               # upload dialog: one node, one temperature
LIM_POSES_GOOD = 20                 # quality_warnings(): fewer is a warning
LIM_MAG_SPREAD = 0.25               # mag_calib.solve(): flat cloud warning
LIM_MAG_FIELD_REL = 0.25            # mag_calib.solve(): field off reference

# Remarks longer than this stay in the record only: the certificate has
# exactly one page and a paragraph of free text could push it onto two.
CERT_REMARKS_MAX = 240

OK, WARN, FAIL = "ok", "warn", "fail"

MAKEFILE_MARK = "# inslib_calib_report"
MAKEFILE = MAKEFILE_MARK + """ -- builds every calibration certificate here.
# Two pdflatex passes, the page count in the footer needs the second one.
# The *_imu.csv / *_mag.csv next to a .tex are attached to its PDF, so
# they have to be here when it is built.
TEX := $(wildcard *.tex)
PDF := $(TEX:.tex=.pdf)
LATEX ?= pdflatex
LATEXFLAGS ?= -interaction=nonstopmode -halt-on-error

all: $(PDF)

%.pdf: %.tex
\t$(LATEX) $(LATEXFLAGS) $<
\t$(LATEX) $(LATEXFLAGS) $<

clean:
\trm -f *.aux *.log *.out

distclean: clean
\trm -f $(PDF)

.PHONY: all clean distclean
"""


@dataclass
class ReportInput:
    """Everything the document prints, gathered by the caller.

    Only `rec` and `cal` are required. The rest is context the window
    knows and the solve does not (who, where, which references), and
    every field that is left empty is printed as not given rather than
    left out, so a reader can tell "not recorded" from "forgotten".

    `init_sec` and `field_ref_ut` have to be the values the solve ran
    with: the record prints the command that repeats the solve, and it
    only reproduces the result with the same inputs."""
    rec: object                         # calib.Recording
    cal: object                         # calib.Calibration (sensor frame)
    magcal: object = None               # mag_calib.MagCalibration
    housing: object = None              # fa.HousingAlignment
    init_sec: float = calib.DEFAULT_INIT_SEC
    pose_sec: float = calib.DEFAULT_POSE_SEC
    operator: str = ""
    place: str = ""
    device: str = ""
    serial: str = ""
    remarks: str = ""
    position: tuple = None              # (lat deg, lon deg, h m or None)
    gravity_source: str = ""
    field_ref_ut: float = 0.0           # 0: no reference, scaled to measured
    field_source: str = ""
    wmm_decl_deg: float = None
    wmm_incl_deg: float = None
    t_start: datetime.datetime = None
    t_end: datetime.datetime = None
    source: str = ""                    # port / UDP / demo, in words
    board_info: dict = None             # ubx.parse_cfginfo() of the session
    rec_cal_frac: float = 0.0           # share of samples the board corrected
    tool: str = "tools/inslib_calib_gui.py"
    created: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now().astimezone())


@dataclass
class RawFile:
    """One raw data file written beside the .tex and attached to the PDF."""
    name: str                           # file name, relative to the .tex
    sha256: str
    rows: int
    what: str


# ===========================================================================
# Numbers
# ===========================================================================

def pose_means(rec, cal):
    """(acc means, gyr means, t_mid) per static pose, raw, sensor frame."""
    t, acc, gyr = rec.arrays()
    a, g, tm = [], [], []
    for (s, e) in cal.intervals:
        a.append(acc[s:e + 1].mean(axis=0))
        g.append(gyr[s:e + 1].mean(axis=0))
        tm.append(0.5 * (t[s] + t[e]))
    return np.asarray(a).reshape(-1, 3), np.asarray(g).reshape(-1, 3), \
        np.asarray(tm)


def acc_params(matrix, bias):
    """The 9-vector of inslib_imu_tk.acc_matrix back out of M and b.

    M = T*K with T unit upper triangular, so the diagonal of M is the
    scale and the off-diagonal terms divided by it are the misalignment."""
    m = np.asarray(matrix, dtype=float)
    sx, sy, sz = m[0, 0], m[1, 1], m[2, 2]
    return np.array([-m[0, 1] / sy, m[0, 2] / sz, -m[1, 2] / sz,
                     sx, sy, sz, *np.asarray(bias, dtype=float)])


def acc_uncertainty(rec, cal):
    """Standard uncertainty (k=1) of the accelerometer parameters.

    Returns {"bias": [3] m/s^2, "scale": [3] (-), "mis": [3] rad or None,
    "s": residual std [m/s^2], "dof": int} or None when the session has no
    degrees of freedom left. One observation per static pose (its mean),
    because the samples inside a pose are not independent measurements of
    anything but the noise."""
    means, _g, _t = pose_means(rec, cal)
    p0 = acc_params(cal.acc_matrix, cal.acc_bias)
    free = list(range(9)) if cal.misalignment_estimated else list(range(3, 9))
    n = len(means)
    dof = n - len(free)
    if dof <= 0:
        return None

    def res(p):
        return cal.gravity - np.linalg.norm(
            imu_tk.apply_calib(means, imu_tk.acc_matrix(p), p[6:9]), axis=1)

    r0 = res(p0)
    jac = np.empty((n, len(free)))
    for k, i in enumerate(free):
        h = 1e-6 * max(1.0, abs(p0[i]))
        pp, pm = p0.copy(), p0.copy()
        pp[i] += h
        pm[i] -= h
        jac[:, k] = (res(pp) - res(pm)) / (2.0 * h)
    s2 = float(r0 @ r0) / dof
    try:
        cov = s2 * np.linalg.inv(jac.T @ jac)
    except np.linalg.LinAlgError:
        return None
    sig = np.full(9, float("nan"))
    sig[free] = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return {"bias": list(sig[6:9]), "scale": list(sig[3:6]),
            "mis": list(sig[0:3]) if cal.misalignment_estimated else None,
            "s": math.sqrt(s2), "dof": dof}


def gyr_bias_uncertainty(cal, init_sec):
    """White noise limit of the gyro bias [rad/s]: sqrt(psd / T).

    psd is sigma^2*dt per sample (psd_from_still), so averaging over T
    seconds leaves sigma/sqrt(N) = sqrt(psd/T)."""
    if not (cal.gyr_psd > 0.0 and init_sec > 0.0):
        return float("nan")
    return math.sqrt(cal.gyr_psd / init_sec)


def scale_error_percent(matrix):
    """Per-axis scale correction [%], the column norms of M minus one
    (the same figure Calibration.summary() prints)."""
    m = np.asarray(matrix, dtype=float)
    return [(float(np.linalg.norm(m[:, j])) - 1.0) * 100.0 for j in range(3)]


def _fibonacci_sphere(n=2000):
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    th = math.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([np.cos(th) * np.sin(phi),
                            np.sin(th) * np.sin(phi), np.cos(phi)])


def coverage(dirs):
    """How well a set of unit directions covers the sphere.

    spread: 3 * smallest eigenvalue of the direction scatter, 1 for an
    even cover, 0 for a plane (the measure mag_calib uses for the field
    cloud). gap_deg: radius of the largest cap no direction falls into.
    axes: which of +x -x +y -y +z -z had a direction within
    AXIS_COVER_DEG."""
    d = np.asarray(dirs, dtype=float).reshape(-1, 3)
    if len(d) == 0:
        return {"n": 0, "spread": 0.0, "gap_deg": 180.0, "axes": [False] * 6}
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    spread = float(3.0 * np.linalg.eigvalsh(d.T @ d / len(d))[0])
    grid = _fibonacci_sphere()
    near = np.clip(grid @ d.T, -1.0, 1.0).max(axis=1)
    gap = float(np.degrees(np.arccos(near.min())))
    cos_lim = math.cos(math.radians(AXIS_COVER_DEG))
    axes = []
    for k in range(3):
        for sgn in (1.0, -1.0):
            axes.append(bool((sgn * d[:, k] >= cos_lim).any()))
    return {"n": len(d), "spread": spread, "gap_deg": gap, "axes": axes}


def coefficient_digest(cal, magcal=None, housing=None):
    """Short SHA-256 over the coefficients.

    Ties a paper copy to one set of numbers: the same session written
    twice gives the same digest, any change to a coefficient gives a
    different one."""
    doc = {"acc_matrix": cal.acc_matrix, "acc_bias": cal.acc_bias,
           "gyr_matrix": cal.gyr_matrix, "gyr_bias": cal.gyr_bias}
    if magcal is not None:
        doc["mag_matrix"] = magcal.matrix
        doc["mag_bias"] = magcal.bias
    if housing is not None:
        doc["housing"] = housing.matrix
    txt = json.dumps(doc, sort_keys=True, separators=(",", ":"),
                     default=float)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16]


def _git(*args):
    """stdout of one git call in this repository, "" when there is none
    (no git on the PATH, or the tools run from an export without .git)."""
    try:
        return subprocess.run(["git", "-C", REPO] + list(args),
                              capture_output=True, text=True,
                              timeout=3).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def software_info():
    """{"version", "repo", "commit", "dirty"} of the tools that wrote the
    document, each "" (or False) where it cannot be found out.

    The commit is the full hash, and "dirty" says whether tracked files
    differed from it: a calibration computed by modified code is only
    reproducible with that modification, so the record has to say so."""
    ver = ""
    try:
        with open(os.path.join(REPO, "CITATION.cff"), encoding="utf-8") as f:
            for line in f:
                if line.startswith("version:"):
                    ver = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    url = _git("config", "--get", "remote.origin.url")
    if url:
        # git@host:owner/name.git and https://host/owner/name(.git) alike
        tail = url.rstrip("/").replace(":", "/").split("/")
        repo = "/".join(tail[-2:])
        repo = repo[:-4] if repo.endswith(".git") else repo
    else:
        repo = os.path.basename(REPO)
    commit = _git("rev-parse", "HEAD")
    dirty = bool(commit) and bool(_git("status", "--porcelain",
                                       "--untracked-files=no"))
    return {"version": ver, "repo": repo, "commit": commit, "dirty": dirty}


def mag_fit_radius(magcal):
    """What the raw ellipsoid measured as field strength [uT].

    Not stored on MagCalibration once a reference was given (the result
    is scaled to the reference), so recovered from the matrix: the fit is
    field * unit-sphere map, and the unit-sphere map has determinant
    radius^-3."""
    det = abs(float(np.linalg.det(np.asarray(magcal.matrix, dtype=float))))
    if not (det > 0.0):
        return float("nan")
    return magcal.field_ut / det ** (1.0 / 3.0)


def checks(inp, cov_pose):
    """[(criterion, limit text, measured text, OK/WARN/FAIL)].

    FAIL is kept for what makes the coefficients wrong rather than merely
    less good: too few poses to solve at all, a fit that does not
    reproduce |g|, or a recording the board had already corrected (the
    result is then a residual of the stored calibration)."""
    cal = inp.cal
    out = []
    n = cal.n_positions
    out.append(("Static poses",
                r"$\geq %d$ (recommended $\geq %d$)"
                % (calib.MIN_POSITIONS, LIM_POSES_GOOD), "%d" % n,
                FAIL if n < calib.MIN_POSITIONS else
                (WARN if n < LIM_POSES_GOOD else OK)))
    rms = cal.acc_stats.get("full", {}).get("rms", cal.residual_rms)
    out.append((r"Accelerometer: $|a|-|g|$ RMS after calibration",
                r"$\leq$ %s" % _si(LIM_ACC_RMS, r"\metre\per\second\squared"),
                _si(rms, r"\metre\per\second\squared", 4),
                FAIL if not (rms <= LIM_ACC_RMS) else OK))
    gdir = cal.gyr_stats.get("full", {}).get("rms", cal.gyr_residual_rms)
    out.append(("Gyroscope: direction error RMS",
                r"$\leq$ %s" % _si(math.degrees(LIM_GYR_DIR_RAD), r"\degree", 2),
                _si(math.degrees(gdir), r"\degree", 2) if gdir == gdir else "--",
                OK if gdir == gdir and gdir <= LIM_GYR_DIR_RAD else WARN))
    span = cal.temp_hi - cal.temp_lo
    out.append(("IMU temperature span",
                r"$\leq$ %s" % _si(LIM_TEMP_SPAN_C, r"\kelvin", 1),
                _si(span, r"\kelvin", 1) if math.isfinite(span) else
                "not recorded",
                OK if math.isfinite(span) and span <= LIM_TEMP_SPAN_C else WARN))
    frac = inp.rec_cal_frac if inp.rec_cal_frac == inp.rec_cal_frac else 0.0
    out.append(("Raw data (no on-board correction applied)",
                r"\SI{0}{\percent} corrected",
                _si(100.0 * frac, r"\percent", 1),
                FAIL if frac > 0.0 else OK))
    nsat = getattr(inp.rec, "n_saturated", 0)
    out.append(("Samples at sensor full scale", "0", "%d" % nsat,
                WARN if nsat else OK))
    naxes = sum(cov_pose["axes"])
    out.append((r"Coverage: $\pm x, \pm y, \pm z$ visited",
                r"6 of 6 (within \SI{%.0f}{\degree})" % AXIS_COVER_DEG,
                "%d of 6" % naxes, OK if naxes == 6 else WARN))
    mc = inp.magcal
    if mc is not None:
        out.append(("Magnetometer: direction coverage",
                    r"$\geq$ \num{%.2f}" % LIM_MAG_SPREAD,
                    r"\num{%.2f}" % mc.spread,
                    OK if mc.spread >= LIM_MAG_SPREAD else WARN))
        if inp.field_ref_ut > 0.0:
            rel = abs(mag_fit_radius(mc) - inp.field_ref_ut) / inp.field_ref_ut
            out.append(("Magnetometer: measured vs. reference field",
                        r"$\leq$ %s" % _si(100 * LIM_MAG_FIELD_REL, r"\percent", 0),
                        _si(100.0 * rel, r"\percent", 1),
                        OK if rel <= LIM_MAG_FIELD_REL else WARN))
        out.append(("Magnetometer: alignment to IMU solved", "yes",
                    "yes" if mc.align is not None else "no",
                    OK if mc.align is not None else WARN))
    nw = len(solver_notes(inp))
    out.append(("Solver warnings", "none", "%d" % nw, WARN if nw else OK))
    return out


def solver_notes(inp):
    """Every warning the solves raised, verbatim."""
    notes = list(inp.cal.warnings)
    mc, hs = inp.magcal, inp.housing
    if mc is not None:
        notes += list(mc.warnings)
        if mc.align is not None:
            notes += list(mc.align.warnings)
    if hs is not None:
        notes += list(hs.warnings)
    return notes


def verdict(rows):
    states = [r[3] for r in rows]
    if FAIL in states:
        return FAIL
    if WARN in states:
        return WARN
    return OK


# ===========================================================================
# Raw data
# ===========================================================================

def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_raw(rec, base):
    """The whole recording as replay-format CSV, [RawFile].

    The format inslib_imu_calib.py --csv reads, so the record's reproduce
    command runs on exactly these files. %.17g round-trips a double, so
    the recomputed calibration is the one in the document and not a
    near miss of it."""
    out = []
    imu_path = base + "_imu.csv"
    t_us = rec.t_us
    temps = getattr(rec, "temp_c", [])
    with open(imu_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# t_us, gyr_frd_xyz [rad/s], acc_frd_xyz [m/s^2],"
                " imu_temp_c [degC]\n")
        for i in range(len(t_us)):
            g, a = rec.gyr[i], rec.acc[i]
            tc = temps[i] if i < len(temps) else float("nan")
            f.write("%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.4g\n"
                    % (t_us[i], g[0], g[1], g[2], a[0], a[1], a[2], tc))
    out.append(RawFile(os.path.basename(imu_path), _sha256_file(imu_path),
                       len(t_us), "IMU samples"))
    if rec.has_mag():
        mag_path = base + "_mag.csv"
        with open(mag_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("# t_us, mag_frd_xyz [uT], mag_temp_c [degC]\n")
            for i in range(len(rec.mag)):
                m = rec.mag[i]
                tc = rec.mag_temp_c[i] if i < len(rec.mag_temp_c) \
                    else float("nan")
                f.write("%d,%.17g,%.17g,%.17g,%.4g\n"
                        % (rec.mag_t_us[i], m[0], m[1], m[2], tc))
        out.append(RawFile(os.path.basename(mag_path),
                           _sha256_file(mag_path), len(rec.mag),
                           "magnetometer samples"))
    return out


def reproduce_command(inp, raw):
    """The console call that recomputes this calibration from `raw`."""
    if not raw:
        return ""
    cmd = ["python3 tools/inslib_imu_calib.py", "--csv", raw[0].name]
    if len(raw) > 1:
        cmd += ["--mag-csv", raw[1].name]
    else:
        cmd.append("--no-mag")
    cmd += ["--gravity", "%.6f" % inp.cal.gravity,
            "--init-sec", "%g" % inp.init_sec]
    if not inp.cal.misalignment_estimated:
        cmd.append("--no-misalignment")
    if inp.magcal is not None and inp.field_ref_ut > 0.0:
        cmd += ["--mag-field-ut", "%.6g" % inp.field_ref_ut]
    cmd += ["-o", "recomputed.yaml"]
    return " ".join(cmd)


# ===========================================================================
# LaTeX helpers
# ===========================================================================

_TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
            "<": r"\textless{}", ">": r"\textgreater{}"}


def tex(s):
    """Text for LaTeX, from anything a user may type into a field."""
    return "".join(_TEX_ESC.get(c, c) for c in str(s))


def _given(s):
    s = (s or "").strip()
    return tex(s) if s else r"\textit{not given}"


def _num(v, prec=4, sign=False):
    if v is None or not math.isfinite(v):
        return "--"
    return r"\num{%s}" % (("%+." if sign else "%.") + "%df" % prec) % v


def _si(v, unit, prec=3, sign=False):
    if v is None or not math.isfinite(v):
        return "--"
    return r"\SI{%s}{%s}" % ((("%+." if sign else "%.") + "%df" % prec) % v,
                             unit)


def _matrix(m, prec=6):
    rows = [" & ".join(r"\num{%+.*f}" % (prec, float(x)) for x in row)
            for row in m]
    return r"\begin{pmatrix}" + r" \\ ".join(rows) + r"\end{pmatrix}"


def _state(s):
    return {OK: r"\textcolor{okc}{\textbf{pass}}",
            WARN: r"\textcolor{warnc}{\textbf{note}}",
            FAIL: r"\textcolor{failc}{\textbf{fail}}"}[s]


VERDICT_TEXT = {OK: r"\textcolor{okc}{PASSED}",
                WARN: r"\textcolor{warnc}{PASSED WITH REMARKS}",
                FAIL: r"\textcolor{failc}{FAILED}"}


def _envelope(t, y, n=PLOT_BUCKETS):
    """Min/max per bucket, as one coordinate list in time order."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) == 0:
        return []
    if len(t) <= 2 * n:
        return list(zip(t, y))
    edges = np.linspace(0, len(t), n + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = y[a:b]
        i, j = int(np.argmin(seg)), int(np.argmax(seg))
        for k in sorted((i, j)):
            out.append((t[a + k], seg[k]))
    return out


def _coords(pts, fmt="(%.3f,%.6g)"):
    return " ".join(fmt % (float(a), float(b)) for a, b in pts)


def _addplot(opts, pts, legend=None):
    if not pts:
        return ""
    s = r"\addplot[%s] coordinates {%s};" % (opts, _coords(pts))
    if legend:
        s += r"\addlegendentry{%s}" % legend
    return s + "\n"


AXIS_TAIL = r"\end{axis}\end{tikzpicture}\end{minipage}\end{center}"


def _axis_head(caption, opts):
    """Caption line, then the axis with its legend in one row above it.

    Inside the axis a legend sits on top of the curves: these plots run
    across the whole width, so there is no empty corner to put it in. The
    minipage keeps the caption on the page of its plot."""
    return (r"\begin{center}\begin{minipage}{\linewidth}\centering"
            r"{\small\bfseries %s}\\[1pt]"
            r"\begin{tikzpicture}\begin{axis}[%s, legend style={at={(0.5,1.02)},"
            r" anchor=south, legend columns=-1, /tikz/every even column/"
            r".append style={column sep=8pt}}]" % (caption, opts))


def _dirs_azel(d):
    """(azimuth, elevation) [deg] of unit vectors, FRD body frame."""
    d = np.asarray(d, dtype=float).reshape(-1, 3)
    if len(d) == 0:
        return []
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    az = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    el = np.degrees(np.arcsin(np.clip(d[:, 2], -1.0, 1.0)))
    return list(zip(az, el))


def _kv_table(rows, first="4.4cm"):
    """Two-column key/value table, its own paragraph."""
    out = [r"\par\noindent\begin{tabularx}{\linewidth}{@{}L{%s}X@{}}"
           r"\toprule " % first]
    out += [r"%s & %s \\" % kv for kv in rows]
    out.append(r"\bottomrule\end{tabularx}\par")
    return "\n".join(out)


PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[english]{babel}
\usepackage[a4paper,left=1.8cm,right=1.8cm,top=2.0cm,bottom=1.8cm,
  headheight=14pt]{geometry}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage{xcolor}
\usepackage{siunitx}
\usepackage{pgfplots}
\usepackage{fancyhdr}
\usepackage{needspace}
\usepackage{embedfile}
\usepackage[hidelinks]{hyperref}
\pgfplotsset{compat=1.16}
\sisetup{group-digits=false}
\pgfkeys{/pgf/number format/.cd, 1000 sep={}}
\definecolor{okc}{HTML}{1E8449}
\definecolor{warnc}{HTML}{B9770E}
\definecolor{failc}{HTML}{B03A2E}
\definecolor{rawc}{HTML}{D68910}
\definecolor{calc}{HTML}{1F618D}
\definecolor{refc}{HTML}{7F8C8D}
\newcommand{\mg}{\ensuremath{\mathrm{m}g}}
\newcommand{\sqrth}{\ensuremath{/\!\sqrt{\mathrm{h}}}}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\pgfplotsset{every axis/.append style={width=\linewidth, height=5.2cm,
  grid=major, grid style={gray!25}, tick label style={font=\footnotesize},
  label style={font=\footnotesize}, legend style={font=\footnotesize,
  fill=white, fill opacity=0.85, text opacity=1, draw=gray!40}}}
"""


# ===========================================================================
# The document
# ===========================================================================

class _Doc:
    """The quantities both parts print, computed once."""

    def __init__(self, inp, raw):
        self.inp, self.raw = inp, raw or []
        rec, cal, mc = inp.rec, inp.cal, inp.magcal
        self.t0 = inp.t_start or inp.created
        self.cert_id = "KP-%s" % self.t0.strftime("%Y%m%d-%H%M%S")
        tag = "".join(c for c in (inp.serial or "").strip()
                      if c.isalnum() or c in "-_")[:20]
        if tag:
            self.cert_id += "-" + tag
        self.digest = coefficient_digest(cal, mc, inp.housing)
        self.t, self.acc, self.gyr = rec.arrays()
        self.acc_cal = imu_tk.apply_calib(self.acc, np.asarray(cal.acc_matrix),
                                          np.asarray(cal.acc_bias))
        self.gyr_cal = imu_tk.apply_calib(self.gyr, np.asarray(cal.gyr_matrix),
                                          np.asarray(cal.gyr_bias))
        self.pa, self.pg, self.ptm = pose_means(rec, cal)
        self.cov_pose = coverage(self.pa)
        self.tm, self.mag = rec.mag_arrays()
        self.mag_cal = None
        self.cov_mag = None
        if mc is not None and len(self.mag):
            self.mag_cal = mag_calib.apply_calib(self.mag, mc.matrix, mc.bias)
            self.cov_mag = coverage(
                self.mag_cal[::max(1, len(self.mag_cal) // 1500)])
        self.unc = acc_uncertainty(rec, cal)
        self.sig_bg = gyr_bias_uncertainty(cal, inp.init_sec)
        self.rows = checks(inp, self.cov_pose)
        self.verdict = verdict(self.rows)
        off = self.t0.strftime("%z")
        self.tz = ("UTC%s:%s" % (off[:3], off[3:])) if off else ""

    def time_span(self):
        s = self.t0.strftime("%H:%M:%S")
        if self.inp.t_end is not None:
            s += " -- " + self.inp.t_end.strftime("%H:%M:%S")
        return (s + " " + self.tz).strip()

    def position(self):
        if self.inp.position is None:
            return r"\textit{not given}"
        lat, lon, h = self.inp.position
        s = r"\num{%.6f}\,\si{\degree}, \num{%.6f}\,\si{\degree}" % (lat, lon)
        if h is not None:
            s += r", $h_\mathrm{ell}$ = \SI{%.0f}{\metre}" % h
        return s

    def field_ref(self):
        inp = self.inp
        if inp.field_ref_ut > 0.0:
            s = _si(inp.field_ref_ut, r"\micro\tesla", 1)
            if inp.wmm_decl_deg is not None and inp.wmm_incl_deg is not None:
                s += r" (decl.\ %s, incl.\ %s)" % (
                    _si(inp.wmm_decl_deg, r"\degree", 1, True),
                    _si(inp.wmm_incl_deg, r"\degree", 1, True))
            return s + r", %s" % _given(inp.field_source)
        return r"none, scaled to the measured field"

    def temp_range(self):
        cal = self.inp.cal
        if not math.isfinite(cal.temp_lo):
            return r"\textit{not recorded}"
        return r"%s \,\dots\, %s" % (_si(cal.temp_lo, r"\degreeCelsius", 1),
                                     _si(cal.temp_hi, r"\degreeCelsius", 1))


def _certificate(d):
    """Page 1: everything the customer gets, on exactly one sheet."""
    inp, cal, mc, hs = d.inp, d.inp.cal, d.inp.magcal, d.inp.housing
    L = [r"\thispagestyle{cert}", r"{\small"]
    L.append(r"\begin{center}{\LARGE\bfseries Calibration Certificate}\\[3pt]"
             r"{\large Inertial measurement unit%s}\end{center}"
             % (" and magnetometer" if mc is not None else ""))
    L.append(r"\vspace{-4pt}\noindent\begin{tabularx}{\linewidth}"
             r"{@{}L{2.9cm}X L{2.6cm}X@{}}\toprule")
    L.append(r"Device & %s & Certificate no. & \texttt{%s} \\"
             % (_given(inp.device), tex(d.cert_id)))
    L.append(r"Serial number & %s & Date & %s \\"
             % (_given(inp.serial), d.t0.strftime("%Y-%m-%d")))
    L.append(r"Username & %s & Time & %s \\"
             % (_given(inp.operator), d.time_span()))
    L.append(r"Location & %s & Coordinates & %s \\"
             % (_given(inp.place), d.position()))
    L.append(r"\bottomrule\end{tabularx}")

    L.append(r"\subsection*{Reference conditions}")
    L.append(r"\vspace{-6pt}\noindent\begin{tabularx}{\linewidth}"
             r"{@{}L{4.4cm}X@{}}\toprule")
    L.append(r"Local gravity & %s, %s \\"
             % (_si(cal.gravity, r"\metre\per\second\squared", 5),
                _given(inp.gravity_source)))
    if mc is not None:
        L.append(r"Magnetic field & %s \\" % d.field_ref())
    L.append(r"IMU temperature & %s \\" % d.temp_range())
    L.append(r"\bottomrule\end{tabularx}")

    L.append(r"\subsection*{Calibration results}")
    L.append(r"\vspace{-6pt}Sensor frame, "
             r"$\mathbf{x}_\mathrm{cal} = \mathbf{M}(\mathbf{x}_\mathrm{raw}"
             r" - \mathbf{b})$. $U$: expanded uncertainty ($k=2$), "
             r"statistical part.")
    L.append(r"\par\noindent\begin{tabularx}{\linewidth}"
             r"{@{}l X r r r r@{}}\toprule "
             r"Sensor & Parameter & $x$ & $y$ & $z$ & $U$ \\ \midrule")
    ub = d.unc["bias"] if d.unc else [float("nan")] * 3
    us = d.unc["scale"] if d.unc else [float("nan")] * 3
    g0 = calib.G_MPS2
    asc = scale_error_percent(cal.acc_matrix)
    L.append(r"Accelerometer & bias [\mg] & %s & %s & %s & %s \\"
             % (*[_num(b / g0 * 1e3, 2, True) for b in cal.acc_bias],
                r"$\leq$ " + _num(2.0 * max(ub) / g0 * 1e3, 2)))
    L.append(r" & scale factor correction [\%%] & %s & %s & %s & %s \\"
             % (*[_num(v, 3, True) for v in asc],
                r"$\leq$ " + _num(2.0 * max(us) * 100.0, 3)))
    gsc = scale_error_percent(cal.gyr_matrix)
    L.append(r"\midrule Gyroscope & bias [\si{\degree\per\second}] & %s & %s "
             r"& %s & %s \\"
             % (*[_num(b * DEG, 3, True) for b in cal.gyr_bias],
                _num(2.0 * d.sig_bg * DEG, 4)))
    L.append(r" & scale factor correction [\%%] & %s & %s & %s & \\"
             % tuple(_num(v, 3, True) for v in gsc))
    if mc is not None:
        L.append(r"\midrule Magnetometer & hard iron [\si{\micro\tesla}] & "
                 r"%s & %s & %s & \\" % tuple(_num(b, 2, True) for b in mc.bias))
        L.append(r" & soft iron, semi-axes [\%%] & %s & %s & %s & \\"
                 % tuple(_num(v, 2, True) for v in mc.soft_iron_percent))
        if mc.align is not None:
            L.append(r" & alignment to IMU, r/p/y [\si{\degree}] & %s & %s & "
                     r"%s & \\" % tuple(_num(v, 2, True) for v in mc.align.rpy_deg))
    if hs is not None:
        L.append(r"\midrule Housing & IMU in housing, r/p/y [\si{\degree}] & "
                 r"%s & %s & %s & \\" % tuple(_num(v, 2, True) for v in hs.rpy_deg))
    L.append(r"\bottomrule\end{tabularx}")

    L.append(r"\subsection*{Verification at rest}")
    L.append(r"\vspace{-6pt}\noindent\begin{tabularx}{\linewidth}"
             r"{@{}X r r r@{}}\toprule "
             r"Quantity & uncalibrated & calibrated & limit \\ \midrule")
    a0, a1 = cal.acc_stats.get("raw", {}), cal.acc_stats.get("full", {})
    L.append(r"$|a|-|g|$ RMS over all static poses [\mg] & %s & %s & %s \\"
             % (_num(a0.get("rms", float("nan")) / g0 * 1e3, 2),
                _num(a1.get("rms", cal.residual_rms) / g0 * 1e3, 2),
                r"$\leq$ " + _num(LIM_ACC_RMS / g0 * 1e3, 1)))
    w0 = cal.gyr_stats.get("raw", {}).get("rms", float("nan"))
    w1 = cal.gyr_stats.get("full", {}).get("rms", cal.gyr_residual_rms)
    L.append(r"Gyroscope direction error RMS [\si{\degree}] & %s & %s & %s \\"
             % (_num(math.degrees(w0), 2), _num(math.degrees(w1), 2),
                r"$\leq$ " + _num(math.degrees(LIM_GYR_DIR_RAD), 2)))
    gr = np.linalg.norm(np.asarray(cal.pose_gyr_raw), axis=1) * DEG \
        if len(cal.pose_gyr_raw) else np.zeros(0)
    gc = np.linalg.norm(np.asarray(cal.pose_gyr_cal), axis=1) * DEG \
        if len(cal.pose_gyr_cal) else np.zeros(0)
    if len(gr):
        L.append(r"Rate at rest, max over poses [\si{\degree\per\second}] & "
                 r"%s & %s & \\" % (_num(float(gr.max()), 3),
                                    _num(float(gc.max()), 3)))
    if mc is not None:
        L.append(r"$|m|$ scatter RMS [\si{\micro\tesla}] & %s & %s & \\"
                 % (_num(mc.raw_rms_ut, 2), _num(mc.residual_rms_ut, 2)))
    L.append(r"Static poses / body axes visited & \multicolumn{2}{r}{%d / "
             r"%d of 6} & $\geq %d$ / 6 \\"
             % (cal.n_positions, sum(d.cov_pose["axes"]), calib.MIN_POSITIONS))
    L.append(r"\bottomrule\end{tabularx}")

    L.append(r"\par\vspace{6pt}\noindent\fbox{\parbox{\dimexpr\linewidth-2"
             r"\fboxsep-2\fboxrule}{\centering\large\textbf{Result: %s}}}"
             % VERDICT_TEXT[d.verdict])
    remark = (inp.remarks or "").strip()
    if remark and len(remark) <= CERT_REMARKS_MAX:
        L.append(r"\par\textbf{Remarks:} %s" % tex(" ".join(remark.split())))
    L.append(r"\par{\sloppy The coefficients apply to the device identified above, "
             r"within the temperature range stated. Coefficient checksum "
             r"\texttt{%s} (SHA-256). A detailed calibration record is kept "
             r"by the manufacturer under the certificate number.\par}"
             % d.digest)

    L.append(r"\par\vspace{4pt}\noindent{\renewcommand{\arraystretch}{1.5}"
             r"\begin{tabularx}{\linewidth}{@{}L{2.2cm}X@{\hspace{1cm}}X@{}}")
    L.append(r"& \textbf{Calibrated by} & \textbf{Checked / released by} \\")
    # Name left blank on purpose: whoever signs writes it in by hand, the
    # login name above is not necessarily a person's name.
    L.append(r"Name & & \\[-6pt] & \rule{\linewidth}{0.3pt} & "
             r"\rule{\linewidth}{0.3pt} \\")
    L.append(r"Date & %s & \\[-6pt] & \rule{\linewidth}{0.3pt} & "
             r"\rule{\linewidth}{0.3pt} \\" % d.t0.strftime("%Y-%m-%d"))
    L.append(r"Signature & & \\[12pt] & \rule{\linewidth}{0.3pt} & "
             r"\rule{\linewidth}{0.3pt} \\")
    L.append(r"\end{tabularx}}")
    # Where page 1 ends: the test builds the PDF and requires this label on
    # page i, which is how "the certificate is one sheet" is enforced.
    L.append(r"\label{cert:end}}")
    return L


def _record(d):
    """The manufacturer's record: everything, on as many pages as it takes."""
    inp, cal, mc, hs, rec = (d.inp, d.inp.cal, d.inp.magcal, d.inp.housing,
                             d.inp.rec)
    t, g0 = d.t, calib.G_MPS2
    sw = software_info()
    L = [r"\clearpage\pagenumbering{arabic}\pagestyle{record}"]
    L.append(r"\begin{center}{\Large\bfseries Calibration Record}\\[2pt]"
             r"for certificate \texttt{%s} -- manufacturer's copy"
             r"\end{center}" % tex(d.cert_id))

    # ---- R1 session ---------------------------------------------------------
    L.append(r"\section{Session}")
    rows = [("Certificate no.", r"\texttt{%s}" % tex(d.cert_id)),
            ("Coefficient checksum", r"\texttt{%s} (SHA-256 of the "
             r"coefficients, truncated)" % d.digest),
            ("Device / serial", "%s / %s" % (_given(inp.device),
                                             _given(inp.serial))),
            ("Username", _given(inp.operator)),
            ("Location", _given(inp.place)),
            ("Coordinates (WGS84)", d.position()),
            ("Recording", "%s, %s" % (d.t0.strftime("%Y-%m-%d"),
                                      d.time_span())),
            ("Document created", inp.created.strftime("%Y-%m-%d %H:%M:%S")),
            ("Data source", _given(inp.source))]
    bi = inp.board_info
    if bi:
        imu = bi.get("imu", {})
        rows.append(("Sensor configuration",
                     r"ODR \SI{%s}{\hertz}, range $\pm$%s\,$g$ / "
                     r"$\pm$\SI{%s}{\degree\per\second}, low-pass "
                     r"\SI{%s}{\hertz} / \SI{%s}{\hertz}"
                     % (imu.get("odr_hz", "?"), imu.get("accel_fs_g", "?"),
                        imu.get("gyro_fs_dps", "?"),
                        imu.get("accel_lpf_hz", "?"),
                        imu.get("gyro_lpf_hz", "?"))))
        rows.append(("Board configuration",
                     r"CRC \texttt{0x%08X}, sequence %d, message version %d, "
                     r"stored nodes acc/gyr/mag %s"
                     % (int(bi.get("cfg_crc", 0)), int(bi.get("cfg_seq", 0)),
                        int(bi.get("version", 0)),
                        "/".join(str(bi.get("npts", {}).get(k, "?"))
                                 for k in ("acc", "gyr", "mag")))))
        rows.append(("On-board correction",
                     "IMU %s, magnetometer %s"
                     % ("on" if bi.get("imu_cal_applied") else "off",
                        "on" if bi.get("mag_cal_applied") else "off")))
    else:
        rows.append(("Sensor configuration",
                     r"\textit{not reported by the board}"))
    frac = inp.rec_cal_frac if inp.rec_cal_frac == inp.rec_cal_frac else 0.0
    rows += [
        ("Samples already corrected", _si(100.0 * frac, r"\percent", 1)),
        ("Samples at full scale", "%d" % getattr(rec, "n_saturated", 0)),
        ("IMU samples", r"%d over %s at %s"
         % (len(rec), _si(cal.duration_sec, r"\second", 1),
            _si(cal.rate_hz, r"\hertz", 2))),
        ("Magnetometer samples", "%d" % len(d.mag)),
        ("Rest period / hold time", "%s / %s"
         % (_si(inp.init_sec, r"\second", 1), _si(inp.pose_sec, r"\second", 1))),
        ("Misalignment estimated", "yes" if cal.misalignment_estimated
         else "no (scale and bias only)"),
        ("Reference gravity", "%s, %s"
         % (_si(cal.gravity, r"\metre\per\second\squared", 6),
            _given(inp.gravity_source))),
        ("Reference field", d.field_ref() if mc is not None or
         inp.field_ref_ut > 0 else r"\textit{no magnetometer}"),
        ("Software", r"INSLIB %s, \texttt{%s}"
         % (tex(sw["version"] or "?"), tex(inp.tool))),
        ("Repository", r"\texttt{%s}" % tex(sw["repo"]) if sw["repo"]
         else r"\textit{unknown}"),
        ("Git commit", (r"\texttt{%s}%s" % (sw["commit"],
                        r" \textbf{with uncommitted changes}"
                        if sw["dirty"] else "")) if sw["commit"]
         else r"\textit{unknown (no git repository)}"),
        ("Frame", "body FRD (x forward, y right, z down), SI units"),
    ]
    L.append(_kv_table(rows))

    # ---- R2 acceptance --------------------------------------------------------
    L.append(r"\section{Acceptance criteria}")
    L.append(r"\par\noindent\begin{tabularx}{\linewidth}"
             r"{@{}X L{3.7cm} L{2.8cm} l@{}}\toprule "
             r"Criterion & Limit & Measured & \\ \midrule")
    for crit, lim, meas, st in d.rows:
        L.append(r"%s & %s & %s & %s \\" % (crit, lim, meas, _state(st)))
    L.append(r"\midrule Overall & & & %s \\" % {
        OK: _state(OK), WARN: _state(WARN), FAIL: _state(FAIL)}[d.verdict])
    L.append(r"\bottomrule\end{tabularx}")
    L.append(r"{\footnotesize fail: too few poses, a fit that does not "
             r"reproduce $|g|$, or data the board had already corrected; "
             r"the coefficients must not be used. note: usable, reduced "
             r"quality.}")

    # ---- R3 coefficients --------------------------------------------------------
    L.append(r"\section{Coefficients}")
    L.append(r"Sensor frame, as stored on the board: "
             r"$\mathbf{x}_\mathrm{cal} = \mathbf{M}(\mathbf{x}_\mathrm{raw}"
             r" - \mathbf{b})$. Full precision, row-major.")
    L.append(r"\subsection{Accelerometer}")
    L.append(r"\[\mathbf{M}_a = %s \quad \mathbf{b}_a = %s\,"
             r"\si{\metre\per\second\squared}\]"
             % (_matrix(cal.acc_matrix, 9),
                _matrix([[b] for b in cal.acc_bias], 9)))
    u = d.unc
    L.append(r"\begin{center}\begin{tabular}{@{}lrrrrrr@{}}\toprule "
             r"Axis & bias [\mg] & $u$ [\mg] & scale corr. [\%] & $u$ [ppm] "
             r"& misalign. [mrad] & $u$ [mrad] \\ \midrule")
    mis = acc_params(cal.acc_matrix, cal.acc_bias)[0:3]
    asc = scale_error_percent(cal.acc_matrix)
    for i, (ax, mname) in enumerate(zip("xyz", ("yz", "zy", "zx"))):
        L.append(r"%s & %s & %s & %s & %s & %s (%s) & %s \\"
                 % (ax, _num(cal.acc_bias[i] / g0 * 1e3, 3, True),
                    _num(u["bias"][i] / g0 * 1e3, 3) if u else "--",
                    _num(asc[i], 4, True),
                    _num(u["scale"][i] * 1e6, 0) if u else "--",
                    _num(mis[i] * 1e3, 3, True), mname,
                    _num(u["mis"][i] * 1e3, 3) if u and u["mis"] else "--"))
    L.append(r"\bottomrule\end{tabular}\end{center}")
    if u:
        L.append(r"Standard uncertainties ($k=1$) from the covariance of the "
                 r"fit over one mean per static pose: residual standard "
                 r"deviation %s, %d degrees of freedom. Statistical (Type A) "
                 r"part only."
                 % (_si(u["s"], r"\metre\per\second\squared", 5), u["dof"]))
    L.append(r"\subsection{Gyroscope}")
    L.append(r"\[\mathbf{M}_\omega = %s \quad \mathbf{b}_\omega = %s\,"
             r"\si{\radian\per\second}\]"
             % (_matrix(cal.gyr_matrix, 9),
                _matrix([[b] for b in cal.gyr_bias], 9)))
    gsc = scale_error_percent(cal.gyr_matrix)
    L.append(r"\begin{center}\begin{tabular}{@{}lrrr@{}}\toprule "
             r"Axis & bias [\si{\degree\per\second}] & $u$ "
             r"[\si{\degree\per\second}] & scale corr. [\%] \\ \midrule")
    for i, ax in enumerate("xyz"):
        L.append(r"%s & %s & %s & %s \\"
                 % (ax, _num(cal.gyr_bias[i] * DEG, 5, True),
                    _num(d.sig_bg * DEG, 5), _num(gsc[i], 4, True)))
    L.append(r"\bottomrule\end{tabular}\end{center}")
    L.append(r"$u$ of the bias is the white-noise limit of averaging over "
             r"the rest period, $\sqrt{\mathrm{PSD}/T}$. Bias instability is "
             r"not included.")
    if mc is not None:
        L.append(r"\subsection{Magnetometer}")
        L.append(r"\[\mathbf{M}_m = %s \quad \mathbf{b}_m = %s\,"
                 r"\si{\micro\tesla}\]"
                 % (_matrix(mc.matrix, 9), _matrix([[b] for b in mc.bias], 6)))
        rows = [
            (r"Hard iron $|\mathbf{b}_m|$", _si(mc.hard_iron_ut, r"\micro\tesla", 3)),
            ("Soft iron, semi-axes", r"%s \si{\percent}"
             % ", ".join(_num(v, 3, True) for v in mc.soft_iron_percent)),
            ("Scaled to", r"%s (%s)" % (_si(mc.field_ut, r"\micro\tesla", 3),
                                        tex(mc.field_source))),
            ("Field measured by the fit", _si(mag_fit_radius(mc),
                                              r"\micro\tesla", 3)),
            (r"$|m|$ scatter uncal. $\rightarrow$ cal.",
             r"%s $\rightarrow$ %s" % (_si(mc.raw_rms_ut, r"\micro\tesla", 4),
                                        _si(mc.residual_rms_ut,
                                            r"\micro\tesla", 4))),
            ("Samples used / rejected", "%d / %d" % (mc.n_samples,
                                                     mc.n_dropped)),
            ("Direction coverage", r"\num{%.3f}" % mc.spread)]
        if mc.align is not None:
            al = mc.align
            rows += [
                ("Alignment to IMU (roll, pitch, yaw)",
                 r"%s, %s, %s (total %s)"
                 % (*[_si(v, r"\degree", 3, True) for v in al.rpy_deg],
                    _si(al.angle_deg, r"\degree", 3))),
                ("Inclination measured%s" % (" / WMM" if inp.wmm_incl_deg
                                              is not None else ""),
                 _si(al.dip_deg, r"\degree", 3, True)
                 + ((" / " + _si(inp.wmm_incl_deg, r"\degree", 3, True))
                    if inp.wmm_incl_deg is not None else "")),
                ("Inclination scatter over %d poses" % al.n_poses,
                 r"%s $\rightarrow$ %s"
                 % (_si(al.scatter_before_deg, r"\degree", 3),
                    _si(al.scatter_after_deg, r"\degree", 3))),
                ("Alignment observability", r"\num{%.3f}" % al.observability)]
        L.append(_kv_table(rows, "5.4cm"))
    L.append(r"\subsection{Housing alignment}")
    if hs is not None:
        L.append(r"From %d placement(s) on a level surface: %s."
                 % (len(hs.placements),
                    tex(", ".join(fa.PLACEMENT_LABEL.get(p, p)
                                  for p in hs.placements))))
        L.append(r"\[\mathbf{R}_\mathrm{housing} = %s\]" % _matrix(hs.matrix, 9))
        L.append(r"IMU in the housing: roll %s, pitch %s, yaw %s, total %s%s. "
                 r"Residual per placement: %s. Applied on top of all three "
                 r"sensors: $\mathbf{x} = \mathbf{R}_\mathrm{housing}"
                 r"\mathbf{M}(\mathbf{x}_\mathrm{raw} - \mathbf{b})$."
                 % (*[_si(v, r"\degree", 3, True) for v in hs.rpy_deg],
                    _si(hs.angle_deg, r"\degree", 3),
                    " (roll and pitch only, yaw not observable)"
                    if hs.tilt_only else "",
                    ", ".join(_si(r, r"\degree", 3) for r in hs.residual_deg)
                    or "--"))
    else:
        L.append(r"\textit{Not measured in this session.}")

    # ---- R4 verification ------------------------------------------------------
    L.append(r"\section{Verification}")
    L.append(r"Over every sample of the static poses used by the fit. Rows "
             r"are cumulative: uncalibrated, bias only, full model.")
    L.append(r"\begin{center}\begin{tabular}{@{}lrrrr@{}}\toprule"
             r"& \multicolumn{3}{c}{$|a|-|g|$ [\si{\metre\per\second\squared}]}"
             r" & gyroscope \\ \cmidrule(lr){2-4}"
             r"Model & RMS & mean & max & direction RMS [\si{\degree}] \\ "
             r"\midrule")
    for label, key in (("uncalibrated", "raw"), ("+ bias", "bias"),
                       ("+ scale/misalignment" if cal.misalignment_estimated
                        else "+ scale", "full")):
        a = cal.acc_stats.get(key)
        if a is None:
            continue
        gr = cal.gyr_stats.get(key, {}).get("rms", float("nan"))
        L.append(r"%s & %s & %s & %s & %s \\"
                 % (label, _num(a["rms"], 5), _num(a["mean"], 5, True),
                    _num(a["max"], 5), _num(math.degrees(gr), 3)
                    if gr == gr else "--"))
    L.append(r"\bottomrule\end{tabular}\end{center}")

    an_raw = np.linalg.norm(d.acc, axis=1)
    an_cal = np.linalg.norm(d.acc_cal, axis=1)
    dev = [abs(v - cal.gravity) for v in list(cal.pose_acc_raw)
           + list(cal.pose_acc_cal)]
    half = min(max(0.15, 3.0 * max(dev) if dev else 0.5), 5.0)
    L.append(_axis_head(
        r"Magnitude of acceleration (window around $|g|$)",
        r"xlabel={time [s]}, ylabel={$|a|$ [\si{\metre\per\second"
        r"\squared}]}, ymin=%.4f, ymax=%.4f, xmin=0, xmax=%.1f"
        % (cal.gravity - half, cal.gravity + half, t[-1])))
    L.append(_addplot("rawc, thin", _envelope(t, an_raw), "uncalibrated"))
    L.append(_addplot("calc, thin", _envelope(t, an_cal), "calibrated"))
    L.append(r"\addplot[refc, dashed, domain=0:%.1f] {%.5f};"
             r"\addlegendentry{reference $|g|$}" % (t[-1], cal.gravity))
    L.append(AXIS_TAIL)
    floor = 1e-3
    L.append(_axis_head(
        r"Magnitude of angular rate (log scale, calibrated $|\omega| "
        r"\approx 0$ at rest)",
        r"xlabel={time [s]}, ylabel={$|\omega|$ [\si{\degree\per"
        r"\second}]}, ymode=log, xmin=0, xmax=%.1f" % t[-1]))
    L.append(_addplot("rawc, thin", _envelope(
        t, np.maximum(np.linalg.norm(d.gyr, axis=1) * DEG, floor)),
        "uncalibrated"))
    L.append(_addplot("calc, thin", _envelope(
        t, np.maximum(np.linalg.norm(d.gyr_cal, axis=1) * DEG, floor)),
        "calibrated"))
    L.append(AXIS_TAIL)
    if d.mag_cal is not None:
        L.append(_axis_head(
            r"Magnitude of magnetic field",
            r"xlabel={time [s]}, ylabel={$|m|$ [\si{\micro\tesla}]}, "
            r"xmin=0, xmax=%.1f" % t[-1]))
        L.append(_addplot("rawc, thin", _envelope(
            d.tm, np.linalg.norm(d.mag, axis=1)), "uncalibrated"))
        L.append(_addplot("calc, thin", _envelope(
            d.tm, np.linalg.norm(d.mag_cal, axis=1)), "calibrated"))
        L.append(r"\addplot[refc, dashed, domain=0:%.1f] {%.3f};"
                 r"\addlegendentry{reference $|F|$}" % (t[-1], mc.field_ut))
        L.append(AXIS_TAIL)
    n = len(cal.pose_acc_raw)
    if n:
        x = np.arange(1, n + 1)
        L.append(_axis_head(
            r"Per pose: deviation from $|g|$",
            r"height=4.6cm, xlabel={static pose}, ylabel={$|a|-|g|$ [\mg]}, "
            r"xmin=0.5, xmax=%.1f" % (n + 0.5)))
        L.append(_addplot("rawc, only marks, mark=o, mark size=1.6pt",
                          list(zip(x, (np.asarray(cal.pose_acc_raw)
                                       - cal.gravity) / g0 * 1e3)),
                          "uncalibrated"))
        L.append(_addplot("calc, only marks, mark=*, mark size=1.6pt",
                          list(zip(x, (np.asarray(cal.pose_acc_cal)
                                       - cal.gravity) / g0 * 1e3)),
                          "calibrated"))
        L.append(r"\addplot[refc, dashed, domain=0.5:%.1f] {0};" % (n + 0.5))
        L.append(AXIS_TAIL)
        L.append(_axis_head(
            r"Per pose: mean angular rate at rest",
            r"height=4.6cm, xlabel={static pose}, ylabel={$|\bar\omega|$ "
            r"[\si{\degree\per\second}]}, xmin=0.5, xmax=%.1f, ymin=0"
            % (n + 0.5)))
        L.append(_addplot("rawc, only marks, mark=o, mark size=1.6pt",
                          list(zip(x, np.linalg.norm(np.asarray(
                              cal.pose_gyr_raw), axis=1) * DEG)),
                          "uncalibrated"))
        L.append(_addplot("calc, only marks, mark=*, mark size=1.6pt",
                          list(zip(x, np.linalg.norm(np.asarray(
                              cal.pose_gyr_cal), axis=1) * DEG)),
                          "calibrated"))
        L.append(AXIS_TAIL)

    # ---- R5 temperature ----------------------------------------------------
    L.append(r"\section{Temperature}")
    temps = np.asarray(getattr(rec, "temp_c", []), dtype=float)
    mt = np.asarray(rec.mag_temp_c, dtype=float)
    rows = [("IMU", r"%s (span %s)" % (
        d.temp_range(), _si(cal.temp_hi - cal.temp_lo, r"\kelvin", 2)))]
    if np.isfinite(mt).any():
        rows.append(("Magnetometer", r"%s \,\dots\, %s"
                     % (_si(float(np.nanmin(mt)), r"\degreeCelsius", 1),
                        _si(float(np.nanmax(mt)), r"\degreeCelsius", 1))))
    L.append(_kv_table(rows))
    if len(temps) == len(t) and np.isfinite(temps).sum() > 1:
        ok = np.isfinite(temps)
        L.append(_axis_head(
            r"Sensor temperature over the session",
            r"height=4.2cm, xlabel={time [s]}, ylabel={temperature "
            r"[\si{\degreeCelsius}]}, xmin=0, xmax=%.1f" % t[-1]))
        L.append(_addplot("calc, thick", _envelope(t[ok], temps[ok], 250),
                          "IMU"))
        okm = np.isfinite(mt)
        if okm.sum() > 1 and len(d.tm) == len(mt):
            L.append(_addplot("rawc, thick", _envelope(d.tm[okm], mt[okm], 250),
                              "magnetometer"))
        L.append(AXIS_TAIL)

    # ---- R6 coverage -------------------------------------------------------
    L.append(r"\section{Orientation coverage}")
    L.append(r"Direction of the measured specific force of every static pose "
             r"in the sensor frame (azimuth $\operatorname{atan2}(a_y, a_x)$, "
             r"elevation $\arcsin(a_z/|a|)$), and of the calibrated magnetic "
             r"field. Crosses: the six body axes.")
    ax_names = ["+x", "-x", "+y", "-y", "+z", "-z"]
    rows = [("Static poses", "%d" % d.cov_pose["n"]),
            ("Spread (1 even, 0 planar)", r"\num{%.3f}" % d.cov_pose["spread"]),
            ("Largest empty cap, radius", _si(d.cov_pose["gap_deg"],
                                              r"\degree", 1)),
            (r"Axes visited (within \SI{%.0f}{\degree})" % AXIS_COVER_DEG,
             ", ".join("%s %s" % (nm, r"\checkmark" if c else r"$\times$")
                       for nm, c in zip(ax_names, d.cov_pose["axes"])))]
    if d.cov_mag is not None:
        rows.append(("Magnetic field: spread / largest gap",
                     r"\num{%.3f} / %s" % (d.cov_mag["spread"],
                                           _si(d.cov_mag["gap_deg"],
                                               r"\degree", 1))))
    L.append(_kv_table(rows, "5.4cm"))
    L.append(r"\begin{center}\begin{tikzpicture}\begin{axis}["
             r"width=0.8\linewidth, height=7cm, xmin=-180, xmax=180, "
             r"ymin=-90, ymax=90, xtick={-180,-135,...,180}, "
             r"ytick={-90,-45,...,90}, xlabel={azimuth [\si{\degree}]}, "
             r"ylabel={elevation [\si{\degree}]}, legend pos=outer north east, "
             r"legend style={font=\scriptsize}]")
    if d.mag_cal is not None:
        step = max(1, len(d.mag_cal) // 1200)
        L.append(_addplot("violet!45, only marks, mark=*, mark size=0.5pt",
                          _dirs_azel(d.mag_cal[::step]), "magnetic field"))
    L.append(_addplot("calc, only marks, mark=*, mark size=2.2pt",
                      _dirs_azel(d.pa), "poses (gravity)"))
    L.append(_addplot("black, only marks, mark=x, mark size=4pt",
                      _dirs_azel(np.vstack([np.eye(3), -np.eye(3)])), "axes"))
    L.append(r"\end{axis}\end{tikzpicture}\end{center}")

    # ---- R7 noise ------------------------------------------------------------
    L.append(r"\section{Noise (rest period)}")
    arw = math.sqrt(cal.gyr_psd) * DEG * 60.0 if cal.gyr_psd > 0 else float("nan")
    vrw = math.sqrt(cal.acc_psd) * 60.0 if cal.acc_psd > 0 else float("nan")
    L.append(_kv_table([
        ("Gyroscope PSD / angle random walk",
         r"\num{%.4e} $\mathrm{(rad/s)^2/Hz}$ / %s\sqrth"
         % (cal.gyr_psd, _si(arw, r"\degree", 4))),
        ("Accelerometer PSD / velocity random walk",
         r"\num{%.4e} $\mathrm{(m/s^2)^2/Hz}$ / %s\sqrth"
         % (cal.acc_psd, _si(vrw, r"\metre\per\second", 5)))], "6.2cm"))

    # ---- R8 notes ------------------------------------------------------------
    L.append(r"\section{Solver notes and remarks}")
    notes = solver_notes(inp)
    if notes:
        L.append(r"\begin{itemize}")
        L += [r"\item \small %s" % tex(w) for w in notes]
        L.append(r"\end{itemize}")
    else:
        L.append(r"The solvers raised no warnings.")
    if (inp.remarks or "").strip():
        L.append(r"\par\textbf{Remarks:}\\ %s"
                 % tex(inp.remarks.strip()).replace("\n", r"\\ "))

    # ---- R9 per pose ---------------------------------------------------------
    if n:
        L.append(r"\section{Static poses: accelerometer}")
        L.append(r"Raw means over each static pose as found by the solver "
                 r"(sample index range into the attached IMU file, time "
                 r"relative to its first sample), and $|a|$ before and after "
                 r"calibration.")
        L.append(r"{\footnotesize\setlength{\tabcolsep}{3.5pt}"
                 r"\begin{longtable}{@{}rrrrrrrrr@{}}\toprule "
                 r"\# & samples & $t$ [s] & $\bar a_x$ & $\bar a_y$ & "
                 r"$\bar a_z$ [\si{\metre\per\second\squared}] & "
                 r"$|a|-|g|$ raw [\mg] & cal. [\mg] & $T$ [\si{\degreeCelsius}]"
                 r" \\ \midrule\endhead \bottomrule\endfoot ")
        temps_ok = len(temps) == len(t)
        for i, (s, e) in enumerate(cal.intervals):
            tp = float(np.nanmean(temps[s:e + 1])) if temps_ok and \
                np.isfinite(temps[s:e + 1]).any() else float("nan")
            L.append(r"%d & %d--%d & %s & %s & %s & %s & %s & %s & %s \\"
                     % (i + 1, s, e, _num(t[s], 2),
                        *[_num(v, 6, True) for v in d.pa[i]],
                        _num((cal.pose_acc_raw[i] - cal.gravity) / g0 * 1e3,
                             3, True),
                        _num((cal.pose_acc_cal[i] - cal.gravity) / g0 * 1e3,
                             3, True), _num(tp, 1)))
        L.append(r"\end{longtable}}")

        L.append(r"\section{Static poses: gyroscope and magnetometer}")
        L.append(r"Raw means over the same poses. The magnetometer columns "
                 r"average its samples inside the time span of each pose.")
        L.append(r"{\footnotesize\setlength{\tabcolsep}{3.5pt}"
                 r"\begin{longtable}{@{}rrrrrrrrr@{}}\toprule "
                 r"\# & $\bar\omega_x$ & $\bar\omega_y$ & $\bar\omega_z$ "
                 r"[\si{\degree\per\second}] & $|\bar\omega|$ cal. & "
                 r"$\bar m_x$ & $\bar m_y$ & $\bar m_z$ [\si{\micro\tesla}] & "
                 r"$N_m$ \\ \midrule\endhead \bottomrule\endfoot ")
        gcal = np.linalg.norm(np.asarray(cal.pose_gyr_cal), axis=1) * DEG
        for i, (s, e) in enumerate(cal.intervals):
            if len(d.mag):
                sel = (d.tm >= t[s]) & (d.tm <= t[e])
                nm = int(sel.sum())
                mm = d.mag[sel].mean(axis=0) if nm else [float("nan")] * 3
            else:
                nm, mm = 0, [float("nan")] * 3
            L.append(r"%d & %s & %s & %s & %s & %s & %s & %s & %d \\"
                     % (i + 1, *[_num(v * DEG, 5, True) for v in d.pg[i]],
                        _num(float(gcal[i]), 4),
                        *[_num(float(v), 3, True) for v in mm], nm))
        L.append(r"\end{longtable}}")

    # ---- R10 raw data --------------------------------------------------------
    L.append(r"\section{Raw data and reproduction}")
    if d.raw:
        L.append(r"The complete recording is attached to this PDF (open the "
                 r"attachments panel of the PDF viewer) and was written next "
                 r"to the \texttt{.tex} source. Replay-format CSV, time "
                 r"stamps in microseconds of the board clock.")
        L.append(r"\par\noindent\begin{tabularx}{\linewidth}{@{}X r@{}}"
                 r"\toprule File, SHA-256 & rows \\ \midrule")
        for f in d.raw:
            L.append(r"\texttt{%s} & %d \\ \multicolumn{2}{@{}l}{\quad"
                     r"\footnotesize\texttt{%s}} \\[2pt]"
                     % (tex(f.name), f.rows, f.sha256))
        L.append(r"\bottomrule\end{tabularx}")
        for f in d.raw:
            L.append(r"\IfFileExists{%s}{\embedfile[desc={%s}]{%s}}{}"
                     % (f.name, tex(f.what), f.name))
        L.append(r"\par The calibration is recomputed from these files, "
                 r"with the inputs of this session, by%s"
                 % ((r" (in \texttt{%s} at commit \texttt{%s}%s)"
                     % (tex(sw["repo"]), sw["commit"][:12],
                        ", plus the uncommitted changes of that session"
                        if sw["dirty"] else ""))
                    if sw["commit"] else ""))
        L.append(r"\begin{center}\begin{minipage}{0.95\linewidth}\raggedright"
                 r"\footnotesize\ttfamily %s\end{minipage}\end{center}"
                 % tex(reproduce_command(inp, d.raw)).replace("-", "-{}"))
        L.append(r"and has to give the coefficients above; their checksum is "
                 r"\texttt{%s}." % d.digest)
    else:
        L.append(r"\textit{The raw recording was not written with this "
                 r"document.}")
    L.append(r"\label{rec:last}")
    return L


def build_tex(inp, raw=None):
    """The whole document as one string. `raw` is write_raw()'s list."""
    d = _Doc(inp, raw)
    L = [PREAMBLE]
    L.append(r"\fancypagestyle{cert}{\fancyhf{}"
             r"\lhead{\textbf{Calibration Certificate}}"
             r"\rhead{No.~\texttt{%s}}"
             r"\rfoot{\footnotesize Page 1 of 1}"
             r"\lfoot{\footnotesize Checksum \texttt{%s}}"
             r"\renewcommand{\headrulewidth}{0.4pt}}" % (tex(d.cert_id),
                                                          d.digest))
    L.append(r"\fancypagestyle{record}{\fancyhf{}"
             r"\lhead{\textbf{Calibration Record} (manufacturer)}"
             r"\rhead{Certificate No.~\texttt{%s}}"
             r"\lfoot{\footnotesize INSLIB, \texttt{%s}}"
             r"\rfoot{\footnotesize Record page \thepage\ of "
             r"\pageref{rec:last}}"
             r"\renewcommand{\headrulewidth}{0.4pt}}"
             % (tex(d.cert_id), tex(inp.tool)))
    L.append(r"\hypersetup{pdftitle={Calibration Certificate %s}}"
             % tex(d.cert_id))
    L.append(r"\begin{document}")
    # Roman on the certificate, arabic on the record: two numberings, so
    # hyperref's page anchors do not collide and the record starts at 1.
    L.append(r"\pagenumbering{roman}")
    L += _certificate(d)
    L += _record(d)
    L.append(r"\end{document}")
    return "\n".join(s for s in L if s) + "\n"


def write_report(path, inp, makefile=True, raw=True):
    """Write the .tex, the raw CSVs and a Makefile. Returns the paths.

    An existing Makefile is only replaced when this module wrote it: a
    certificate saved into a directory that already has a build of its
    own must not take that build over."""
    base = os.path.splitext(path)[0]
    files = write_raw(inp.rec, base) if raw else []
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(build_tex(inp, files))
    folder = os.path.dirname(os.path.abspath(path))
    written = [path] + [os.path.join(folder, f.name) for f in files]
    if makefile:
        mk = os.path.join(folder, "Makefile")
        ours = True
        if os.path.exists(mk):
            try:
                with open(mk, encoding="utf-8", errors="replace") as f:
                    ours = f.readline().startswith(MAKEFILE_MARK)
            except OSError:
                ours = False
        if ours:
            with open(mk, "w", encoding="utf-8", newline="\n") as f:
                f.write(MAKEFILE)
            written.append(mk)
    return written
