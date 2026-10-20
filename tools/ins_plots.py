#!/usr/bin/env python3
"""matplotlib output for tools/replay.py --plot: state history with a
1-σ band (from the filter's own error-state covariance, see
ins.Navigator.stddev()) and, when a ground-truth reference is available,
error-over-time plots with the reference overlaid on the estimate.

Kept out of the INSLIB package proper (python/INSLIB/) since matplotlib is a
replay-tool convenience, not part of the reusable ctypes binding.

(c) Jan Zwiener (jan@zwiener.org)
"""

import math
import os
import warnings

EST_COLOR = "tab:red"    # the estimate, everywhere it appears
REF_COLOR = "tab:blue"   # the ground truth, everywhere it appears
BARO_COLOR = "tab:green" # baro_alt, on the dedicated altitude page
GNSS_COLOR = "0.35"      # raw GNSS fixes, on the dedicated altitude page
BARO_RAW_COLOR = "0.6"   # raw (unfiltered) barometer altitude, altitude page
MAG_RAW_COLOR = "0.6"    # uncalibrated magnetometer field, WMM page
ELLIPSOID_COLOR = "tab:purple"  # nav_suite_get_height_ellipsoid(), altitude page
RTS_COLOR = "#c2188f"    # the RTS smoother's solution (tools/ins_rts.py), its pages
RTS_FILTER_COLOR = "0.25"  # the filter as the thin comparison line on those pages

# Outlier-rejection page: one fixed color per sub-filter, consistent with
# subfilter_overlay_tree's naming in replay.py.
DOWNWEIGHT_COLORS = {
    "full3d": EST_COLOR, "ars": "tab:orange", "ahrs": "tab:purple",
    "baro_alt": BARO_COLOR, "local_gnss": "tab:brown",
}
DOWNWEIGHT_LABELS = {
    "full3d": "INSLIB", "ars": "ARS", "ahrs": "AHRS",
    "baro_alt": "baro_alt", "local_gnss": "baro/GNSS offset",
}

# Sensor-rate page: one fixed color per stream.
SENSOR_RATE_COLORS = {"imu": EST_COLOR, "gnss": "tab:gray",
                      "mag": "tab:purple", "baro": BARO_COLOR}

# Dr. INS findings page: severity int (0=OK, 1=INFO, 2=WARN, 3=CRIT, the
# same ordering as replay.insdoctor_findings) -> print color + tag. Bare
# ints on purpose, so this module needs no import of replay.py.
_FINDING_COLOR = {3: "#c0392b", 2: "#c87f0a", 1: "#2471a3", 0: "#1e8449"}
_FINDING_TAG = {3: "CRIT", 2: "WARN", 1: "INFO", 0: " OK "}

# Print-friendly serif face for every page; falls back to matplotlib's
# built-in DejaVu Serif (no external font install needed).
PLOT_FONT_FAMILY = "DejaVu Serif"

# A4-scale PDF: this many vector points per line is already beyond what's
# visually distinguishable when printed, but long trials at the
# --plot-hz recorder rate produce tens of thousands of points per line
# otherwise.
_MAX_PLOT_POINTS = 2000

# A receiver reports a wildly inflated stddev while it has no real fix (a
# tunnel, a re-acquisition), and a filter's own sigma grows without bound
# across a long dead-reckoning stretch: a single 600 m spike rescales the axis
# so hard that the 0.05 m of the rest of the run is a flat line on the zero
# gridline. Bound the VIEW instead and say so in the corner - the peak's own
# value goes in the caption, the shape of the trace is what the page is for.
#
# Two bounds, whichever is tighter. The quantile is what usually decides: it
# follows the data, so a run that legitimately sits at 50 m still shows 50 m.
# The hard cap is the backstop for a run where even the quantile is absurd
# (a receiver reporting 1e6 m for a third of the trial).
_SIGMA_AXIS_CAP_M = 100.0
_SIGMA_AXIS_CAP_MPS = 10.0
_SIGMA_AXIS_QUANTILE = 0.999  # keep essentially everything but the spikes
_SIGMA_AXIS_HEADROOM = 1.5    # ... and leave air above it


def _quantile(sorted_vals, q):
    """Nearest-rank quantile of an already-sorted, non-empty list."""
    i = int(q * (len(sorted_vals) - 1) + 0.5)
    return sorted_vals[i]


def _sigma_axis_limit(values, cap):
    """Upper view bound for a stddev-like axis, and the true peak.

    Returns (limit, peak), or (None, None) when there is nothing finite to
    plot. limit == peak means the whole series fits and nothing is clipped.
    `values` must be the FULL series, not a decimated one: the decimation
    keeps per-bucket extremes, which would drag the quantile up to the very
    outlier this is meant to look past."""
    finite = sorted(v for v in values if v == v)
    if not finite:
        return None, None
    peak = finite[-1]
    limit = min(cap, _quantile(finite, _SIGMA_AXIS_QUANTILE) * _SIGMA_AXIS_HEADROOM)
    if limit <= 0.0 or limit >= peak:
        return peak, peak
    return limit, peak

# North-East map page: a ground track is a 2D curve, so the per-bucket
# min/max decimation below (which ranks samples along ONE axis) is wrong
# for it - keeping only the northmost/southmost sample of a bucket throws
# the East detail away and turns a tight loop into a star of long chords.
# The map is simplified geometrically instead (see _thin_track): every
# vertex that pulls the drawn line more than _MAP_REL_TOL of the track's
# OWN extent away from the true path survives. Relative, so the detail
# follows the scale of the view: a 2 m track keeps sub-millimetre
# vertices, a 10 km one keeps metres, and both cost about the same page
# size. _MAX_MAP_POINTS is the backstop for a track that is pure noise at
# that tolerance (the tolerance is relaxed until it fits).
_MAP_REL_TOL = 1.0 / 4000.0
_MAX_MAP_POINTS = 20000


def _decimate_indices(y, max_points=_MAX_PLOT_POINTS):
    """Indices to keep from `y` (len n), at most ~max_points: per bucket,
    the local min AND max (not just every Nth point), so a brief spike
    or dropout survives instead of being silently averaged/strided away.
    NaN-safe -- a NaN-only bucket keeps one NaN sample so the gap in the
    line survives too. Returns range(n) unchanged if n <= max_points."""
    n = len(y)
    if n <= max_points:
        return range(n)
    n_buckets = max(1, max_points // 2)
    bucket_size = n / n_buckets
    out = set()
    for b in range(n_buckets):
        lo = int(b * bucket_size)
        hi = min(int((b + 1) * bucket_size), n)
        if hi <= lo:
            continue
        valid = [i for i in range(lo, hi) if y[i] == y[i]]  # drop NaN
        if not valid:
            out.add(lo)
            continue
        out.add(min(valid, key=lambda i: y[i]))
        out.add(max(valid, key=lambda i: y[i]))
    return sorted(out)


def _thin(key, *others, max_points=_MAX_PLOT_POINTS):
    """Decimate `key` and any number of parallel `others` together (the
    same kept indices for all), choosing indices from `key`'s own local
    min/max per bucket -- so e.g. (center, t, sigma) stay aligned and
    the selection is driven by the actual data, not the (monotonic)
    time axis. Returns (key, *others), downsampled if needed."""
    idx = _decimate_indices(key, max_points)
    if len(idx) == len(key):
        return (key,) + others
    return tuple([a[i] for i in idx] for a in (key,) + others)


def _rdp_keep(x, y, eps):
    """Ramer-Douglas-Peucker simplification of the polyline (x, y):
    returns the sorted indices to keep, i.e. a subset whose polyline
    never deviates from a dropped point by more than eps. Endpoints are
    always kept. Iterative (explicit stack), so a long noisy track cannot
    hit Python's recursion limit."""
    n = len(x)
    if n < 3:
        return list(range(n))
    keep = [False] * n
    keep[0] = keep[n - 1] = True
    stack = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        x0, y0 = x[lo], y[lo]
        dx, dy = x[hi] - x0, y[hi] - y0
        chord = math.hypot(dx, dy)
        best_i, best_d = -1, eps
        if chord > 0.0:
            for i in range(lo + 1, hi):
                d = abs(dy * (x[i] - x0) - dx * (y[i] - y0)) / chord
                if d > best_d:
                    best_i, best_d = i, d
        else:
            # Degenerate chord (the track returned to its start): rank by
            # the distance to that common endpoint instead, otherwise a
            # whole excursion between two identical samples is dropped.
            for i in range(lo + 1, hi):
                d = math.hypot(x[i] - x0, y[i] - y0)
                if d > best_d:
                    best_i, best_d = i, d
        if best_i >= 0:
            keep[best_i] = True
            stack.append((lo, best_i))
            stack.append((best_i, hi))
    return [i for i in range(n) if keep[i]]


def _thin_track(x, y, rel_tol=_MAP_REL_TOL, max_points=_MAX_MAP_POINTS):
    """Decimate a 2D ground track for the map page: keep the shape to
    within rel_tol of the track's own bounding-box extent (so the
    absolute resolution scales with the plotted view, see _MAP_REL_TOL),
    at most max_points vertices. NaN samples split the track into
    segments that are simplified separately and rejoined with a single
    NaN, so a gap in the line survives. Returns (x, y) lists."""
    finite = [(a, b) for a, b in zip(x, y) if a == a and b == b]
    if len(x) < 3 or not finite:
        return list(x), list(y)

    span = max(max(p[0] for p in finite) - min(p[0] for p in finite),
               max(p[1] for p in finite) - min(p[1] for p in finite))
    eps = span * rel_tol

    segs, cur_x, cur_y = [], [], []
    for a, b in zip(x, y):
        if a == a and b == b:
            cur_x.append(a)
            cur_y.append(b)
        elif cur_x:
            segs.append((cur_x, cur_y))
            cur_x, cur_y = [], []
    if cur_x:
        segs.append((cur_x, cur_y))

    # A track that is pure noise at this tolerance (or one that never
    # moved at all, span 0) keeps every sample: relax until it fits.
    # Bounded on purpose - doubling from the smallest useful epsilon
    # reaches any real-world extent well inside this many steps.
    kept = None
    for _ in range(80):
        kept = [(sx, sy, _rdp_keep(sx, sy, eps)) for sx, sy in segs]
        if sum(len(k[2]) for k in kept) <= max_points:
            break
        eps = eps * 2.0 if eps > 0.0 else 1e-9

    out_x, out_y = [], []
    for sx, sy, idx in kept:
        if out_x:
            out_x.append(math.nan)
            out_y.append(math.nan)
        out_x.extend(sx[i] for i in idx)
        out_y.extend(sy[i] for i in idx)
    return out_x, out_y


def _band(ax, t, center, sigma, label, color):
    center, t, sigma = _thin(center, t, sigma)
    lo = [c - s for c, s in zip(center, sigma)]
    hi = [c + s for c, s in zip(center, sigma)]
    ax.plot(t, center, label=label, color=color)
    ax.fill_between(t, lo, hi, color=color, alpha=0.25, linewidth=0)


def _col(rows, i):
    return [r[i] for r in rows]


def _last_valid_row(rows):
    """Last row (from the end) whose first element is not NaN, or None --
    rec's per-group rows go all-NaN together (see replay.py's rec-building
    loop), so checking element 0 is enough to know the whole row is valid."""
    for row in reversed(rows):
        if row and row[0] == row[0]:
            return row
    return None


def _apply_grid(fig):
    """Light gridlines on every Axes of `fig` -- called once per figure,
    after its layout is finalized, so it also covers axes turned off
    later (axis("off") already hides the grid along with everything
    else)."""
    for ax in fig.axes:
        ax.grid(True, alpha=0.3, linewidth=0.5)


def _zupt_spans(t, zupt):
    """Contiguous [start, end] time intervals where zupt (parallel to t,
    1.0/0.0 per sample) is active."""
    spans = []
    start = None
    for ti, zi in zip(t, zupt):
        if zi and start is None:
            start = ti
        elif not zi and start is not None:
            spans.append((start, ti))
            start = None
    if start is not None:
        spans.append((start, t[-1]))
    return spans


def _shade_zupt(fig, t, zupt):
    """Shade detected stationary phases (gray band) on every Axes of
    `fig` that shares the (t) time axis -- so stops can be visually
    correlated with bias jumps/state changes, the same rationale as the
    PlotJuggler INSLIB/zaru/* topics (see replay.subfilter_overlay_tree).
    `zupt` is ins's own velocity-aware auto-ZUPT/ZARU (REQ-SUITE-009/-010)
    OR'd with the suite's vertical-channel ZUPT (REQ-SUITE-015, folds in
    the ARS/AHRS velocity-blind fallback), so phases still show even on a
    dataset with no absolute position aiding, where ins never initializes
    and its own detector alone would stay false for the whole replay."""
    if not zupt or not any(zupt):
        return
    spans = _zupt_spans(t, zupt)
    for ax in fig.axes:
        for lo, hi in spans:
            ax.axvspan(lo, hi, color="gray", alpha=0.12, linewidth=0, zorder=0)


def _unwrap_deg(values):
    """Remove artificial +-360 deg jumps (display only) so a heading/yaw
    trace draws as a continuous line instead of tearing at +-180 deg.
    NaN gaps (no reference yet) are left as NaN and don't contaminate the
    unwrap once valid values resume."""
    out = list(values)
    prev_raw = prev_unwrapped = None
    for i, v in enumerate(out):
        if v != v:  # NaN
            continue
        prev_unwrapped = v if prev_raw is None else (
            prev_unwrapped + (((v - prev_raw + 180.0) % 360.0) - 180.0))
        out[i] = prev_unwrapped
        prev_raw = v
    return out


def _findings_page(findings, name):
    """Render the Dr. INS findings (list of (severity_int, text), 0=OK ..
    3=CRIT) as a standalone text page for the front of the PDF -- the same
    prioritized digest replay.py prints to the console, colored by
    severity. Returns None if there are no findings."""
    if not findings:
        return None
    import textwrap
    import matplotlib.pyplot as plt

    ordered = sorted(findings, key=lambda x: -x[0])
    rows = []            # (severity, [wrapped line, ...])
    n_lines = 0
    for sev, text in ordered:
        lines = textwrap.wrap(text, 96) or [""]
        rows.append((sev, lines))
        n_lines += len(lines)

    n_crit = sum(1 for s, _ in ordered if s == 3)
    n_warn = sum(1 for s, _ in ordered if s == 2)
    n_info = sum(1 for s, _ in ordered if s == 1)
    counts = ", ".join(p for p in (
        f"{n_crit} critical" if n_crit else "",
        (f"{n_warn} warning" + ("s" if n_warn != 1 else "")) if n_warn else "",
        f"{n_info} info" if n_info else "") if p) or "all nominal"

    # One slot per printed line plus one blank line between findings.
    slots = n_lines + len(rows) + 1
    fig = plt.figure(figsize=(14, max(2.5, 0.9 + 0.28 * slots)))
    fig.suptitle(f"{name}: Dr. INS findings ({counts})")
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.9])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    step = 1.0 / slots
    y = 1.0 - step
    for sev, lines in rows:
        ax.text(0.0, y, f"[{_FINDING_TAG[sev]}]", color=_FINDING_COLOR[sev],
                fontweight="bold", family="monospace", fontsize=11, va="top")
        for k, line in enumerate(lines):
            ax.text(0.075, y, line, color="0.1", fontsize=11, va="top")
            y -= step
        y -= step  # blank line between findings
    return fig


def plot_results(rec, name, warmup_sec, out_path=None, gnss_delay_curve=None,
                 sensor_rate=None, findings=None, configured_gnss_delay_ms=None,
                 growth_rate=None, baro_growth_rate=None, ahrs_growth_rate=None):
    """``rec`` is the dict of parallel per-sample lists built by
    replay.py's main() loop (keys: t, pos, pos_sigma, vel, vel_sigma,
    rpy_deg, rpy_sigma_deg, ref_pos, ref_vel, ref_rpy_deg, pos_err_ned,
    rpy_err_deg, baro_vel_d, gnss_vel_d, zupt_active, dw_*; ref_*/*_err_*
    rows are NaN where no reference was available yet). zupt_active
    (ins's own velocity-aware auto-ZUPT/ZARU detector, 1.0/0.0) is
    shaded as a gray band on every time-axis page (state history, error,
    bias, ARS/AHRS, baro_alt, altitude, outlier) so stops can be visually
    correlated with bias jumps/state changes -- not on the GNSS-delay or
    sensor-rate pages, whose x-axis isn't a plain replay timeline (lag,
    resp. rate buckets that may not cover the whole trial). ``gnss_delay_
    curve`` is the optional list of (lag_ms, correlation) from replay.
    gnss_delay_correlation_curve(); when given, an extra
    page shows the overview, a zoomed-in window and the correlation-vs-
    lag curve. ``sensor_rate`` is the optional dict of
    stream -> (t_sec, hz) from replay.py's _imu_prepass()/_bucketed_rate()
    (keys among imu/gnss/mag/baro, imu always present when given); when
    given, an extra page shows each stream's sampling rate over time.
    ``findings`` is the optional list of (severity_int, text) from
    replay.insdoctor_findings(); when given, the Dr. INS health-check
    digest is rendered as the FRONT page of the output (a cover summary
    before the detail pages). ``growth_rate``/``baro_growth_rate``/
    ``ahrs_growth_rate`` are the optional dicts from replay.
    process_noise_growth_rate()/baro_alt_growth_rate()/ahrs_growth_rate();
    when given, the front page's summary text adds their process-noise
    growth-rate blocks. Shows the figures interactively, or -- if
    ``out_path`` is given -- writes them as pages of one multi-page PDF at
    that path."""
    import matplotlib
    if out_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = PLOT_FONT_FAMILY

    t = rec["t"]
    if not t:
        print("plot: no recorded samples (filter never initialized?)")
        return
    # A single reference epoch (e.g. a placeholder ref.csv) is forward-
    # filled onto every later sample by replay.py/inspostgui.py, which
    # would otherwise look like a real, flat "ground truth" trajectory
    # here. rec["ref_epoch_count"] is the real, pre-forward-fill count;
    # treat it as usable ground truth only from 2 distinct epochs on.
    has_ref = (rec.get("ref_epoch_count", 2) >= 2
              and any(not math.isnan(row[0]) for row in rec["ref_pos"]))

    # --- Figure 1: state history, 1-σ band, reference overlay ----------
    fig1, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)
    fig1.suptitle(f"{name}: state history (1-σ band)")
    # The vertical channel is stored NED (D, down-positive) but plotted as
    # Up (-D) for readability; the σ band is a half-width, so negating
    # the center/reference is enough.
    # Position at the reference point (score.leverarm_frd, REQ-VER-037), so
    # a reference taken at an antenna does not show the lever arm as error.
    pos_key = "pos_ref_pt" if rec.get("pos_ref_pt") else "pos"
    groups = (
        (pos_key, "pos_sigma", "ref_pos", ("N [m]", "E [m]", "Up [m]")),
        ("vel", "vel_sigma", "ref_vel", ("vN [m/s]", "vE [m/s]", "vUp [m/s]")),
        ("rpy_deg", "rpy_sigma_deg", "ref_rpy_deg",
         ("roll [°]", "pitch [°]", "yaw [°]")),
    )
    for row, (key, skey, rkey, labels) in enumerate(groups):
        is_yaw_row = key == "rpy_deg"
        is_vert_ned = key in (pos_key, "vel")  # D component shown as Up (-D)
        for i in range(3):
            ax = axes[row][i]
            center = _col(rec[key], i)
            if is_yaw_row and i == 2:
                center = _unwrap_deg(center)
            elif is_vert_ned and i == 2:
                center = [-v for v in center]
            _band(ax, t, center, _col(rec[skey], i), "estimate", EST_COLOR)
            if has_ref:
                ref_col = _col(rec[rkey], i)
                if is_yaw_row and i == 2:
                    ref_col = _unwrap_deg(ref_col)
                elif is_vert_ned and i == 2:
                    ref_col = [-v for v in ref_col]
                ref_col, ref_t = _thin(ref_col, t)
                ax.plot(ref_t, ref_col, "--", color=REF_COLOR,
                       linewidth=1.5, label="ground truth")
            ax.set_ylabel(labels[i])
            if row == 0 and i == 0:
                ax.legend(loc="best", fontsize=8)
    for i in range(3):
        axes[-1][i].set_xlabel("time [s]")
    fig1.tight_layout()
    _apply_grid(fig1)
    _shade_zupt(fig1, t, rec.get("zupt_active"))
    figs = [fig1]

    # --- Figure 1b: North-East position (map view) --------------------------
    fig_ne = _ne_page(rec, name)
    if fig_ne is not None:
        figs.append(fig_ne)

    # --- Figure 2: error vs. ground truth, 1-σ consistency band --------
    if has_ref:
        fig2, axes2 = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
        fig2.suptitle(f"{name}: error vs. ground truth (1-σ band)")
        for i, lbl in enumerate(("N [m]", "E [m]", "Up [m]")):
            ax = axes2[0][i]
            err = _col(rec["pos_err_ned"], i)
            if i == 2:  # D error shown as Up error (-D), matching Figure 1
                err = [-v for v in err]
            sig = _col(rec["pos_sigma"], i)
            _band(ax, t, err, sig, "error", EST_COLOR)
            ax.axhline(0.0, color="black", linewidth=0.5)
            ax.set_ylabel(f"pos err {lbl}")
            # Same reasoning as _cap_sigma_axis(): the σ band grows to
            # hundreds of metres across a long dead-reckoning stretch and
            # would otherwise flatten the metre-level error everywhere else.
            _cap_symmetric_axis(ax, err, sig, _SIGMA_AXIS_CAP_M, "m")
        for i, lbl in enumerate(("roll [°]", "pitch [°]", "yaw [°]")):
            ax = axes2[1][i]
            _band(ax, t, _col(rec["rpy_err_deg"], i),
                 _col(rec["rpy_sigma_deg"], i), "error", EST_COLOR)
            ax.axhline(0.0, color="black", linewidth=0.5)
            ax.set_ylabel(f"att err {lbl}")
            ax.set_xlabel("time [s]")
        if warmup_sec > 0:
            for ax in axes2.flat:
                ax.axvline(warmup_sec, color="gray", linestyle=":", linewidth=1)
        fig2.tight_layout()
        _apply_grid(fig2)
        _shade_zupt(fig2, t, rec.get("zupt_active"))
        figs.append(fig2)

    # --- Figure 3: ins IMU bias estimate (1-σ band) -------------------
    fig3 = _bias_page(rec, t, name)
    if fig3 is not None:
        _shade_zupt(fig3, t, rec.get("zupt_active"))
        figs.append(fig3)

    # --- Figure 3r: the RTS smoother's solution, when it ran ----------------
    for fig_rts, timeline in _rts_pages(rec, t, name, has_ref, warmup_sec):
        if timeline:
            _shade_zupt(fig_rts, t, rec.get("zupt_active"))
        figs.append(fig_rts)

    # --- Figure 3a: magnetometer coverage / hard-iron sphere ----------------
    fig_mag = _mag_sphere_page(rec, name)
    if fig_mag is not None:
        figs.append(fig_mag)

    # --- Figure 3b: magnetometer field magnitude vs. WMM reference ----------
    fig_wmm = _wmm_page(rec, name)
    if fig_wmm is not None:
        figs.append(fig_wmm)

    # --- Figure 4/5: ARS / AHRS (roll, pitch, own gyro bias) ----------------
    for prefix, label in (("ars", "ARS"), ("ahrs", "AHRS (magnetometer)")):
        fig_sub = _subfilter_page(rec, t, name, prefix, label)
        if fig_sub is not None:
            _shade_zupt(fig_sub, t, rec.get("zupt_active"))
            figs.append(fig_sub)

    # --- Figure 4c: attitude error comparison + RMS/max statistics ----------
    fig_att = _attitude_compare_page(rec, t, name, warmup_sec)
    if fig_att is not None:
        _shade_zupt(fig_att, t, rec.get("zupt_active"))
        figs.append(fig_att)

    # --- Figure 5a: heading source comparison -------------------------------
    fig_hdg = _heading_page(rec, t, name)
    if fig_hdg is not None:
        _shade_zupt(fig_hdg, t, rec.get("zupt_active"))
        figs.append(fig_hdg)

    # --- Figure 5b: dedicated altitude profile (alt_plot_a, local frame) ----
    fig_alt = _altitude_page(rec, t, name)
    if fig_alt is not None:
        _shade_zupt(fig_alt, t, rec.get("zupt_active"))
        figs.append(fig_alt)

    # --- Figure 5c: same, in the ellipsoid frame (alt_plot_b) ---------------
    fig_alt_ell = _altitude_ellipsoid_page(rec, t, name)
    if fig_alt_ell is not None:
        _shade_zupt(fig_alt_ell, t, rec.get("zupt_active"))
        figs.append(fig_alt_ell)

    # --- Figure 6: baro_alt vertical channel --------------------------------
    fig_baro = _baro_page(rec, t, name)
    if fig_baro is not None:
        _shade_zupt(fig_baro, t, rec.get("zupt_active"))
        figs.append(fig_baro)

    # --- Figure 7: GNSS-delay cross-correlation -----------------------------
    fig_delay = _gnss_delay_page(rec, t, name, gnss_delay_curve,
                                 configured_gnss_delay_ms)
    if fig_delay is not None:
        figs.append(fig_delay)

    # --- Figure 7b: GNSS reported stddev over time --------------------------
    fig_gnss_sd = _gnss_stddev_page(rec, t, name)
    if fig_gnss_sd is not None:
        figs.append(fig_gnss_sd)

    # --- Figure 7c: GNSS fix quality and transport latency ------------------
    fig_gnss_q = _gnss_quality_page(rec, name, configured_gnss_delay_ms)
    if fig_gnss_q is not None:
        figs.append(fig_gnss_q)

    # --- Figure 8: outlier rejection (chi2 downweight) over time ------------
    fig_outlier = _outlier_page(rec, t, name)
    if fig_outlier is not None:
        _shade_zupt(fig_outlier, t, rec.get("zupt_active"))
        figs.append(fig_outlier)

    # --- Figure 8b: ZUPT/ZARU detector timeline ------------------------------
    fig_zupt = _zupt_page(rec, t, name)
    if fig_zupt is not None:
        figs.append(fig_zupt)

    # --- Figure 9: sensor sampling rate over time ---------------------------
    fig_rate = _sensor_rate_page(sensor_rate, name)
    if fig_rate is not None:
        figs.append(fig_rate)

    # --- Front page: trajectory summary + Dr. INS findings digest -----------
    fig_summary = _summary_page(rec, name, growth_rate, baro_growth_rate,
                                ahrs_growth_rate, warmup_sec)
    if fig_summary is not None:
        figs.insert(0, fig_summary)

    fig_find = _findings_page(findings, name)
    if fig_find is not None:
        figs.insert(0, fig_find)

    if out_path:
        _save_pdf(figs, out_path)
        plt.close("all")
        return

    # No --plot-out: try to show interactively, but some environments
    # (e.g. WSL2 without a working GUI backend) silently resolve to Agg,
    # where plt.show() is a no-op that just warns "cannot be shown" --
    # confusing since nothing else indicates the plot never appeared.
    # Detect that specific warning and fall back to saving instead of
    # leaving the user with neither a window nor a file.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plt.show()
    if any("cannot be shown" in str(w.message) for w in caught):
        fallback_path = os.path.join(os.getcwd(), "ins_plots.pdf")
        print(f"plot: no interactive matplotlib backend available "
             f"(backend: {matplotlib.get_backend()}) -- saving to "
             f"{fallback_path} instead (pass --plot-out to choose a path)")
        _save_pdf(figs, fallback_path)
        plt.close("all")


def _has_rts(rec):
    """Whether rec carries the RTS smoother's series (ins_rts.add_rec_series)
    with at least one smoothed sample."""
    rows = rec.get("rts_pos")
    return bool(rows) and len(rows) == len(rec["t"]) and any(
        row[0] == row[0] for row in rows)


def _line(ax, t, values, **kw):
    values, t = _thin(values, t)
    ax.plot(t, values, **kw)


def _rts_pages(rec, t, name, has_ref, warmup_sec):
    """The RTS smoother's pages (tools/ins_rts.py, replay.py --smooth, the
    GUI's RTS smoother): trajectory with the accuracy table, state
    history, error vs. ground truth and IMU biases, each with the
    smoother's own 1-σ band and the filter as a thin line for comparison.
    (figure, has a time axis) pairs, empty when the smoother did not run."""
    if not _has_rts(rec):
        return []
    figs = [(_rts_trajectory_page(rec, name, has_ref), False),
            (_rts_state_page(rec, t, name, has_ref), True)]
    if has_ref:
        figs.append((_rts_error_page(rec, t, name, warmup_sec), True))
    fig_b = _rts_bias_page(rec, t, name)
    if fig_b is not None:
        figs.append((fig_b, True))
    return figs


def _rts_trajectory_page(rec, name, has_ref):
    """North-East track of the filter, the smoother and the reference,
    with the filter-vs-RTS accuracy table (rec["rts_lines"]) next to it."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"{name}: RTS smoother, North-East position")
    gs = fig.add_gridspec(1, 2, width_ratios=(2.2, 1.0))
    ax = fig.add_subplot(gs[0, 0])
    track = rec.get("track_ne") or rec.get("pos_ref_pt") or rec["pos"]
    n, e = _thin_track(_col(track, 0), _col(track, 1))
    ax.plot(e, n, color=RTS_FILTER_COLOR, linewidth=0.8, label="filter")
    rts_track = rec.get("rts_track_ne") or rec["rts_pos_ref_pt"]
    n, e = _thin_track(_col(rts_track, 0), _col(rts_track, 1))
    ax.plot(e, n, color=RTS_COLOR, linewidth=1.4, label="RTS smoother")
    if has_ref:
        ref_track = rec.get("track_ref_ne") or rec["ref_pos"]
        rn, re = _thin_track(_col(ref_track, 0), _col(ref_track, 1))
        ax.plot(re, rn, "--", color=REF_COLOR, linewidth=1.0,
                label="ground truth")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8)
    tx = fig.add_subplot(gs[0, 1])
    tx.axis("off")
    lines = rec.get("rts_lines") or []
    tx.text(0.0, 1.0, "\n".join(lines), transform=tx.transAxes, va="top",
            ha="left", family="monospace", fontsize=8)
    fig.tight_layout()
    _apply_grid(fig)
    return fig


def _rts_state_page(rec, t, name, has_ref):
    """Like the state-history page, for the smoother: its 1-σ band, the
    filter as a thin line, the reference dashed."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)
    fig.suptitle(f"{name}: RTS smoother, state history (1-σ band)")
    f_pos = "pos_ref_pt" if rec.get("pos_ref_pt") else "pos"
    groups = (
        ("rts_pos_ref_pt", "rts_pos_sigma", f_pos, "ref_pos",
         ("N [m]", "E [m]", "Up [m]")),
        ("rts_vel", "rts_vel_sigma", "vel", "ref_vel",
         ("vN [m/s]", "vE [m/s]", "vUp [m/s]")),
        ("rts_rpy_deg", "rts_rpy_sigma_deg", "rpy_deg", "ref_rpy_deg",
         ("roll [°]", "pitch [°]", "yaw [°]")),
    )
    for row, (key, skey, fkey, rkey, labels) in enumerate(groups):
        for i in range(3):
            ax = axes[row][i]

            def shown(values, row=row, i=i):
                if row == 2 and i == 2:
                    return _unwrap_deg(values)
                if row < 2 and i == 2:
                    return [-v for v in values]  # D shown as Up
                return values
            _band(ax, t, shown(_col(rec[key], i)), _col(rec[skey], i),
                  "RTS", RTS_COLOR)
            _line(ax, t, shown(_col(rec[fkey], i)), color=RTS_FILTER_COLOR,
                  linewidth=0.7, label="filter")
            if has_ref:
                _line(ax, t, shown(_col(rec[rkey], i)), linestyle="--",
                      color=REF_COLOR, linewidth=1.0, label="ground truth")
            ax.set_ylabel(labels[i])
            if row == 0 and i == 0:
                ax.legend(loc="best", fontsize=8)
    for i in range(3):
        axes[-1][i].set_xlabel("time [s]")
    fig.tight_layout()
    _apply_grid(fig)
    return fig


def _rts_error_page(rec, t, name, warmup_sec):
    """The smoother's error vs. the ground truth with its own 1-σ band,
    the filter's error as a thin line."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    fig.suptitle(f"{name}: RTS smoother, error vs. ground truth (1-σ band)")
    for i, lbl in enumerate(("N [m]", "E [m]", "Up [m]")):
        ax = axes[0][i]
        sign = -1.0 if i == 2 else 1.0  # D error shown as Up error
        err = [sign * v for v in _col(rec["rts_pos_err_ned"], i)]
        sig = _col(rec["rts_pos_sigma"], i)
        _band(ax, t, err, sig, "RTS", RTS_COLOR)
        _line(ax, t, [sign * v for v in _col(rec["pos_err_ned"], i)],
              color=RTS_FILTER_COLOR, linewidth=0.7, label="filter")
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_ylabel(f"pos err {lbl}")
        _cap_symmetric_axis(ax, err, sig, _SIGMA_AXIS_CAP_M, "m")
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    for i, lbl in enumerate(("roll [°]", "pitch [°]", "yaw [°]")):
        ax = axes[1][i]
        err = _col(rec["rts_rpy_err_deg"], i)
        sig = _col(rec["rts_rpy_sigma_deg"], i)
        _band(ax, t, err, sig, "RTS", RTS_COLOR)
        _line(ax, t, _col(rec["rpy_err_deg"], i), color=RTS_FILTER_COLOR,
              linewidth=0.7, label="filter")
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_ylabel(f"att err {lbl}")
        ax.set_xlabel("time [s]")
    if warmup_sec > 0:
        for ax in axes.flat:
            ax.axvline(warmup_sec, color="gray", linestyle=":", linewidth=1)
    fig.tight_layout()
    _apply_grid(fig)
    return fig


def _rts_bias_page(rec, t, name):
    """The smoother's IMU (and, 18-state, magnetometer) bias with its 1-σ
    band, the filter's estimate as a thin line."""
    if not any(row[0] == row[0] for row in rec["rts_acc_bias"]):
        return None
    has_mag = any(row[0] == row[0] for row in rec.get("rts_mag_bias", []))
    import matplotlib.pyplot as plt
    n_rows = 3 if has_mag else 2
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 3.5 * n_rows), sharex=True)
    fig.suptitle(f"{name}: RTS smoother, IMU bias (1-σ band)")
    rows = [("rts_acc_bias", "rts_acc_bias_sigma", "acc_bias", 1.0,
             "acc bias {} [m/s²]"),
            ("rts_gyr_bias", "rts_gyr_bias_sigma", "gyr_bias", 180.0 / math.pi,
             "gyro bias {} [°/s]")]
    if has_mag:
        rows.append(("rts_mag_bias", "rts_mag_bias_sigma", "mag_bias", 1.0,
                     "mag bias {} [µT]"))
    for r, (key, skey, fkey, scale, label) in enumerate(rows):
        for i, axis_lbl in enumerate(("x", "y", "z")):
            ax = axes[r][i]
            _band(ax, t, [scale * v for v in _col(rec[key], i)],
                  [scale * v for v in _col(rec[skey], i)], "RTS", RTS_COLOR)
            _line(ax, t, [scale * v for v in _col(rec[fkey], i)],
                  color=RTS_FILTER_COLOR, linewidth=0.7, label="filter")
            ax.set_ylabel(label.format(axis_lbl))
            if r == 0 and i == 0:
                ax.legend(loc="best", fontsize=8)
    for i in range(3):
        axes[-1][i].set_xlabel("time [s]")
    fig.tight_layout()
    _apply_grid(fig)
    return fig


def _bias_page(rec, t, name):
    """ins's own accelerometer/gyro bias estimate with a 1-σ band,
    plus (18-state mode only) the magnetometer hard-iron
    bias with its own 1-σ band as a third row. Returns None if
    ins was never ready (all-NaN)."""
    if not any(row[0] == row[0] for row in rec["acc_bias"]):
        return None
    has_mag_bias = any(row[0] == row[0] for row in rec["mag_bias"])
    import matplotlib.pyplot as plt

    n_rows = 3 if has_mag_bias else 2
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 3.5 * n_rows))
    fig.suptitle(f"{name}: ins IMU bias estimate (1-σ band)")
    for i, axis_lbl in enumerate(("x", "y", "z")):
        ax = axes[0][i]
        _band(ax, t, _col(rec["acc_bias"], i), _col(rec["acc_bias_sigma"], i),
             "estimate", EST_COLOR)
        ax.set_ylabel(f"acc bias {axis_lbl} [m/s²]")
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    for i, axis_lbl in enumerate(("x", "y", "z")):
        ax = axes[1][i]
        center = [math.degrees(v) for v in _col(rec["gyr_bias"], i)]
        sigma = [math.degrees(v) for v in _col(rec["gyr_bias_sigma"], i)]
        _band(ax, t, center, sigma, "estimate", EST_COLOR)
        ax.set_ylabel(f"gyro bias {axis_lbl} [°/s]")
        if not has_mag_bias:
            ax.set_xlabel("time [s]")
    if has_mag_bias:
        for i, axis_lbl in enumerate(("x", "y", "z")):
            ax = axes[2][i]
            _band(ax, t, _col(rec["mag_bias"], i), _col(rec["mag_bias_sigma"], i),
                 "estimate", EST_COLOR)
            ax.set_ylabel(f"mag bias {axis_lbl} [µT]")
            ax.set_xlabel("time [s]")
    fig.tight_layout()
    _apply_grid(fig)
    return fig


def _mag_sphere_page(rec, name):
    """3D scatter of every raw magnetometer sample [µT], with the origin and
    the hard-iron offset marked. A clean sensor traces a sphere centred on the
    origin; a hard-iron bias shifts that centre, so the offset origin->centre
    is the hard-iron vector. The spread along each body axis shows how well
    that axis was exercised -- a flat/degenerate cloud means the corresponding
    bias state is poorly observable. An algebraic (Kasa) sphere fit gives the
    empirical centre independent of the filter; the filter's own estimate
    (18-state mag_bias, or a configured fixed_bias) is overlaid when present.
    Returns None if there are no usable mag samples."""
    pts = rec.get("mag_raw")
    if not pts:
        return None
    import numpy as np
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d  # noqa: F401  (registers the 3d projection)

    P = np.asarray(pts, dtype=float)
    P = P[np.isfinite(P).all(axis=1)]
    if len(P) < 4:
        return None

    # Algebraic sphere fit: minimise |2 c.x - (|x|^2 - k)| -> centre c, radius.
    A = np.hstack([2.0 * P, np.ones((len(P), 1))])
    rhs = (P ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    centre = sol[:3]
    radius = math.sqrt(max(sol[3] + float(centre @ centre), 0.0))

    Pd = P if len(P) <= _MAX_PLOT_POINTS else \
        P[np.linspace(0, len(P) - 1, _MAX_PLOT_POINTS).astype(int)]

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle(f"{name}: magnetometer coverage / hard-iron sphere [µT]")

    # Colour by signed distance to the fitted shell so an off-sphere smear
    # (soft-iron / disturbance) stands out from a clean shell: green exactly
    # on the shell (0), red outside (+), blue inside (-). A symmetric range
    # keeps 0 pinned to green.
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    shell = np.linalg.norm(Pd - centre, axis=1) - radius
    shell_cmap = LinearSegmentedColormap.from_list(
        "mag_shell", ["tab:blue", "tab:green", "tab:red"])
    m = float(np.abs(shell).max()) or 1.0
    sc = ax.scatter(Pd[:, 0], Pd[:, 1], Pd[:, 2], s=4, c=shell,
                    cmap=shell_cmap, norm=Normalize(-m, m),
                    alpha=0.5, linewidths=0)
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1,
                 label="distance to fitted shell [µT] (green = on shell)")

    ax.scatter([0], [0], [0], color="k", marker="+", s=140, label="origin")
    ax.scatter(*centre, color=EST_COLOR, marker="o", s=60,
               label=(f"sphere-fit centre ({centre[0]:.1f}, {centre[1]:.1f}, "
                      f"{centre[2]:.1f}), r={radius:.1f}"))
    ax.plot([0, centre[0]], [0, centre[1]], [0, centre[2]],
            color=EST_COLOR, lw=1.5)

    # Overlay the filter's own hard-iron offset when available: the last
    # non-NaN 18-state estimate, else a configured fixed_bias.
    est = None
    mb = [row for row in rec.get("mag_bias", []) if row and row[0] == row[0]]
    if mb:
        est = (tuple(mb[-1]), "18-state estimate")
    elif rec.get("mag_fixed_bias"):
        est = (tuple(rec["mag_fixed_bias"]), "fixed_bias")
    if est is not None:
        (ex, ey, ez), lbl = est
        ax.scatter([ex], [ey], [ez], color="tab:green", marker="D", s=45,
                   label=f"{lbl} ({ex:.1f}, {ey:.1f}, {ez:.1f})")

    # Equal aspect so a sphere looks like a sphere: one cube around the data.
    lo = Pd.min(axis=0)
    hi = Pd.max(axis=0)
    span = max((hi - lo).max(), 1e-6) / 2.0
    mid = (hi + lo) / 2.0
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    # FRD z is down-positive; flip the axis so "down" points down on screen.
    ax.set_zlim(mid[2] + span, mid[2] - span)
    try:
        ax.set_box_aspect((1, 1, 1))
    except (AttributeError, TypeError):
        pass  # older matplotlib: aspect stays default

    ax.set_xlabel("mag x (fwd) [µT]")
    ax.set_ylabel("mag y (right) [µT]")
    ax.set_zlabel("mag z (down) [µT]")
    ax.legend(loc="upper left", fontsize=8)
    return fig


def _rms(values):
    """RMS of a list, or None when it is empty."""
    if not values:
        return None
    return math.sqrt(sum(v * v for v in values) / len(values))


def _mag_field_minus_bias(rec):
    """|B| of the calibrated samples with the 18-state hard-iron estimate
    (as it was at each sample) removed, or None without that estimate."""
    xyz = rec.get("mag_field_xyz")
    tf = rec.get("mag_field_t")
    bias = rec.get("mag_bias")
    tr = rec.get("t")
    if not xyz or not tf or not bias or not tr or len(xyz) != len(tf):
        return None
    ok = [i for i, row in enumerate(bias) if row and row[0] == row[0]]
    if not ok:
        return None
    import numpy as np
    t_ok = np.asarray([tr[i] for i in ok], dtype=float)
    b = np.column_stack([
        np.interp(np.asarray(tf, dtype=float), t_ok,
                  np.asarray([bias[i][k] for i in ok], dtype=float))
        for k in range(3)])
    d = np.asarray(xyz, dtype=float) - b
    return [float(v) for v in np.linalg.norm(d, axis=1)]


def _wmm_page(rec, name):
    """Calibrated magnetic field magnitude |B|(t) [µT] against the WMM total
    field F (horizontal reference), plus the deviation |B|-F below. The
    calibrated trace is what the filter actually fuses: the config.yaml fixed
    calibration (mag: misalignment/fixed_bias) has been applied, the same way
    ins.c applies it internally. When such a calibration is configured the
    uncalibrated magnitude is drawn alongside it, so what the calibration
    bought is visible. A clean, well-calibrated magnetometer hugs F;
    hard/soft-iron error, nearby ferrous mass or electrical disturbances show
    up as departures from it. Returns None if there are no mag samples or no
    WMM reference (mag: wmm_year set)."""
    mag = rec.get("mag_field_mag")
    tf = rec.get("mag_field_t")
    F = rec.get("wmm_field_uT")
    if not mag or not tf or F is None:
        return None
    raw = rec.get("mag_field_mag_raw") or None
    if raw is not None and len(raw) != len(mag):
        raw = None
    est = _mag_field_minus_bias(rec)
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"{name}: magnetometer field vs. WMM reference")

    # One decimation for both traces (driven by the calibrated one) so the
    # comparison stays sample-aligned.
    if raw is None:
        mag_v, mag_t = _thin(mag, tf)
        raw_v = None
    else:
        mag_v, raw_v, mag_t = _thin(mag, raw, tf)
    if est is not None:
        est_v, est_t = _thin(est, tf)

    if raw_v is not None:
        ax0.plot(mag_t, raw_v, color=MAG_RAW_COLOR, linewidth=1.0,
                 label="uncalibrated |B|")
    cal_word = "calibrated" if raw is not None else "measured"
    ax0.plot(mag_t, mag_v, color=EST_COLOR, linewidth=1.0,
             label=f"{cal_word} |B|")
    if est is not None:
        ax0.plot(est_t, est_v, color="tab:green", linewidth=1.0,
                 label="|B| minus estimated bias (18-state)")
    ax0.axhline(F, color=REF_COLOR, linestyle="--", linewidth=1.5,
                label=f"WMM |B| = {F:.1f} µT")
    ax0.set_ylabel("field magnitude [µT]")
    ax0.legend(loc="best", fontsize=8)

    # Deviation from the reference, with its RMS in the legend: the single
    # number that says whether the calibration moved the field onto F.
    dev = [v - F for v in mag]
    if raw is None:
        dev_v, dev_t = _thin(dev, tf)
    else:
        dev_raw = [v - F for v in raw]
        dev_v, dev_raw_v, dev_t = _thin(dev, dev_raw, tf)
        ax1.plot(dev_t, dev_raw_v, color=MAG_RAW_COLOR, linewidth=1.0,
                 label=f"uncalibrated, RMS {_rms(dev_raw):.2f} µT")
    ax1.plot(dev_t, dev_v, color=EST_COLOR, linewidth=1.0,
             label=f"{cal_word}, RMS {_rms(dev):.2f} µT")
    if est is not None:
        dev_est = [v - F for v in est]
        de_v, de_t = _thin(dev_est, tf)
        ax1.plot(de_t, de_v, color="tab:green", linewidth=1.0,
                 label=f"minus estimated bias, RMS {_rms(dev_est):.2f} µT")
    ax1.axhline(0.0, color="black", linewidth=0.5)
    ax1.set_ylabel("|B| - WMM [µT]")
    ax1.set_xlabel("time [s]")
    ax1.legend(loc="best", fontsize=8)

    fig.tight_layout()
    _apply_grid(fig)
    return fig


def _pos_err_lines(rec, warmup_sec):
    """Summary lines with the position error vs the reference (north, east,
    up, horizontal, 3D) as RMS / max |e| from the end of the warm-up on.
    Empty when there is no usable reference."""
    t = rec["t"]
    errs = rec.get("pos_err_ned") or []
    rows = []
    for ti, e in zip(t, errs):
        if ti >= warmup_sec and e and all(v == v for v in e[:3]):
            rows.append(e)
    if not rows:
        return []
    cols = {
        "north": [e[0] for e in rows],
        "east": [e[1] for e in rows],
        "up": [-e[2] for e in rows],
        "horizontal": [math.hypot(e[0], e[1]) for e in rows],
        "3D": [math.sqrt(e[0] ** 2 + e[1] ** 2 + e[2] ** 2) for e in rows],
    }
    lines = ["", "position error vs ground truth (RMS / max |e| in m):"]
    for key, vals in cols.items():
        rms = math.sqrt(sum(v * v for v in vals) / len(vals))
        lines.append(f"  {key:11s}{rms:7.3f} / "
                     f"{max(abs(v) for v in vals):7.3f}")
    return lines


def _summary_page(rec, name, growth_rate=None, baro_growth_rate=None,
                  ahrs_growth_rate=None, warmup_sec=0.0):
    """Trajectory summary: total time/distance and the velocity/acceleration
    envelope, taken from ground truth when available (rec['ref_*'], else the
    filter estimate). Kinematic acceleration = |d v/dt| is gravity-free by
    construction. Rendered as a text page for the front of the PDF. Returns
    None if there is no usable position/velocity history. ``growth_rate``/
    ``baro_growth_rate``/``ahrs_growth_rate`` are the optional dicts from
    replay.process_noise_growth_rate()/baro_alt_growth_rate()/
    ahrs_growth_rate()."""
    t = rec["t"]
    if not t or len(t) < 2:
        return None
    # See has_ref in plot_results() above: a single forward-filled
    # reference epoch (e.g. a placeholder ref.csv) must not be scored as a
    # flat, zero-speed ground-truth trajectory.
    use_ref = (rec.get("ref_epoch_count", 2) >= 2
              and any(row and row[0] == row[0] for row in rec["ref_vel"]))
    vel = rec["ref_vel"] if use_ref else rec["vel"]
    pos = rec["ref_pos"] if use_ref else rec["pos"]
    src = "ground truth" if use_ref else "filter estimate"

    # Keep only samples where BOTH velocity and position are finite, with
    # their timestamps, so the derivatives/integrals below stay aligned.
    keep = [i for i in range(len(t))
            if vel[i] and vel[i][0] == vel[i][0]
            and pos[i] and pos[i][0] == pos[i][0]]
    if len(keep) < 2:
        return None

    total_time = t[keep[-1]] - t[keep[0]]
    speeds, horiz, up, down, accels, dist = [], [], [], [], [], 0.0
    for k, i in enumerate(keep):
        vn, ve, vd = vel[i]
        speeds.append(math.sqrt(vn * vn + ve * ve + vd * vd))
        horiz.append(math.hypot(vn, ve))
        up.append(-vd)
        down.append(vd)
        if k > 0:
            j = keep[k - 1]
            dt = t[i] - t[j]
            if dt > 1e-6:
                pvn, pve, pvd = vel[j]
                accels.append(math.sqrt((vn - pvn) ** 2 + (ve - pve) ** 2
                                        + (vd - pvd) ** 2) / dt)
            dp = [pos[i][c] - pos[j][c] for c in range(3)]
            dist += math.sqrt(sum(c * c for c in dp))

    avg_speed = sum(speeds) / len(speeds)
    max_horiz = max(horiz)
    max_climb = max(0.0, max(up))
    max_descent = max(0.0, max(down))
    max_acc = max(accels) if accels else 0.0

    def _kmh(mps):
        return mps * 3.6

    dist_str = (f"{dist / 1000.0:.2f} km" if dist >= 1000.0
                else f"{dist:.1f} m")
    hms = f"{int(total_time) // 3600:d}:{(int(total_time) % 3600) // 60:02d}:" \
          f"{int(total_time) % 60:02d}"
    lines = [
        f"total time            : {total_time:8.1f} s   ({hms})",
        f"total distance        : {dist_str:>12s}",
        f"average speed         : {avg_speed:8.2f} m/s ({_kmh(avg_speed):6.1f} km/h)",
        f"max horizontal speed  : {max_horiz:8.2f} m/s ({_kmh(max_horiz):6.1f} km/h)",
        f"max climb rate        : {max_climb:8.2f} m/s",
        f"max descent rate      : {max_descent:8.2f} m/s",
        f"max acceleration      : {max_acc:8.2f} m/s² ({max_acc / 9.80665:5.2f} g,"
        f" gravity-free)",
    ]

    # Final filter-reported 1-σ of the ins error state (last recorded
    # epoch), including biases -- the exact sqrt of the covariance diagonal,
    # i.e. how uncertain ins believes its own solution is at the end of the
    # run (not a vs-truth error). Mirrors replay.py's console "final 1-σ"
    # block; sourced from rec (no live Navigator at PDF-export time).
    p = _last_valid_row(rec.get("pos_sigma", []))
    v = _last_valid_row(rec.get("vel_sigma", []))
    a = _last_valid_row(rec.get("rpy_sigma_deg", []))
    ab = _last_valid_row(rec.get("acc_bias", []))
    ab_s = _last_valid_row(rec.get("acc_bias_sigma", []))
    gb = _last_valid_row(rec.get("gyr_bias", []))
    gb_s = _last_valid_row(rec.get("gyr_bias_sigma", []))
    if p and v and a and ab and ab_s and gb and gb_s:
        lines.append("")
        lines.append("final 1-σ (last epoch, filter-reported):")
        lines.append(f"  pos NED   {p[0]:.3f} {p[1]:.3f} {p[2]:.3f} m")
        lines.append(f"  vel NED   {v[0]:.3f} {v[1]:.3f} {v[2]:.3f} m/s")
        lines.append(f"  att RPY   {a[0]:.3f} {a[1]:.3f} {a[2]:.3f} °")
        lines.append(f"  acc bias  {ab[0]:.4f} {ab[1]:.4f} {ab[2]:.4f} m/s² "
                     f"(1-σ {ab_s[0]:.4f} {ab_s[1]:.4f} {ab_s[2]:.4f})")
        lines.append(f"  gyr bias  {math.degrees(gb[0]):.4f} "
                     f"{math.degrees(gb[1]):.4f} {math.degrees(gb[2]):.4f} "
                     f"°/s (1-σ {math.degrees(gb_s[0]):.4f} "
                     f"{math.degrees(gb_s[1]):.4f} {math.degrees(gb_s[2]):.4f})")
        mb = _last_valid_row(rec.get("mag_bias", []))
        if mb:
            mb_s = _last_valid_row(rec.get("mag_bias_sigma", []))
            sig = (f" (1-σ {mb_s[0]:.3f} {mb_s[1]:.3f} {mb_s[2]:.3f})"
                   if mb_s else "")
            lines.append(f"  mag bias  {mb[0]:.3f} {mb[1]:.3f} {mb[2]:.3f} µT{sig}")

    # Most recent fused GNSS fix's OWN reported 1-σ (receiver
    # covariance), side-by-side with the filter's final 1-σ above.
    gp = _last_valid_row(rec.get("gnss_pos_sigma", []))
    if gp:
        lines.append("last GNSS fix accuracy (receiver-reported 1-σ):")
        lines.append(f"  pos NED   {gp[0]:.3f} {gp[1]:.3f} {gp[2]:.3f} m")
        gv = _last_valid_row(rec.get("gnss_vel_sigma", []))
        if gv:
            lines.append(f"  vel NED   {gv[0]:.3f} {gv[1]:.3f} {gv[2]:.3f} m/s")

    # Position accuracy against the reference (RMS and worst case), same
    # warm-up rule as the attitude statistics below.
    lines.extend(_pos_err_lines(rec, warmup_sec))

    # Attitude accuracy of INSLIB / ARS / AHRS (RMS and worst case), see
    # _attitude_compare_page() for the traces behind these numbers.
    att_rows = _att_stat_rows(rec, warmup_sec)
    if att_rows:
        last_ref = None
        for ref_label, label, ax, s in att_rows:
            if ref_label != last_ref:
                lines.append("")
                lines.append(f"attitude error vs {ref_label} "
                             f"(RMS / max |e| in °):")
                last_ref = ref_label
            lines.append(f"  {label + ' ' + _ATT_AXIS_NAMES[ax]:14s}"
                         f"{s['rms']:7.3f} / {abs(s['max']):7.3f}")

    # Theoretical free-inertial (no aiding) growth rate from the configured
    # imu process noise -- see replay.process_noise_growth_rate() and its
    # baro_alt_growth_rate()/ahrs_growth_rate() counterparts. growth_rate is
    # None when the dataset leaves pos/vel/rpy_pred_stddev_*_sqrts at 0
    # (deferring to ins.c's own built-in default): replay.
    # process_noise_growth_rate() deliberately refuses to guess that value
    # rather than hold a second, driftable copy of a C #define.
    if growth_rate is not None:
        g = growth_rate
        lines.append(f"ins process noise growth rate (avg over "
                     f"{g['horizon_sec']:.0f} s free-inertial coasting, "
                     f"no aiding):")
        lines.append(f"  pos   {g['pos_mps']:.4f} m/s")
        lines.append(f"  vel   {g['vel_mps2']:.4f} m/s²")
        lines.append(f"  att   {math.degrees(g['att_radps']):.4f} °/s")
    else:
        lines.append("ins process noise growth rate: N/A - set "
                     "imu.pos_pred_stddev_m_sqrts / vel_pred_stddev_mps_sqrts "
                     "/ rpy_pred_stddev_rad_sqrts explicitly in config.yaml")
    if baro_growth_rate is not None:
        g = baro_growth_rate
        lines.append(f"baro_alt process noise growth rate (avg over "
                     f"{g['horizon_sec']:.0f} s free vertical coasting, "
                     f"no baro):")
        lines.append(f"  height    {g['h_mps']:.4f} m/s")
        lines.append(f"  vel (-D)  {g['v_mps2']:.4f} m/s²")
        lines.append(f"  acc bias  {g['bias_mps2_per_s']:.6f} m/s²/s  "
                     f"(bias_rw {g['bias_rw_mps2_sqrths']:.2e} "
                     f"m/s²/√Hz)")
    else:
        lines.append("baro_alt process noise growth rate: N/A - "
                     "set baro.acc_noise_mps2_sqrthz / acc_bias_rw "
                     "explicitly in config.yaml")
    if ahrs_growth_rate is not None:
        g = ahrs_growth_rate
        lines.append(f"ars/ahrs process noise growth rate (avg over "
                     f"{g['horizon_sec']:.0f} s free coasting):")
        lines.append(f"  att   {math.degrees(g['att_radps']):.4f} °/s")
    else:
        lines.append("ars/ahrs process noise growth rate: N/A - "
                     "set ahrs.gyr_noise_psd / gyr_bias_rw explicitly in "
                     "config.yaml")

    import matplotlib.pyplot as plt
    slots = len(lines) + 1
    fig = plt.figure(figsize=(14, max(2.5, 0.9 + 0.32 * slots)))
    fig.suptitle(f"{name}: trajectory summary (from {src})")
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.86])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    step = 1.0 / slots
    y = 1.0 - step
    for line in lines:
        ax.text(0.0, y, line, color="0.1", family="monospace",
                fontsize=13, va="top")
        y -= step
    return fig


def _subfilter_page(rec, t, name, prefix, label):
    """ARS/AHRS page: own roll/pitch (and, for the magnetometer-aided
    AHRS, yaw) + 1-σ band vs. full3d vs. ground truth, plus own gyro
    bias + 1-σ band vs. full3d's own bias estimate. The AHRS yaw is
    absolute (magnetometer); the gyro-only ARS yaw is free gyro
    compassing with an arbitrary start offset, so it is shown shifted onto
    the reference at the first common sample. Returns None if this
    sub-filter was never active."""
    rpy = rec[f"{prefix}_rpy_deg"]
    if not any(row[0] == row[0] for row in rpy):
        return None
    import matplotlib.pyplot as plt

    # A single reference epoch (e.g. a placeholder ref.csv) is forward-
    # filled onto every later sample by replay.py/inspostgui.py, which
    # would otherwise look like a real, flat "ground truth" trajectory
    # here. rec["ref_epoch_count"] is the real, pre-forward-fill count;
    # treat it as usable ground truth only from 2 distinct epochs on.
    has_ref = (rec.get("ref_epoch_count", 2) >= 2
              and any(not math.isnan(row[0]) for row in rec["ref_pos"]))

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(f"{name}: {label}")

    # roll/pitch always; yaw only for the magnetometer-aided AHRS (its
    # heading is absolute) -- the gyro-only ARS yaw is free-running.
    # The ARS yaw is a free gyro-compassing heading with an arbitrary start
    # offset: shifted onto the reference at the first common sample (like the
    # heading page) so only the relative drift remains visible.
    free_yaw = prefix == "ars"
    panels = [(0, "roll [°]"), (1, "pitch [°]"),
              (2, "free gyro heading [°]\n(synced at start)" if free_yaw
               else "yaw [°]")]
    rpy_sigma = rec[f"{prefix}_rpy_sigma_deg"]
    for i, lbl in panels:
        ax = axes[0][i]
        col = _unwrap_deg(_col(rpy, i)) if i == 2 else _col(rpy, i)
        f3d_col = _unwrap_deg(_col(rec["rpy_deg"], i)) if i == 2 else _col(rec["rpy_deg"], i)
        ref_raw = None
        if has_ref:
            ref_raw = (_unwrap_deg(_col(rec["ref_rpy_deg"], i)) if i == 2
                       else _col(rec["ref_rpy_deg"], i))
            if i == 2:
                # A reference without real attitude (position-only truth
                # padded with a constant yaw) is no heading reference:
                # syncing to it would offset the ARS from full3d by the
                # unknown true heading. Fall back to full3d for the sync.
                vals = [v for v in ref_raw if v == v]
                if not vals or max(vals) - min(vals) < 1.0:
                    ref_raw = None
        if i == 2 and free_yaw:
            sync = _sync_offset(ref_raw if ref_raw is not None else f3d_col, col)
            if sync is not None:
                col = [c + sync[0] if c == c else math.nan for c in col]
        _band(ax, t, col, _col(rpy_sigma, i), label, EST_COLOR)
        full3d, tf = _thin(f3d_col, t)
        ax.plot(tf, full3d, "--", color="0.4", linewidth=1.2, label="full3d")
        if ref_raw is not None:
            ref_col, tr = _thin(ref_raw, t)
            ax.plot(tr, ref_col, ":", color=REF_COLOR, linewidth=1.5,
                   label="ground truth")
        ax.set_ylabel(lbl)
        ax.set_xlabel("time [s]")
        if i == 0:
            ax.legend(loc="best", fontsize=8)

    bias_key = f"{prefix}_gyr_bias"
    sigma_key = f"{prefix}_gyr_bias_sigma"
    for i, axis_lbl in enumerate(("x", "y", "z")):
        ax = axes[1][i]
        center = [math.degrees(v) for v in _col(rec[bias_key], i)]
        sigma = [math.degrees(v) for v in _col(rec[sigma_key], i)]
        _band(ax, t, center, sigma, label, EST_COLOR)
        full3d_bias, tf = _thin([math.degrees(v) for v in _col(rec["gyr_bias"], i)], t)
        ax.plot(tf, full3d_bias, "--", color="0.4", linewidth=1.2, label="full3d")
        ax.set_ylabel(f"gyro bias {axis_lbl} [°/s]")
        ax.set_xlabel("time [s]")
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _apply_grid(fig)
    return fig


_ATT_AXIS_NAMES = ("roll", "pitch", "yaw")


def _wrap180(d):
    return (d + 180.0) % 360.0 - 180.0


def _att_ref_axes(rec):
    """Attitude axes (0 roll, 1 pitch, 2 yaw) for which rec holds a real
    ground-truth attitude. A position-only reference is padded with constant
    angles, so an axis only counts when the reference actually moves (yaw:
    1 deg, roll/pitch: 0.5 deg; roll/pitch also count when yaw is real)."""
    if rec.get("ref_epoch_count", 2) < 2:
        return ()
    rows = rec.get("ref_rpy_deg") or []
    cols = [[v for v in _col(rows, i) if v == v] for i in range(3)]
    if not all(cols):
        return ()
    cols[2] = [v for v in _unwrap_deg(cols[2]) if v == v]
    spread = [max(c) - min(c) for c in cols]
    yaw_ok = spread[2] >= 1.0
    rp_ok = yaw_ok or max(spread[0], spread[1]) >= 0.5
    return tuple(i for i, ok in ((0, rp_ok), (1, rp_ok), (2, yaw_ok)) if ok)


def _att_sources(rec):
    """(label, color, rpy rec key, axes) of every attitude filter that
    produced output. The gyro-only ARS has no absolute yaw, so no yaw axis."""
    out = []
    for label, ck, key, axes in (("INSLIB", "full3d", "rpy_deg", (0, 1, 2)),
                                 ("ARS", "ars", "ars_rpy_deg", (0, 1)),
                                 ("AHRS", "ahrs", "ahrs_rpy_deg", (0, 1, 2))):
        rows = rec.get(key)
        if rows and any(r[0] == r[0] for r in rows):
            out.append((label, DOWNWEIGHT_COLORS[ck], key, axes))
    return out


def _att_error(est, ref, axis):
    """Per-sample angle error est - ref in deg, NaN where either is NaN.
    Yaw is wrapped to +-180 deg."""
    out = []
    for e, r in zip(est, ref):
        if e == e and r == r:
            d = e - r
            out.append(_wrap180(d) if axis == 2 else d)
        else:
            out.append(math.nan)
    return out


def _att_error_series(rec):
    """Attitude error traces, ground truth first (if usable), then INSLIB as
    the reference. Returns a list of (ref_label, [(label, color, axis, err)])
    where err is parallel to rec['t']."""
    sources = _att_sources(rec)
    groups = []
    ref_axes = _att_ref_axes(rec)
    if ref_axes:
        traces = []
        for label, color, key, axes in sources:
            for ax in axes:
                if ax in ref_axes:
                    traces.append((label, color, ax, _att_error(
                        _col(rec[key], ax), _col(rec["ref_rpy_deg"], ax), ax)))
        if traces:
            groups.append(("ground truth", traces))
    if sources and sources[0][0] == "INSLIB":
        traces = []
        for label, color, key, axes in sources[1:]:
            for ax in axes:
                traces.append((label, color, ax, _att_error(
                    _col(rec[key], ax), _col(rec["rpy_deg"], ax), ax)))
        if traces:
            groups.append(("INSLIB", traces))
    return groups


def _att_stats(err, t, warmup_sec):
    """RMS, mean, max |error| (signed value and time) of the finite samples
    from warmup_sec on, or None."""
    vals = [(e, ti) for e, ti in zip(err, t) if e == e and ti >= warmup_sec]
    if not vals:
        return None
    n = len(vals)
    emax, tmax = max(vals, key=lambda p: abs(p[0]))
    return {"rms": math.sqrt(sum(e * e for e, _ in vals) / n),
            "mean": sum(e for e, _ in vals) / n,
            "max": emax, "tmax": tmax, "n": n}


def _att_stat_rows(rec, warmup_sec):
    """Flat list of (ref_label, label, axis, stats) for the attitude
    comparison table and the front page."""
    t = rec["t"]
    rows = []
    for ref_label, traces in _att_error_series(rec):
        for label, _color, ax, err in traces:
            s = _att_stats(err, t, warmup_sec)
            if s is not None:
                rows.append((ref_label, label, ax, s))
    return rows


def _att_table_lines(rows):
    lines = []
    last_ref = None
    for ref_label, label, ax, s in rows:
        if ref_label != last_ref:
            if last_ref is not None:
                lines.append("")
            lines.append(f"error vs {ref_label}:")
            lines.append(f"  {'':16s}{'RMS':>8s}{'mean':>9s}{'max |e|':>10s}"
                         f"{'at t [s]':>10s}   [°]")
            last_ref = ref_label
            best = {}
            for r2, l2, a2, s2 in rows:
                if r2 == ref_label:
                    best[a2] = min(best.get(a2, (math.inf, l2)),
                                   (s2["rms"], l2))
        mark = "*" if best.get(ax, (0, ""))[1] == label else " "
        lines.append(f"  {label + ' ' + _ATT_AXIS_NAMES[ax]:16s}"
                     f"{s['rms']:8.3f}{s['mean']:9.3f}{abs(s['max']):10.3f}"
                     f"{s['tmax']:10.1f} {mark}")
    if lines:
        lines.append("")
        lines.append("  * = lowest RMS of that axis. Statistics from the end "
                     "of the warm-up on.")
    return lines


def _attitude_compare_page(rec, t, name, warmup_sec):
    """Attitude error of INSLIB / ARS / AHRS against the ground-truth
    attitude and of ARS / AHRS against the INSLIB solution (roll/pitch for
    the ARS, plus yaw for INSLIB and AHRS), with RMS / mean / max statistics.
    The largest deviation of each trace is marked. Returns None when there
    is nothing to compare."""
    groups = _att_error_series(rec)
    if not groups:
        return None
    rows = _att_stat_rows(rec, warmup_sec)
    lines = _att_table_lines(rows)
    import matplotlib.pyplot as plt

    n_plot = len(groups)
    plot_h = 3.0
    table_h = 0.26 * len(lines) + 0.6
    fig = plt.figure(figsize=(14, 1.0 + plot_h * n_plot + table_h))
    gs = fig.add_gridspec(n_plot + 1, 3,
                          height_ratios=[plot_h] * n_plot + [table_h])
    fig.suptitle(f"{name}: attitude error, INSLIB vs ARS vs AHRS")
    for r, (ref_label, traces) in enumerate(groups):
        for ax_i in range(3):
            ax = fig.add_subplot(gs[r, ax_i])
            ax.axhline(0.0, color="black", linewidth=0.5)
            drawn = False
            for label, color, a, err in traces:
                if a != ax_i:
                    continue
                y, ty = _thin(err, t)
                ax.plot(ty, y, color=color, linewidth=1.0, label=label)
                s = _att_stats(err, t, warmup_sec)
                if s is not None:
                    ax.plot([s["tmax"]], [s["max"]], "v", color=color,
                            markersize=6, markeredgecolor="black",
                            markeredgewidth=0.5)
                drawn = True
            ax.set_ylabel(f"{_ATT_AXIS_NAMES[ax_i]} err vs {ref_label} [°]")
            ax.set_xlabel("time [s]")
            if warmup_sec > 0:
                ax.axvline(warmup_sec, color="gray", linestyle=":",
                           linewidth=1)
            if drawn:
                ax.legend(loc="best", fontsize=8)
            else:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                        ha="center", va="center", color="0.5")
    _apply_grid(fig)
    ax_t = fig.add_subplot(gs[n_plot, :])
    ax_t.axis("off")
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)
    step = 1.0 / max(1, len(lines))
    y = 1.0
    for line in lines:
        ax_t.text(0.0, y, line, color="0.1", family="monospace",
                  fontsize=10, va="top")
        y -= step
    fig.tight_layout()
    return fig


def _sync_offset(ref_u, src_u):
    """Constant offset that makes ``src_u`` coincide with ``ref_u`` at the
    first sample where both are valid (unwrapped, parallel arrays). Returns
    (offset, i0) or None if they never overlap. Removes the free-running /
    magnetic-declination constant so only relative drift/walk remains."""
    for i, (r, s) in enumerate(zip(ref_u, src_u)):
        if r == r and s == s:  # both non-NaN
            return r - s, i
    return None


def _heading_page(rec, t, name):
    """Dedicated heading comparison: every absolute-heading source on one
    unwrapped time axis (no +-180 deg tearing), each synced to the
    reference at its own first common-valid sample so free-running offsets
    (ARS gyro-only start, magnetometer's magnetic-vs-true-north
    declination) don't hide the actual divergence. Top panel overlays the
    synced headings; bottom panel shows each source's deviation from the
    reference. Reference is ground truth when present, else full3d. Returns
    None if no yaw is available at all."""
    import matplotlib.pyplot as plt

    has_ref = any(row[2] == row[2] for row in rec["ref_rpy_deg"])
    ref_u = _unwrap_deg(_col(rec["ref_rpy_deg" if has_ref else "rpy_deg"], 2))
    if not any(v == v for v in ref_u):
        return None
    ref_label = "ground truth" if has_ref else "full3d"

    # (label, color, linestyle, unwrapped yaw). full3d is dropped as a
    # separate trace when it already IS the reference. mag_heading_deg /
    # gnss_course_deg are optional (missing from callers that don't record
    # them, e.g. inspostgui) -- absent keys simply contribute no trace.
    candidates = [
        ("full3d", DOWNWEIGHT_COLORS["full3d"], "-", _unwrap_deg(_col(rec["rpy_deg"], 2))),
        ("ARS", DOWNWEIGHT_COLORS["ars"], "-", _unwrap_deg(_col(rec["ars_rpy_deg"], 2))),
        ("AHRS", DOWNWEIGHT_COLORS["ahrs"], "-", _unwrap_deg(_col(rec["ahrs_rpy_deg"], 2))),
    ]
    if rec.get("mag_heading_deg"):
        candidates.append(
            ("magnetometer", "tab:green", "-", _unwrap_deg(rec["mag_heading_deg"])))
    if rec.get("gnss_course_deg"):
        candidates.append(("GNSS course (automotive)", "tab:brown", "-",
                           _unwrap_deg(rec["gnss_course_deg"])))
    if not has_ref:
        candidates = [c for c in candidates if c[0] != "full3d"]

    sources = []
    for label, color, ls, src_u in candidates:
        sync = _sync_offset(ref_u, src_u)
        if sync is None:
            continue  # never active / no overlap with the reference
        off, _ = sync
        sources.append((label, color, ls, [s + off if s == s else math.nan
                                           for s in src_u]))
    if not sources:
        return None

    fig, (ax_abs, ax_dev) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(f"{name}: heading sources (unwrapped, synced at start)")

    ref_plot, tr = _thin(ref_u, t)
    ax_abs.plot(tr, ref_plot, ":", color=REF_COLOR, linewidth=1.6,
                label=ref_label)
    for label, color, ls, synced in sources:
        y, ty = _thin(synced, t)
        ax_abs.plot(ty, y, ls, color=color, linewidth=1.2, label=label)
    ax_abs.set_ylabel("heading [°]")
    ax_abs.legend(loc="best", fontsize=8, ncol=2)

    ax_dev.axhline(0.0, color=REF_COLOR, linestyle=":", linewidth=1.6,
                   label=f"{ref_label} (reference)")
    for label, color, ls, synced in sources:
        dev = [s - r if (s == s and r == r) else math.nan
               for s, r in zip(synced, ref_u)]
        y, ty = _thin(dev, t)
        ax_dev.plot(ty, y, ls, color=color, linewidth=1.2, label=label)
    ax_dev.set_ylabel(f"heading error vs {ref_label} [°]")
    ax_dev.set_xlabel("time [s]")
    ax_dev.legend(loc="best", fontsize=8, ncol=2)

    fig.tight_layout()
    _apply_grid(fig)
    return fig


def _ne_page(rec, name):
    """North-East position (map view): the estimate's ground track, with
    the ground-truth track overlaid when available. Equal aspect so
    North/East distances are visually comparable, unlike Figure 1's
    per-axis-vs-time panels. Returns None if ins was never ready
    (all-NaN position).

    Drawn from rec["track_ne"]/rec["track_ref_ne"] when present: a
    dedicated North/East-only recorder that replay.py samples faster than
    the state history the other pages use (--plot-track-hz vs.
    --plot-hz), because a track that only spans a couple of metres shows
    every missing sample as a chord across the real path. Falls back to
    the state history when that recorder is off. Either way the vertices
    are thinned geometrically (_thin_track), not per-axis like the
    time-series pages."""
    if not any(row[0] == row[0] for row in rec["pos"]):
        return None
    import matplotlib.pyplot as plt

    # A single reference epoch (e.g. a placeholder ref.csv) is forward-
    # filled onto every later sample by replay.py/inspostgui.py, which
    # would otherwise look like a real, flat "ground truth" trajectory
    # here. rec["ref_epoch_count"] is the real, pre-forward-fill count;
    # treat it as usable ground truth only from 2 distinct epochs on.
    has_ref = (rec.get("ref_epoch_count", 2) >= 2
              and any(not math.isnan(row[0]) for row in rec["ref_pos"]))

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.suptitle(f"{name}: North-East position")

    track = rec.get("track_ne") or rec.get("pos_ref_pt") or rec["pos"]
    n, e = _thin_track(_col(track, 0), _col(track, 1))
    ax.plot(e, n, color=EST_COLOR, label="estimate")
    if has_ref:
        ref_track = rec.get("track_ref_ne") or rec["ref_pos"]
        rn, re = _thin_track(_col(ref_track, 0), _col(ref_track, 1))
        ax.plot(re, rn, "--", color=REF_COLOR, linewidth=1.2,
               label="ground truth")
    # Start and end of the estimate: on a loop or an out-and-back drive the
    # track alone does not say which way it was driven.
    valid = [(row[0], row[1]) for row in track
             if not (math.isnan(row[0]) or math.isnan(row[1]))]
    if valid:
        ax.plot(valid[0][1], valid[0][0], "o", markersize=10, markerfacecolor="white",
                markeredgecolor="black", markeredgewidth=1.8, zorder=5, label="start")
        ax.plot(valid[-1][1], valid[-1][0], "s", markersize=9, color="black",
                zorder=5, label="end")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _apply_grid(fig)
    return fig


def _altitude_page(rec, t, name):
    """Dedicated altitude profile (alt_plot_a): ins's own vertical
    estimate and baro_alt's, each with their 1-σ band, plus baro raw,
    ground truth and the raw GNSS altitude on the same axis for a direct
    visual cross-check -- all five in nav_suite's local NED frame (see
    ref_to_local_ned, replay.py), so they share one datum with no offset
    correction needed. Complements _baro_page (climb rate, z-accel bias,
    baro/GNSS offset) and its absolute-ellipsoid counterpart
    _altitude_ellipsoid_page (alt_plot_b), which repeats this same
    comparison in the WGS84 ellipsoid frame instead of this local one.

    When local_gnss_offset and rec["origin_ellipsoid_h_m"] are both
    available, INSLIB/baro_alt/baro raw also get an "(offset-corrected)"
    variant (formula spelled out in a footnote on the figure itself):
    + local_gnss_offset - origin_ellipsoid_h_m, the same offset-filter
    correction _altitude_ellipsoid_page applies, just re-expressed back in
    THIS page's local frame (undoing the ellipsoid page's + origin_
    ellipsoid_h_m) instead of its own -- so it lands right next to ground
    truth/gnss here instead of hundreds of metres away on the ellipsoid
    page's axis. Algebraically the same curve as the ellipsoid page's,
    shifted by the constant origin_ellipsoid_h_m; kept as an addition
    here (not a replacement) so the uncorrected drift stays visible too.

    Returns None if neither ins nor baro_alt ever produced a height."""
    has_ins = any(row[2] == row[2] for row in rec["pos"])
    has_baro = any(v == v for v in rec["baro_h_d"])
    if not has_ins and not has_baro:
        return None
    import matplotlib.pyplot as plt

    # A single reference epoch (e.g. a placeholder ref.csv) is forward-
    # filled onto every later sample by replay.py/inspostgui.py, which
    # would otherwise look like a real, flat "ground truth" trajectory
    # here. rec["ref_epoch_count"] is the real, pre-forward-fill count;
    # treat it as usable ground truth only from 2 distinct epochs on.
    has_ref = (rec.get("ref_epoch_count", 2) >= 2
              and any(not math.isnan(row[0]) for row in rec["ref_pos"]))
    has_gnss = any(v == v for v in rec["fix_pos_d"])
    raw_d = rec.get("baro_raw_d", [])
    has_raw = any(v == v for v in raw_d)
    origin_h = rec.get("origin_ellipsoid_h_m")
    gnss_offset_ts = rec["local_gnss_offset"]
    has_corr = (any(v == v for v in gnss_offset_ts)
               and origin_h is not None and origin_h == origin_h)

    up, shifted = _ref_pt_up(rec, len(t))
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"{name}: altitude profile"
                 + (" (board curves at the reference point)" if shifted else ""))

    if has_ins:
        alt = [-v + u for v, u in zip(_col(rec["pos"], 2), up)]
        _band(ax, t, alt, _col(rec["pos_sigma"], 2), "INSLIB", EST_COLOR)
        if has_corr:
            ins_corr = [a + o - origin_h if a == a and o == o else math.nan
                       for a, o in zip(alt, gnss_offset_ts)]
            alt_ic, t_ic = _thin(ins_corr, t)
            ax.plot(t_ic, alt_ic, "-.", color=EST_COLOR, linewidth=1.2,
                   label="INSLIB (offset-corrected)")
    if has_baro:
        alt_b = [-v + u for v, u in zip(rec["baro_h_d"], up)]
        _band(ax, t, alt_b, rec["baro_h_sigma"], "baro_alt", BARO_COLOR)
        if has_corr:
            baro_corr = [a + o - origin_h if a == a and o == o else math.nan
                        for a, o in zip(alt_b, gnss_offset_ts)]
            alt_bc, t_bc = _thin(baro_corr, t)
            ax.plot(t_bc, alt_bc, "-.", color=BARO_COLOR, linewidth=1.2,
                   label="baro_alt (offset-corrected)")
    if has_raw:
        # Raw ISA-pressure altitude uses an arbitrary QNH-vs-true absolute
        # reference, not the local NED datum the other traces share, so
        # only the SHAPE is meaningful here -- align it to baro_alt (or,
        # lacking that, INSLIB) at their first jointly-valid sample, purely
        # for a comparable overlay on the same axis (no filtering applied).
        alt_raw_pu = [-v + u if v == v else v for v, u in zip(raw_d, up)]
        anchor = alt_b if has_baro else (alt if has_ins else None)
        anchor_off = next((alt_raw_pu[i] - anchor[i] for i in range(len(alt_raw_pu))
                          if alt_raw_pu[i] == alt_raw_pu[i] and anchor is not None
                          and anchor[i] == anchor[i]), 0.0) if anchor else 0.0
        alt_raw_aligned = [v - anchor_off if v == v else v for v in alt_raw_pu]
        alt_raw, t_raw = _thin(alt_raw_aligned, t)
        ax.plot(t_raw, alt_raw, ":", color=BARO_RAW_COLOR, linewidth=1.0,
               label="baro raw (aligned)")
        if has_corr:
            raw_corr = [a + o - origin_h if a == a and o == o else math.nan
                       for a, o in zip(alt_raw_aligned, gnss_offset_ts)]
            alt_rc, t_rc = _thin(raw_corr, t)
            ax.plot(t_rc, alt_rc, "-.", color=BARO_RAW_COLOR, linewidth=1.2,
                   label="baro raw (offset-corrected)")
    if has_ref:
        alt_r, tr = _thin([-row[2] for row in rec["ref_pos"]], t)
        ax.plot(tr, alt_r, "--", color=REF_COLOR, linewidth=1.5,
               label="ground truth")
    if has_gnss:
        alt_g, tg = _thin([-v for v in rec["fix_pos_d"]], t)
        ax.plot(tg, alt_g, ":", color=GNSS_COLOR, linewidth=1.0,
               label="gnss raw")
    ax.set_ylabel("altitude [m]")
    ax.set_xlabel("time [s]")
    ax.legend(loc="best", fontsize=8)
    if has_corr:
        fig.text(0.01, 0.005,
                 "(offset-corrected) = local value + local_gnss_offset "
                 "- origin ellipsoid height, i.e. the baro-to-ellipsoid "
                 "offset filter's current estimate applied and undone "
                 "back into this page's local frame",
                 fontsize=7, color="0.4", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1) if has_corr else (0, 0, 1, 1))
    _apply_grid(fig)
    return fig


def _ref_pt_up(rec, n):
    """Per-sample up shift [m] from the sensor board (ins, baro_alt, the
    barometer) to the reference point (score.leverarm_frd, REQ-VER-037), 0
    where unknown, and whether it is anywhere nonzero."""
    up = rec.get("ref_pt_up") or []
    up = [u if u == u else 0.0 for u in up[:n]] + [0.0] * max(0, n - len(up))
    return up, any(u != 0.0 for u in up)


def _altitude_ellipsoid_page(rec, t, name):
    """Absolute-ellipsoid-height counterpart to _altitude_page (alt_plot_a):
    the same five curves -- full3d, baro_alt, baro raw, ground truth, gnss
    raw -- but in the WGS84 ellipsoid frame instead of nav_suite's local
    NED frame. alt_plot_a's ground truth/GNSS already share baro_alt/
    full3d's local origin by construction (ref_to_local_ned, replay.py),
    so no offset belongs there; this page undoes that local-origin
    conversion instead (adding rec["origin_ellipsoid_h_m"], the origin's
    own ellipsoid height, back onto ref_pos/fix_pos_d) and shifts
    full3d/baro_alt/baro raw the other way, by local_gnss_offset -- the
    same single translation applied uniformly to every curve that started
    out in the local frame. This is where the offset filter's estimate is
    actually checked against ground truth.

    Returns None if rec["origin_ellipsoid_h_m"] was never recorded (no
    GNSS-anchored origin) or none of the five source curves are
    available."""
    origin_h = rec.get("origin_ellipsoid_h_m")
    if origin_h is None or origin_h != origin_h:
        return None

    has_offset = any(v == v for v in rec["local_gnss_offset"])
    has_ins = has_offset and any(row[2] == row[2] for row in rec["pos"])
    has_baro = has_offset and any(v == v for v in rec["baro_h_d"])
    raw_d = rec.get("baro_raw_d", [])
    has_raw = has_offset and has_baro and any(v == v for v in raw_d)
    # A single reference epoch (e.g. a placeholder ref.csv) is forward-
    # filled onto every later sample by replay.py/inspostgui.py, which
    # would otherwise look like a real, flat "ground truth" trajectory
    # here. rec["ref_epoch_count"] is the real, pre-forward-fill count;
    # treat it as usable ground truth only from 2 distinct epochs on.
    has_ref = (rec.get("ref_epoch_count", 2) >= 2
              and any(not math.isnan(row[0]) for row in rec["ref_pos"]))
    has_gnss = any(v == v for v in rec["fix_pos_d"])
    if not (has_ins or has_baro or has_raw or has_ref or has_gnss):
        return None
    import matplotlib.pyplot as plt

    offset = rec["local_gnss_offset"]
    up, shifted = _ref_pt_up(rec, len(t))

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"{name}: altitude profile (ellipsoid height)"
                 + (", board curves at the reference point" if shifted else ""))

    if has_ins:
        ell = [-row[2] + u + o if row[2] == row[2] and o == o else math.nan
              for row, o, u in zip(rec["pos"], offset, up)]
        alt_i, ti = _thin(ell, t)
        ax.plot(ti, alt_i, color=EST_COLOR, label="INSLIB")
    baro_up = None
    if has_baro:
        baro_up = [-h + u if h == h else math.nan for h, u in zip(rec["baro_h_d"], up)]
        baro_ell = [b + o if b == b and o == o else math.nan
                   for b, o in zip(baro_up, offset)]
        alt_b, tb = _thin(baro_ell, t)
        ax.plot(tb, alt_b, color=BARO_COLOR, label="baro_alt")
    if has_raw:
        # Same anchor-alignment as _altitude_page: raw ISA-pressure altitude
        # uses an arbitrary QNH-vs-true reference, not baro_alt's own local
        # datum, so it is shifted to coincide with baro_alt (alt_plot_a's
        # anchor) at their first jointly-valid sample before local_gnss_
        # offset -- which belongs to baro_alt's datum -- applies to it too.
        alt_raw_pu = [-v + u if v == v else v for v, u in zip(raw_d, up)]
        anchor_off = next((alt_raw_pu[i] - baro_up[i] for i in range(len(alt_raw_pu))
                          if alt_raw_pu[i] == alt_raw_pu[i] and baro_up[i] == baro_up[i]),
                          0.0)
        raw_ell = [v - anchor_off + o if v == v and o == o else math.nan
                  for v, o in zip(alt_raw_pu, offset)]
        alt_r, tr = _thin(raw_ell, t)
        ax.plot(tr, alt_r, ":", color=BARO_RAW_COLOR, linewidth=1.0,
               label="baro raw (aligned)")
    if has_ref:
        ref_ell = [origin_h - row[2] if row[2] == row[2] else math.nan
                  for row in rec["ref_pos"]]
        alt_g, tg = _thin(ref_ell, t)
        ax.plot(tg, alt_g, "--", color=REF_COLOR, linewidth=1.5,
               label="ground truth")
    if has_gnss:
        fix_ell = [origin_h - v if v == v else math.nan for v in rec["fix_pos_d"]]
        alt_f, tf = _thin(fix_ell, t)
        ax.plot(tf, alt_f, ":", color=GNSS_COLOR, linewidth=1.0,
               label="gnss raw")

    ax.set_ylabel("ellipsoid height [m]")
    ax.set_xlabel("time [s]")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _apply_grid(fig)
    return fig


def _baro_page(rec, t, name):
    """baro_alt vertical channel: height and climb rate (positive up, its
    own natural convention) of the two filters that actually produce a
    vertical estimate, baro_alt vs. full3d -- ground truth/GNSS are
    deliberately left off (see alt_plot_a/_altitude_page and its
    ellipsoid counterpart _altitude_ellipsoid_page/alt_plot_b for those
    cross-checks; this page is baro_alt-vs-full3d only) -- plus baro_alt's
    own z-accel bias + 1-σ band, and (if a baro/GNSS pair was ever seen)
    the baro-to-ellipsoid offset filter's estimate + 1-σ band. Returns
    None if baro_alt was never active. Everything is stored NED-down
    internally (rec's usual convention, see replay.py); negated here only
    for display."""
    if not any(v == v for v in rec["baro_h_d"]):
        return None
    import matplotlib.pyplot as plt

    has_offset = any(v == v for v in rec["local_gnss_offset"])

    n_rows = 4 if has_offset else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 13 if has_offset else 10))
    fig.suptitle(f"{name}: baro_alt vertical channel")

    ax_h = axes[0]
    h, th = _thin([-v for v in rec["baro_h_d"]], t)
    ax_h.plot(th, h, color=EST_COLOR, label="baro_alt")
    full3d_h, tf = _thin([-row[2] for row in rec["pos"]], t)
    ax_h.plot(tf, full3d_h, "--", color="0.4", linewidth=1.2, label="full3d")
    ax_h.set_ylabel("height [m]")
    ax_h.set_xlabel("time [s]")
    ax_h.legend(loc="best", fontsize=8)

    ax_v = axes[1]
    v, tv = _thin([-x for x in rec["baro_vel_d"]], t)
    ax_v.plot(tv, v, color=EST_COLOR, label="baro_alt")
    full3d_v, tfv = _thin([-row[2] for row in rec["vel"]], t)
    ax_v.plot(tfv, full3d_v, "--", color="0.4", linewidth=1.2, label="full3d")
    ax_v.set_ylabel("climb rate [m/s]")
    ax_v.set_xlabel("time [s]")
    ax_v.legend(loc="best", fontsize=8)

    ax_b = axes[2]
    _band(ax_b, t, rec["baro_acc_bias"], rec["baro_acc_bias_sigma"],
         "baro_alt z-accel bias", EST_COLOR)
    ax_b.set_ylabel("z-accel bias [m/s²]")
    ax_b.set_xlabel("time [s]")
    ax_b.legend(loc="best", fontsize=8)

    if has_offset:
        ax_o = axes[3]
        _band(ax_o, t, rec["local_gnss_offset"], rec["local_gnss_offset_sigma"],
             "baro-to-ellipsoid offset", EST_COLOR)
        ax_o.set_ylabel("gnss offset [m]")
        ax_o.set_xlabel("time [s]")
        ax_o.legend(loc="best", fontsize=8)

    fig.tight_layout()
    _apply_grid(fig)
    return fig


def _gnss_delay_page(rec, t, name, curve, configured_delay_ms=None):
    """Overview + zoom + correlation-vs-lag for the GNSS-delay estimate.
    Overlaying the whole trial (often an hour+) makes a
    sub-second shift invisible, so the zoom panel centers on the
    strongest vertical motion instead, where the shift is actually
    visible by eye; it also overlays GNSS shifted by the estimated
    delay, so a good fit is immediately obvious. ``configured_delay_ms``
    (the current config `gnss: delay_ms`), when given, drives the
    same actionable hint the console prints. Returns None if
    ``curve`` is None (estimate not computed/applicable)."""
    if curve is None:
        return None
    import matplotlib.pyplot as plt

    baro = rec["baro_vel_d"]
    gnss = rec["gnss_vel_d"]
    delay_ms, corr = max(curve, key=lambda row: row[1])
    delay_s = delay_ms / 1000.0

    fig, (ax_over, ax_zoom, ax_corr) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f"{name}: GNSS delay estimate "
                f"({delay_ms:.0f} ms, correlation {corr:.2f})")

    # Overview: whole trial. A small delay won't show as a visible shift
    # at this zoom level (see the caveat in replay.py) -- it's here for
    # context (where is there vertical motion at all), not for reading
    # off the delay itself.
    baro_o, t_baro_o = _thin(baro, t)
    gnss_o, t_gnss_o = _thin(gnss, t)
    ax_over.plot(t_baro_o, baro_o, color=EST_COLOR, linewidth=0.8, label="baro_alt vD")
    ax_over.plot(t_gnss_o, gnss_o, color=REF_COLOR, linewidth=0.8, label="GNSS vD")
    ax_over.set_ylabel("vD [m/s]")
    ax_over.set_xlabel("time [s]")
    ax_over.set_title("full trial")
    ax_over.legend(loc="best", fontsize=8)

    # Zoom: centered on the strongest vertical motion, where a shift
    # between the two curves is actually visible by eye. The dotted
    # curve is GNSS shifted by the estimated delay -- if it now tracks
    # baro_alt closely, that's a visual confirmation of the number.
    valid = [(ti, b, g) for ti, b, g in zip(t, baro, gnss) if b == b and g == g]
    if valid:
        peak_t = max(valid, key=lambda row: abs(row[1]))[0]
        # A sub-second delay is invisible over a window spanning minutes
        # (see the overview panel above); 15 s is wide enough to place
        # the maneuver in context but still resolve a sub-second shift.
        span = 15.0
        lo, hi = peak_t - span / 2.0, peak_t + span / 2.0
        zoom_rows = [(ti, b, g) for ti, b, g in zip(t, baro, gnss) if lo <= ti <= hi]
        zt = [r[0] for r in zoom_rows]
        zb = [r[1] for r in zoom_rows]
        zg = [r[2] for r in zoom_rows]
        zt_shifted = [r[0] - delay_s for r in zoom_rows]
        # Direct line through the actual measurement points (not a
        # held/step rendering) for both GNSS traces, so the shift is
        # directly comparable point-to-point against baro_alt.
        ax_zoom.plot(zt, zb, color=EST_COLOR, label="baro_alt vD")
        ax_zoom.plot(zt, zg, color=REF_COLOR, label="GNSS vD")
        ax_zoom.plot(zt_shifted, zg, ":", color=REF_COLOR, alpha=0.7,
                    linewidth=2, label=f"GNSS vD shifted {delay_ms:.0f} ms")
        ax_zoom.set_xlim(lo, hi)
    ax_zoom.set_ylabel("vD [m/s]")
    ax_zoom.set_xlabel("time [s]")
    ax_zoom.set_title("zoom on the strongest vertical motion")
    ax_zoom.legend(loc="best", fontsize=8)

    # Correlation vs. lag: a sharp, narrow peak means a trustworthy
    # estimate; a broad/flat one (little vertical motion) means don't
    # trust the exact number, only that some delay is plausible.
    lags = [row[0] for row in curve]
    corrs = [row[1] for row in curve]
    ax_corr.plot(lags, corrs, color="black", linewidth=1.0)
    ax_corr.axvline(delay_ms, color=EST_COLOR, linestyle="--")
    ax_corr.plot([delay_ms], [corr], "o", color=EST_COLOR)
    ax_corr.axhline(0.0, color="gray", linewidth=0.5)
    ax_corr.set_xlabel("lag [ms] (positive: GNSS behind baro_alt)")
    ax_corr.set_ylabel("correlation")
    ax_corr.set_title("correlation vs. lag (sharp peak = trustworthy)")

    # Same actionable hint as the console: which config knob consumes this
    # number, and its current value. Only recommend applying it when the
    # peak is sharp enough to trust (mirrors replay.py's corr > 0.7 gate).
    cur = (f", now {configured_delay_ms:.0f} ms"
           if configured_delay_ms is not None else "")
    hint = (f"set config  gnss: delay_ms: {delay_ms:.0f}{cur}"
            if corr > 0.7 else
            f"config gnss: delay_ms{cur}: correlation too low to trust")
    ax_corr.text(0.5, -0.32, hint, transform=ax_corr.transAxes,
                 ha="center", va="top", fontsize=9,
                 color=(EST_COLOR if corr > 0.7 else "gray"))

    fig.tight_layout()
    _apply_grid(fig)
    return fig


def _cap_sigma_axis(ax, values, cap, unit):
    """Bound a stddev axis per _sigma_axis_limit(), annotating the corner
    with the true peak when anything was clipped. Leaves the data untouched:
    the trace still runs off the top, only the view is bounded."""
    limit, peak = _sigma_axis_limit(values, cap)
    if limit is None:
        return
    ax.set_ylim(0.0, limit if limit > 0.0 else None)
    if limit < peak:
        ax.text(0.99, 0.95, f"clipped at {limit:.4g} {unit} (peak {peak:.4g} {unit})",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color=EST_COLOR)


def _cap_symmetric_axis(ax, center, sigma, cap, unit):
    """Bound a zero-centred error+sigma axis to +/- the limit of
    _sigma_axis_limit() applied to |center| + sigma, annotating the corner
    with the true peak when anything was clipped."""
    envelope = [abs(c) + abs(sg) for c, sg in zip(center, sigma)
                if c == c and sg == sg]
    limit, peak = _sigma_axis_limit(envelope, cap)
    if limit is None or limit <= 0.0:
        return
    ax.set_ylim(-limit, limit)
    if limit < peak:
        ax.text(0.99, 0.95,
                f"clipped at +/-{limit:.4g} {unit} (peak {peak:.4g} {unit})",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color=EST_COLOR)


def _gnss_stddev_page(rec, t, name):
    """GNSS position/velocity NED stddev as reported by the receiver
    (sqrt of the diagonal of the fix's own cov_pos/cov_vel, replay.py's
    rec['gnss_pos_sigma']/['gnss_vel_sigma'], held between fixes like the
    other raw-GNSS traces -- see meas_overlay_tree). This is the accuracy
    the receiver claims BEFORE ins.c's own scale/floor conditioning
    (REQ-NAV-038/-041, gnss.pos_cov_scale etc. in config.yaml) and before
    the chi2 downweight gate, so a stretch of degraded/optimistic
    receiver-reported accuracy (multipath, an RTK float-to-fix or
    DGPS-mode switch, ...) is visible here even where _outlier_page's
    downweight counters stay flat. Returns None if no GNSS fix ever
    carried a covariance."""
    has_pos = any(row[0] == row[0] for row in rec.get("gnss_pos_sigma", []))
    if not has_pos:
        return None
    import matplotlib.pyplot as plt

    has_vel = any(row[0] == row[0] for row in rec.get("gnss_vel_sigma", []))
    n_rows = 2 if has_vel else 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 3.5 * n_rows), sharex=True,
                             squeeze=False)
    fig.suptitle(f"{name}: GNSS reported stddev (receiver, NED)")
    for i, axis_lbl in enumerate(("N", "E", "D")):
        ax = axes[0][i]
        raw = _col(rec["gnss_pos_sigma"], i)
        y, ty = _thin(raw, t)
        ax.step(ty, y, where="post", color=GNSS_COLOR)
        ax.set_ylabel(f"pos {axis_lbl} stddev [m]")
        _cap_sigma_axis(ax, raw, _SIGMA_AXIS_CAP_M, "m")
        if not has_vel:
            ax.set_xlabel("time [s]")
    if has_vel:
        for i, axis_lbl in enumerate(("N", "E", "D")):
            ax = axes[1][i]
            raw = _col(rec["gnss_vel_sigma"], i)
            y, ty = _thin(raw, t)
            ax.step(ty, y, where="post", color=GNSS_COLOR)
            ax.set_ylabel(f"vel {axis_lbl} stddev [m/s]")
            _cap_sigma_axis(ax, raw, _SIGMA_AXIS_CAP_MPS, "m/s")
            ax.set_xlabel("time [s]")
    fig.tight_layout()
    _apply_grid(fig)
    return fig


# gnss.csv quality (RTKLIB numbering): label and colour of each level, in
# the order the y axis lists them (best on top).
_GNSS_Q_LEVELS = ((1, "fix", "#1e8449"), (2, "float", "#c87f0a"),
                  (6, "ppp", "#2471a3"), (4, "dgnss", "#7d3c98"),
                  (3, "sbas", "#5d6d7e"), (5, "single", "#c0392b"))


# gnss.csv jamming/spoofing/sig_auth/nav_auth: row label and the colour of
# each known state (0 unknown is not drawn). Good green, doubtful orange,
# bad red.
_GNSS_IND_ROWS = (
    ("jamming", "jamming", {1: "#1e8449", 2: "#c87f0a", 3: "#c0392b"}),
    ("spoofing", "spoofing", {1: "#1e8449", 2: "#c87f0a", 3: "#c0392b"}),
    ("sig_auth", "signal auth", {1: "#c87f0a", 2: "#1e8449", 3: "#c0392b"}),
    ("nav_auth", "nav msg auth", {1: "#c87f0a", 2: "#1e8449", 3: "#c0392b"}))
_GNSS_IND_STATE_NAMES = {
    "jamming": {1: "none", 2: "warning", 3: "critical"},
    "spoofing": {1: "none", 2: "indicated", 3: "confirmed"},
    "sig_auth": {1: "not authenticated", 2: "authenticated", 3: "failed"},
    "nav_auth": {1: "not authenticated", 2: "authenticated", 3: "failed"}}


def _gnss_quality_page(rec, name, configured_delay_ms=None):
    """GNSS fix quality over time (gnss.csv's quality column, one dot per
    fix on its level), the transport latency of every fix (arrival on
    the recorder's clock minus the fix's own GPS time, via timesync.csv,
    replay.gnss_latency_stats) against the configured gnss: delay_ms, the
    satellites used (num_sv) and the receiver's jamming/spoofing and
    authentication states (one row each, coloured by state). Each panel
    only where the data carries it, None when it carries none."""
    qt, q = rec.get("gnss_q_t") or [], rec.get("gnss_q") or []
    lt, lat = rec.get("gnss_lat_t") or [], rec.get("gnss_lat_ms") or []
    st, sv = rec.get("gnss_sv_t") or [], rec.get("gnss_sv") or []
    it, ind = rec.get("gnss_ind_t") or [], rec.get("gnss_ind") or {}
    ind_rows = [r for r in _GNSS_IND_ROWS if any(ind.get(r[0]) or [])]
    if not q and not lat and not sv and not ind_rows:
        return None
    import matplotlib.pyplot as plt
    rows = sum(1 for p in (q, lat, sv, ind_rows) if p)
    fig, axes = plt.subplots(rows, 1, figsize=(12, 3.5 * rows), sharex=True,
                             squeeze=False)
    fig.suptitle(f"{name}: GNSS fix quality and latency")
    ax_i = 0
    if q:
        ax = axes[ax_i][0]
        ax_i += 1
        levels = [lv for lv in _GNSS_Q_LEVELS if lv[0] in set(q)]
        ypos = {lv[0]: len(levels) - i for i, lv in enumerate(levels)}
        n = len(q)
        for code, label, color in levels:
            sel = [i for i, v in enumerate(q) if v == code]
            ax.plot([qt[i] for i in sel], [ypos[code]] * len(sel), "|",
                    color=color, markersize=12,
                    label=f"{label} {100.0 * len(sel) / n:.1f}%")
        ax.set_yticks([ypos[lv[0]] for lv in levels])
        ax.set_yticklabels([lv[1] for lv in levels])
        ax.set_ylim(0.5, len(levels) + 0.5)
        ax.set_ylabel("fix quality")
        ax.legend(loc="upper right", fontsize=8, ncol=len(levels))
    if lat:
        ax = axes[ax_i][0]
        ax_i += 1
        lat_v, lat_t = _thin(list(lat), list(lt))
        ax.plot(lat_t, lat_v, ".", markersize=2, color=EST_COLOR,
                label="transport latency")
        if configured_delay_ms is not None:
            ax.axhline(configured_delay_ms, color="black", linestyle="--",
                       linewidth=1, label=f"configured delay_ms {configured_delay_ms:.0f}")
        s = sorted(lat)
        med = s[len(s) // 2]
        lo, hi = s[int(0.01 * (len(s) - 1))], s[int(0.99 * (len(s) - 1))]
        # Zero, the data (1..99 %) and the configured delay always in
        # view, an all-negative latency included.
        bottom = min(0.0, lo)
        top = max(0.0, hi, configured_delay_ms or 0.0)
        ax.set_ylim(bottom - 0.1 * (top - bottom) - 10.0,
                    top + 0.2 * (top - bottom) + 10.0)
        ax.set_ylabel("latency [ms]")
        ax.set_title(f"arrival minus GPS time of validity, median {med:.0f} ms",
                     fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
    if sv:
        ax = axes[ax_i][0]
        ax_i += 1
        sv_v, sv_t = _thin(list(sv), list(st))
        ax.step(sv_t, sv_v, where="post", color=GNSS_COLOR)
        ax.set_ylim(0, max(sv) + 2)
        ax.set_ylabel("satellites used")
    if ind_rows:
        ax = axes[ax_i][0]
        ax_i += 1
        n_rows = len(ind_rows)
        for i, (key, label, colors) in enumerate(ind_rows):
            y = n_rows - i
            vals = ind[key]
            for state, color in colors.items():
                sel = [j for j, v in enumerate(vals) if v == state]
                if sel:
                    ax.plot([it[j] for j in sel], [y] * len(sel), "|",
                            color=color, markersize=12,
                            label=f"{label}: {_GNSS_IND_STATE_NAMES[key][state]}")
        ax.set_yticks([n_rows - i for i in range(n_rows)])
        ax.set_yticklabels([r[1] for r in ind_rows])
        ax.set_ylim(0.5, n_rows + 0.5)
        ax.set_ylabel("receiver integrity")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
    axes[-1][0].set_xlabel("time [s]")
    fig.tight_layout()
    _apply_grid(fig)
    return fig


def _outlier_page(rec, t, name):
    """Cumulative chi2-downweight counters over time, one
    step line per sub-filter, so an outlier-heavy stretch shows up as a
    jump at the time it happened rather than only as a total in the
    end-of-run printout (REQ-VER-012). Each line is included only if
    that sub-filter was ever active (same has_* gating as the other
    pages); returns None if none were."""
    sources = (
        ("full3d", "dw_full3d", lambda: any(row[0] == row[0] for row in rec["pos"])),
        ("ars", "dw_ars", lambda: any(row[0] == row[0] for row in rec["ars_rpy_deg"])),
        ("ahrs", "dw_ahrs", lambda: any(row[0] == row[0] for row in rec["ahrs_rpy_deg"])),
        ("baro_alt", "dw_baro_alt", lambda: any(v == v for v in rec["baro_h_d"])),
        ("local_gnss", "dw_local_gnss", lambda: any(v == v for v in rec["local_gnss_offset"])),
    )
    active = [(key, dwkey) for key, dwkey, has in sources if has()]
    if not active:
        return None
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"{name}: outlier rejection (cumulative chi2 downweights)")
    for key, dwkey in active:
        y, ty = _thin(rec[dwkey], t)
        ax.step(ty, y, where="post", color=DOWNWEIGHT_COLORS[key],
               label=DOWNWEIGHT_LABELS[key])
    ax.set_ylabel("cumulative downweighted count")
    ax.set_xlabel("time [s]")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _apply_grid(fig)
    return fig


def _zupt_page(rec, t, name):
    """Timeline of the three independent stillness detectors -- ins, ARS/
    AHRS, baro_alt -- one horizontal bar row each, instead of the single
    zupt_active OR'd together for the gray shading on the other pages, so
    a mismatch between them (e.g. the ARS's own velocity-blind fallback
    considers the platform still while ins's position-aided detector
    doesn't, because ins has no GNSS lock yet) is visible directly.

    INS auto-ZUPT/ZARU: ins's own detector, velocity-aware (uses ins's
    estimated velocity, GNSS-informed when available), needs ins ready
    (position aiding), REQ-SUITE-009/-010.

    ARS/AHRS ZARU applied: the ARS's and AHRS's own last-epoch "did a
    zero-rotation trigger actually fire" flag (ars_zaru_applied/
    ahrs_zaru_applied, REQ-AHRS-024), OR'd together into one row. NOT
    ars_auto_zaru_active/ahrs_auto_zaru_active -- that pair is only a
    loose "stillness run started" gate that stays true through genuine
    constant-velocity cruise (can't tell that from a stop, velocity-
    blind), well before the stricter windowed-variance criterion and
    dwell time this row actually requires. This row already includes
    whatever ins relayed down to the ARS/AHRS (nav_suite's zaru_active()/
    last_zaru_trigger, src/nav_suite.c:594), so it is always >= the ins
    row above -- the interesting part is where it goes true WITHOUT ins.

    baro_alt vertical ZUPT: whatever zero-velocity update actually reached
    baro_alt's vertical filter this epoch (REQ-SUITE-015) -- baro_alt has
    no stillness detector of its own (see baro_alt_zero_velocity_update,
    src/baro_alt.c:628). It is exactly the OR of the two rows above (ins
    auto-ZUPT/ZARU OR ARS/AHRS ZARU applied, src/nav_suite.c:728-729).

    Returns None if none of the three detectors were ever active."""
    ars = rec.get("ars_zaru_applied", [])
    ahrs = rec.get("ahrs_zaru_applied", [])
    ars_ahrs = [1.0 if a == 1.0 or h == 1.0 else 0.0 for a, h in zip(ars, ahrs)]
    rows = (
        ("INS auto-ZUPT/ZARU", rec.get("auto_zupt_active", []), EST_COLOR),
        ("ARS/AHRS ZARU applied", ars_ahrs, "tab:orange"),
        ("baro_alt vertical ZUPT", rec.get("vertical_zupt_active", []), BARO_COLOR),
    )
    active = [(label, sig, color) for label, sig, color in rows if any(v == 1.0 for v in sig)]
    if not active:
        return None
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 1.0 * len(active) + 1.5))
    fig.suptitle(f"{name}: ZUPT/ZARU detector timeline")
    for i, (label, sig, color) in enumerate(active):
        spans = _zupt_spans(t, sig)
        bars = [(lo, hi - lo) for lo, hi in spans]
        ax.broken_barh(bars, (i + 0.1, 0.8), facecolors=color)
    ax.set_yticks([i + 0.5 for i in range(len(active))])
    ax.set_yticklabels([label for label, _, _ in active])
    ax.set_ylim(0, len(active))
    ax.invert_yaxis()
    ax.set_xlabel("time [s]")
    ax.set_xlim(t[0], t[-1])
    fig.tight_layout()
    _apply_grid(fig)
    return fig


def _sensor_rate_page(sensor_rate, name):
    """Each input stream's sampling rate [Hz] over time (bucketed by
    replay.py's _imu_prepass()/_bucketed_rate()) -- a time-
    resolved complement to the avg-Hz/max-gap numbers in the data-quality
    summary, so a rate drop or dropout can be located and correlated
    against the other pages. Returns None if ``sensor_rate`` wasn't
    given (only computed together with --plot)."""
    if not sensor_rate:
        return None
    import matplotlib.pyplot as plt

    order = [k for k in ("imu", "gnss", "mag", "baro") if k in sensor_rate]
    fig, axes = plt.subplots(len(order), 1, figsize=(12, 2.6 * len(order)),
                             sharex=True, squeeze=False)
    fig.suptitle(f"{name}: sensor sampling rate over time")
    for i, key in enumerate(order):
        ax = axes[i][0]
        ts, hz = sensor_rate[key]
        hz, ts = _thin(hz, ts)
        ax.plot(ts, hz, color=SENSOR_RATE_COLORS[key])
        ax.fill_between(ts, 0.0, hz, color=SENSOR_RATE_COLORS[key], alpha=0.2,
                        linewidth=0)
        ax.set_ylabel(f"{key} [Hz]")
        ax.set_ylim(bottom=0.0)
    axes[-1][0].set_xlabel("time [s]")
    fig.tight_layout()
    _apply_grid(fig)
    return fig


def _save_pdf(figs, out_path):
    from matplotlib.backends.backend_pdf import PdfPages

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with PdfPages(out_path) as pdf:
        for fig in figs:
            pdf.savefig(fig)
    print(f"wrote {out_path} ({len(figs)} page{'s' if len(figs) != 1 else ''})")
