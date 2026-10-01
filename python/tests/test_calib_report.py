#!/usr/bin/env python3
"""Tests for the calibration certificate and record,
tools/inslib_calib_report.py.

The document prints numbers the solve already produced, so most of it is
formatting. What is tested here is the part that is not: the uncertainty
it states has to cover the truth of a session whose truth is known, the
coverage figures have to mean what they claim, the verdict has to fail
what must not be used, anything a user types has to come through LaTeX
intact, a Makefile that belongs to somebody else must not be replaced,
and the raw recording written beside the document has to reproduce the
calibration exactly (same coefficient checksum). When pdflatex is on the
PATH the document is also built, which is the only check that the LaTeX
is valid and that the certificate really is one page.

The session is the synthetic one of test_calib_csv.py (known accelerometer
bias, scale and misalignment, known magnetometer errors).

Runs under pytest or standalone:

    python3 python/tests/test_calib_report.py
"""

import copy
import datetime
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, HERE)
import inslib_calib_report as rp   # noqa: E402
import inslib_frame_align as fa     # noqa: E402
import inslib_imu_calib as calib    # noqa: E402
import inslib_imu_tk as imu_tk      # noqa: E402
import test_calib_csv as session   # noqa: E402

INIT_SEC = session.INIT_SEC - 1.0   # what _solved() ran with


def _input(**kw):
    rec, cal, magcal = session._solved()
    args = dict(rec=rec, cal=cal, magcal=magcal, init_sec=INIT_SEC,
                operator="tester", device="unit under test",
                place="Lab 2 & Co_#1 {100%}", serial="SN-7",
                position=(48.91, 8.4656, 160.0), field_ref_ut=session.FIELD_UT,
                field_source="test", wmm_decl_deg=session.DECL_DEG,
                wmm_incl_deg=session.DIP_DEG,
                t_start=datetime.datetime(2026, 9, 21, 10, 0, 0).astimezone(),
                t_end=datetime.datetime(2026, 9, 21, 10, 3, 0).astimezone())
    args.update(kw)
    return rp.ReportInput(**args)


def _housing():
    return fa.housing_rotation([("bottom", [0.05, -0.08, -9.806]),
                                ("left", [0.1, 9.80, 0.05])])


def test_accel_uncertainty_covers_the_truth():
    rec, cal, _ = session._solved()
    unc = rp.acc_uncertainty(rec, cal)
    assert unc is not None
    assert unc["dof"] == cal.n_positions - 9
    err = np.abs(np.asarray(cal.acc_bias) - session.B_ACC)
    sig = np.asarray(unc["bias"])
    assert (sig > 0).all()
    # k=4 so the test is not flaky, but tight enough that an uncertainty
    # off by an order of magnitude in either direction fails one of these.
    assert (err <= 4.0 * sig).all(), (err, sig)
    assert (sig < 0.01).all(), sig
    scale = np.diag(np.asarray(cal.acc_matrix))
    true_scale = np.diag(session.M_ACC)
    assert (np.abs(scale - true_scale) <= 4.0 * np.asarray(unc["scale"])).all()


def test_acc_params_round_trip():
    _rec, cal, _ = session._solved()
    p = rp.acc_params(cal.acc_matrix, cal.acc_bias)
    assert np.allclose(imu_tk.acc_matrix(p), np.asarray(cal.acc_matrix),
                       atol=1e-12)
    assert np.allclose(p[6:9], cal.acc_bias)


def test_coverage_figures():
    six = np.vstack([np.eye(3), -np.eye(3)])
    c = rp.coverage(six)
    assert all(c["axes"])
    assert abs(c["spread"] - 1.0) < 1e-9
    # The largest empty cap around the six axes is centred on a cube
    # diagonal, 54.7 deg from each of them. The gap is searched on a grid
    # about 4.5 deg apart, so it comes out up to a couple of degrees short.
    assert 52.0 < c["gap_deg"] <= 54.8, c["gap_deg"]
    ang = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    flat = np.column_stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)])
    c = rp.coverage(flat)
    assert c["spread"] < 1e-9
    assert c["axes"] == [True, True, True, True, False, False]
    assert 87.0 < c["gap_deg"] <= 90.0, c["gap_deg"]


def test_verdict_fails_a_residual_calibration():
    ok = rp.verdict(rp.checks(_input(), {"axes": [True] * 6}))
    assert ok in (rp.OK, rp.WARN)
    bad = rp.checks(_input(rec_cal_frac=0.4), {"axes": [True] * 6})
    assert rp.verdict(bad) == rp.FAIL


def test_verdict_fails_too_few_poses():
    _rec, cal, _ = session._solved()
    few = copy.copy(cal)
    few.n_positions = 3
    rows = rp.checks(_input(cal=few), {"axes": [True] * 6})
    assert rp.verdict(rows) == rp.FAIL


def test_document_content_and_escaping():
    txt = rp.build_tex(_input())
    for part in ("Calibration Certificate", "Calibration Record",
                 r"\section{Acceptance criteria}",
                 r"\section{Orientation coverage}",
                 r"\section{Static poses: accelerometer}",
                 "KP-20260921-100000-SN-7", "Page 1 of 1",
                 r"Lab 2 \& Co\_\#1 \{100\%\}", "tester", r"\begin{longtable}",
                 "2026-09-21", r"\label{cert:end}", r"\label{rec:last}"):
        assert part in txt, part
    assert txt.rstrip().endswith(r"\end{document}")
    # The certificate comes first and ends before the record starts.
    assert txt.index(r"\label{cert:end}") < txt.index(r"\pagestyle{record}")
    # The digest is a function of the coefficients alone.
    _rec, cal, magcal = session._solved()
    assert rp.coefficient_digest(cal, magcal) in txt
    other = copy.copy(cal)
    other.acc_bias = [cal.acc_bias[0] + 1e-6] + list(cal.acc_bias[1:])
    assert rp.coefficient_digest(other, magcal) != \
        rp.coefficient_digest(cal, magcal)


def test_document_without_magnetometer_or_position():
    txt = rp.build_tex(_input(magcal=None, position=None, field_ref_ut=0.0))
    assert r"\subsection{Magnetometer}" not in txt
    assert r"\textit{not given}" in txt
    assert txt.rstrip().endswith(r"\end{document}")


def test_raw_recording_reproduces_the_calibration():
    """What the record promises: its attached files, run through the
    solve with the inputs it prints, give the same coefficients."""
    inp = _input()
    with tempfile.TemporaryDirectory() as d:
        written = rp.write_report(os.path.join(d, "c.tex"), inp)
        imu = os.path.join(d, "c_imu.csv")
        mag = os.path.join(d, "c_mag.csv")
        assert imu in written and mag in written
        rec = calib.load_csv_recording(imu, mag)
        with open(os.path.join(d, "c.tex"), encoding="utf-8") as f:
            txt = f.read()
    assert "c_imu.csv" in txt and "c\\_mag.csv" in txt
    cmd = rp.reproduce_command(inp, [rp.RawFile("c_imu.csv", "", 0, ""),
                                     rp.RawFile("c_mag.csv", "", 0, "")])
    assert "--gravity %.6f" % inp.cal.gravity in cmd
    assert "--init-sec %g" % INIT_SEC in cmd
    assert "--mag-field-ut" in cmd
    assert len(rec) == len(inp.rec) and len(rec.mag) == len(inp.rec.mag)
    assert np.allclose(rec.mag_temp_c, inp.rec.mag_temp_c, equal_nan=True)
    cal, magcal = calib.solve_session(rec, gravity=inp.cal.gravity,
                                      init_sec=INIT_SEC,
                                      field_ut=inp.field_ref_ut)
    assert rp.coefficient_digest(cal, magcal) == \
        rp.coefficient_digest(inp.cal, inp.magcal)


def test_foreign_makefile_is_left_alone():
    with tempfile.TemporaryDirectory() as d:
        mk = os.path.join(d, "Makefile")
        with open(mk, "w", encoding="utf-8") as f:
            f.write("all:\n\techo mine\n")
        written = rp.write_report(os.path.join(d, "a.tex"), _input(),
                                  raw=False)
        assert written == [os.path.join(d, "a.tex")]
        with open(mk, encoding="utf-8") as f:
            assert "mine" in f.read()
        os.remove(mk)
        written = rp.write_report(os.path.join(d, "a.tex"), _input(),
                                  raw=False)
        assert mk in written
        # Rewriting our own is fine, that is how it picks up a new version.
        assert mk in rp.write_report(os.path.join(d, "a.tex"), _input(),
                                     raw=False)


def _build(d, name):
    for _ in range(2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             name + ".tex"], cwd=d, capture_output=True, text=True,
            timeout=300, encoding="utf-8", errors="replace")
        assert proc.returncode == 0, proc.stdout[-2000:]
    with open(os.path.join(d, name + ".aux"), encoding="utf-8",
              errors="replace") as f:
        return f.read()


def test_pdflatex_builds_it_and_the_certificate_is_one_page():
    """Optional: skipped where there is no pdflatex (CI, most boxes).

    The worst case for page 1: magnetometer with alignment, a housing
    rotation, the longest remarks the certificate still prints, and long
    free-text fields. The certificate ends with a label, and while it is
    being set the pages are numbered roman: the label has to land on
    page i."""
    if shutil.which("pdflatex") is None:
        print("   (pdflatex not found, skipped)")
        return
    long = "x" * 30 + " "
    inp = _input(housing=_housing(),
                 remarks=("remark " * 60)[:rp.CERT_REMARKS_MAX],
                 place=long * 2, device=long * 2,
                 operator="Firstname Lastname-Longname",
                 gravity_source=long * 2)
    with tempfile.TemporaryDirectory() as d:
        rp.write_report(os.path.join(d, "k.tex"), inp)
        aux = _build(d, "k")
        assert re.search(r"\\newlabel\{cert:end\}\{\{[^}]*\}\{i\}", aux), aux
        with open(os.path.join(d, "k.pdf"), "rb") as f:
            pdf = f.read()
        assert len(pdf) > 10000
        assert b"k_imu.csv" in pdf and b"/EmbeddedFile" in pdf


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("ok    %s" % name)
        except AssertionError as e:
            failures += 1
            print("FAIL  %s: %s" % (name, e))
    print("==== %d failures ====" % failures)
    sys.exit(1 if failures else 0)
