"""Navigator.set_ahrs_tuning(): the ahrs_config_t leveling/magnetometer
tuning fields reach both the ARS and the AHRS, and 0 leaves ahrs's own
defaults untouched. Observed through the filters' own reported attitude
stddev: fewer accelerometer corrections per second leave a larger roll/pitch
uncertainty after the same static run. Runs under pytest or standalone

(c) Jan Zwiener (jan@zwiener.org)
(``python3 python/tests/test_ahrs_tuning.py``)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from INSLIB import Navigator, Config  # noqa: E402

_G = 9.80665
_DT = 0.01
_ACC_VAR = (3.8e-6,) * 3
_GYR_VAR = (3.4e-11,) * 3


def _run(tuning=None, seconds=5.0):
    """Static IMU for ``seconds``; returns (ARS, AHRS) roll/pitch stddev."""
    cfg = Config(lat_rad=math.radians(48.783), lon_rad=math.radians(9.181),
                 h_m=300.0, auto_init=False, rpy_init_rad=(0.0, 0.0, 0.0),
                 magnetic_n=(20.0, 0.0, 40.0), auto_zupt_disable=True,
                 allow_unlimited_deadreckoning=True)
    nav = Navigator(cfg)
    try:
        if tuning is not None:
            nav.set_ahrs_tuning(**tuning)
        t = 0
        for i in range(int(seconds / _DT)):
            t += int(_DT * 1e6)
            nav.imu(t, _DT, (0.0, 0.0, -_G), (0.0, 0.0, 0.0), _ACC_VAR, _GYR_VAR)
            if i == 0:
                nav.mag((20.0, 0.0, 40.0), (1.0,) * 3)
            nav.update()
        return nav.rpy_stddev_ars()[:2], nav.rpy_stddev_ahrs()[:2]
    finally:
        nav.close()


def test_zero_leaves_defaults():
    base = _run()
    assert _run({}) == base
    assert _run({"acc_freq_hz": 0.0, "chi2_threshold": 0.0}) == base
    print("ok  set_ahrs_tuning with zeros keeps the ahrs defaults")


def test_acc_freq_reaches_both_filters():
    base = _run()
    slow = _run({"acc_freq_hz": 0.5})
    for i in (0, 1):  # ARS, AHRS
        assert all(s > b * 1.5 for s, b in zip(slow[i], base[i])), (slow, base)
    print("ok  acc_freq_hz changes the ARS and the AHRS")


if __name__ == "__main__":
    test_zero_leaves_defaults()
    test_acc_freq_reaches_both_filters()
