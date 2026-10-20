"""Top-level config key kalman_update_dt_sec (REQ-VER-049): accepted by the
replay schema, carried into the ins options by tools/replay.py's
build_config, and 0 keeps ins's own covariance-prediction default.

Observed through the filter's own covariance: after a few IMU epochs that
are shorter than the default prediction period, a filter predicting at the
IMU rate has already grown its position uncertainty, one at the default
period has not. Runs under pytest or standalone

(``python3 python/tests/test_kalman_cadence.py``)."""

import math
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "tools"))

import replay                    # noqa: E402
from INSLIB import Navigator     # noqa: E402

_G = 9.80665
_DT = 0.005
_ACC_VAR = (3.8e-6,) * 3
_GYR_VAR = (3.4e-11,) * 3
_LAT, _LON, _H = math.radians(48.783), math.radians(9.181), 300.0


def _config(extra):
    """build_config() output for a minimal dataset config plus `extra`."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.yaml"), "w", encoding="utf-8") as fh:
            fh.write("name: cadence\naiding: none\ninit: auto\n"
                     "imu:\n  gyr_psd: 1.0e-8\n  acc_psd: 1.0e-5\n" + extra)
        spec, _ = replay.load_config(d)
    cfg = replay.build_config(spec, None, 0, _LAT, _LON, _H, None)
    cfg.auto_init = False
    cfg.rpy_init_rad = (0.0, 0.0, 0.0)
    cfg.allow_unlimited_deadreckoning = True
    cfg.auto_zupt_disable = True
    return cfg


def _pos_variance_after(cfg, epochs):
    nav = Navigator(cfg)
    try:
        t = 0
        for _ in range(epochs):
            t += int(_DT * 1e6)
            nav.imu(t, _DT, (0.0, 0.0, -_G), (0.0, 0.0, 0.0), _ACC_VAR, _GYR_VAR)
            nav.update()
        P = nav.covariance()
        return P[0][0] + P[1][1] + P[2][2]
    finally:
        nav.close()


def test_key_reaches_ins_options():
    assert _config("").kalman_update_dt_sec == 0.0
    assert _config("kalman_update_dt_sec: 0.005\n").kalman_update_dt_sec == 0.005
    print("ok  kalman_update_dt_sec is accepted and carried into the options")


def test_imu_rate_prediction_grows_covariance_sooner():
    # 4 epochs of 5 ms: well inside the default prediction period, so only
    # the filter told to predict every IMU sample has propagated anything
    first = _pos_variance_after(_config(""), 1)
    default = _pos_variance_after(_config(""), 4)
    fast = _pos_variance_after(_config("kalman_update_dt_sec: 0.005\n"), 4)
    assert default == first, (default, first)
    assert fast > default, (fast, default)
    print("ok  kalman_update_dt_sec changes the ins prediction cadence")


if __name__ == "__main__":
    test_key_reaches_ins_options()
    test_imu_rate_prediction_grows_covariance_sooner()
