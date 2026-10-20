"""Self-test for replay.py's standstill accelerometer-vs-gravity check
(acc_gravity_metrics): it uses the median, so a few outliers from a
disturbed standstill do not move it, and it stays silent without enough
samples. Runs under pytest or standalone
(``python3 python/tests/test_acc_gravity.py``).

(c) Jan Zwiener (jan@zwiener.org)
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "tools"))

import replay  # noqa: E402

_LAT = math.radians(49.7)


def test_too_few_samples_gives_nothing():
    assert replay.acc_gravity_metrics([9.81] * (replay.ACC_GRAVITY_MIN_SAMPLES - 1),
                                      _LAT, 100.0) is None


def test_median_ignores_outliers():
    g = replay.normal_gravity(_LAT, 100.0)
    assert g is not None
    norms = [g * 1.05] * 100 + [30.0] * 10
    m = replay.acc_gravity_metrics(norms, _LAT, 100.0)
    assert abs(m["median"] - g * 1.05) < 1e-3 * g, m
    assert abs(m["dev"] - 0.05) < 1e-3, m
    assert m["n"] == 110


def test_matching_sensor_has_no_deviation():
    g = replay.normal_gravity(_LAT, 100.0)
    m = replay.acc_gravity_metrics([g] * 200, _LAT, 100.0)
    assert abs(m["dev"]) < 1e-6, m


_TESTS = [test_too_few_samples_gives_nothing, test_median_ignores_outliers,
          test_matching_sensor_has_no_deviation]


if __name__ == "__main__":
    fails = 0
    for t in _TESTS:
        try:
            t()
            print(f"ok    {t.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{fails} failures")
    sys.exit(1 if fails else 0)
