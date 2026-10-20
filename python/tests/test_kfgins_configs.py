#!/usr/bin/env python3
"""The configs in the datasets/kfgins subdirectories have no data next to
them in this repository, so no replay ever reads them. Loading them through
the replay schema catches a renamed or dropped key.

Runs under pytest or standalone:

    python3 python/tests/test_kfgins_configs.py
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import replay   # noqa: E402

BASE = os.path.join(REPO, "datasets", "kfgins")


def test_kfgins_subdirectory_configs_stay_loadable():
    subdirs = sorted(d for d in os.listdir(BASE)
                     if os.path.isfile(os.path.join(BASE, d, "config.yaml")))
    assert subdirs, BASE
    for d in subdirs:
        spec, _ = replay.load_config(os.path.join(BASE, d))
        assert spec["init"] == "ref" and spec["kalman_update_dt_sec"] > 0.0, d
        assert os.path.isfile(os.path.join(BASE, d, "kf-gins.yaml")), d


if __name__ == "__main__":
    test_kfgins_subdirectory_configs_stay_loadable()
    print("ok  datasets/kfgins subdirectory configs load")
