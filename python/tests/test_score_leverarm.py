#!/usr/bin/env python3
"""score: leverarm_frd applied wherever the estimate meets the reference
(REQ-VER-037): replay.ref_point_offset_ned(), which the ellipsoid height
score and the --plot recorders of python/replay.py and python/inspostgui.py
add to the IMU-point quantities, and ins_plots._ref_pt_up(), which shifts
the board curves of the altitude pages.

Runs under pytest or standalone:

    python3 python/tests/test_score_leverarm.py
"""

import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))

import replay     # noqa: E402
import ins_plots  # noqa: E402


class _Nav:
    def __init__(self, rpy):
        self._rpy = rpy
        self.calls = 0

    def rpy(self):
        self.calls += 1
        return self._rpy


def test_height_projection_matches_rotated_lever_arm():
    la = (-2.0, 0.1, -1.3)  # antenna behind, right of and above the IMU
    for rpy_deg in ((0.0, 0.0, 0.0), (5.0, -10.0, 0.0), (5.0, -10.0, 135.0),
                    (-20.0, 15.0, -60.0)):
        rpy = tuple(math.radians(v) for v in rpy_deg)
        d = replay.ref_point_offset_ned(_Nav(rpy), la)
        # Same rotation as ins: R_b_to_n from ZYX roll/pitch/yaw.
        R = replay._rotmat_from_rpy(rpy)
        want = [sum(R[r][k] * la[k] for k in range(3)) for r in range(3)]
        assert all(abs(a - b) < 1e-12 for a, b in zip(d, want)), (rpy_deg, d, want)
        # The height part depends on roll and pitch only.
        r, p = rpy[0], rpy[1]
        down = (-math.sin(p) * la[0] + math.sin(r) * math.cos(p) * la[1]
                + math.cos(r) * math.cos(p) * la[2])
        assert abs(d[2] - down) < 1e-12, (rpy_deg, d[2], down)
    # Level: the antenna 1.3 m above the IMU is 1.3 m up.
    assert abs(replay.ref_point_offset_ned(_Nav((0.0, 0.0, 0.0)), la)[2] + 1.3) < 1e-12
    # No attitude yet: treated as level, so the height is still right.
    assert abs(replay.ref_point_offset_ned(_Nav(None), la)[2] + 1.3) < 1e-12
    # The altitude pages shift every board curve by -down, per sample.
    up, shifted = ins_plots._ref_pt_up({"ref_pt_up": [1.3, float("nan"), 1.2]}, 4)
    assert shifted and up == [1.3, 0.0, 1.2, 0.0], up


def test_zero_lever_arm_is_a_noop():
    nav = _Nav((0.3, -0.2, 1.0))
    assert replay.ref_point_offset_ned(nav, (0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)
    assert nav.calls == 0
    up, shifted = ins_plots._ref_pt_up({"ref_pt_up": [0.0, 0.0]}, 2)
    assert not shifted and up == [0.0, 0.0]
    up, shifted = ins_plots._ref_pt_up({}, 3)
    assert not shifted and up == [0.0, 0.0, 0.0]


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:     # noqa: BLE001
                failed += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if failed else 0)
