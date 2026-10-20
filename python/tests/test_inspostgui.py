#!/usr/bin/env python3
"""The Qt side of tools/inspostgui.py, driven headless (offscreen platform).

The replay itself is covered by test_replay_core.py. This file is about the
window: the config form's unsaved-changes tracking and the guards built on
it, the trail legend, the keyboard shortcuts, the "Last run" panel after a
replay and the split error report of a failed one.

Needs PyQt6 and is skipped without it. Runs under pytest or standalone:

    python3 python/tests/test_inspostgui.py
"""

import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_range_stream  # noqa: E402  (its A_ideal dataset)

try:
    from PyQt6 import QtCore, QtGui, QtTest, QtWidgets
except ImportError:
    QtCore = None


def _window(settings_dir):
    """A MainWindow with QSettings of its own in settings_dir, so the test
    leaves the user's real window settings alone."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import inspostgui
    os.makedirs(settings_dir, exist_ok=True)
    settings = QtCore.QSettings(os.path.join(settings_dir, "gui.ini"),
                                QtCore.QSettings.Format.IniFormat)
    return app, inspostgui, inspostgui.MainWindow(settings)


def _answer(button):
    """Make QMessageBox.question() answer `button` from now on, returns
    the list the asked questions are collected in."""
    asked = []

    def question(_parent, title, text, *_a, **_k):
        asked.append((title, text))
        return button

    QtWidgets.QMessageBox.question = staticmethod(question)
    return asked


def _type_into(line_edit, text):
    line_edit.clear()
    QtTest.QTest.keyClicks(line_edit, text)


def _run_and_wait(app, win, timeout=240.0):
    win._on_run()
    t_end = time.time() + timeout
    while win.worker.isRunning() and time.time() < t_end:
        app.processEvents()
        time.sleep(0.02)
    win.worker.wait(5000)
    app.processEvents()  # the queued finished/error signal


def test_gui_config_dirty_tracking_and_guards():
    if QtCore is None:
        print("  skip  PyQt6 not installed")
        return
    Btn = QtWidgets.QMessageBox.StandardButton
    with tempfile.TemporaryDirectory() as d:
        app, gui, win = _window(os.path.join(d, "settings"))
        os.makedirs(os.path.join(d, "a"))
        ds_a = test_range_stream._dataset(os.path.join(d, "a"), 1e6, False)
        ds_b = os.path.join(d, "b")
        shutil.copytree(ds_a, ds_b)

        assert win.open_dataset(ds_a)
        assert not win.editor.is_dirty()
        assert "*" not in win.windowTitle()
        assert win.tabs.tabText(win.TAB_CONFIG) == "Config"

        # A typed value is an edit, programmatic loads are not.
        _, _, name_edit = win.editor._widgets[("name",)]
        old = name_edit.text()
        _type_into(name_edit, "renamed")
        assert win.editor.is_dirty()
        assert win.windowTitle().endswith("*")
        assert win.tabs.tabText(win.TAB_CONFIG) == "Config *"
        assert "modified" in win.cfg_label.text()
        # ... and typing the old value back is no edit any more.
        _type_into(name_edit, old)
        assert not win.editor.is_dirty()
        assert win.tabs.tabText(win.TAB_CONFIG) == "Config"

        # Cancel keeps the form, the combo goes back to the open config.
        _type_into(name_edit, "renamed")
        asked = _answer(Btn.Cancel)
        win.dataset_combo.addItem("b", os.path.join(ds_b, "config.yaml"))
        win.dataset_combo.setCurrentIndex(win.dataset_combo.count() - 1)
        win._on_dataset_selected(win.dataset_combo.currentIndex())
        assert asked and win.editor.is_dirty()
        assert win.cfg_path == os.path.join(ds_a, "config.yaml")
        assert (os.path.abspath(win.dataset_combo.currentData())
                == os.path.abspath(win.cfg_path))

        # Discard drops the edit and opens the other config.
        _answer(Btn.Discard)
        assert win.open_dataset(ds_b)
        assert not win.editor.is_dirty()
        assert win.cfg_path == os.path.join(ds_b, "config.yaml")

        # Save writes the file and clears the marker.
        _type_into(name_edit, "saved name")
        QtWidgets.QMessageBox.question = staticmethod(
            lambda *a, **k: Btn.Yes if "Overwrite" in a[1] else Btn.Save)
        assert win.open_dataset(ds_a)
        with open(os.path.join(ds_b, "config.yaml"), encoding="utf-8") as f:
            assert "saved name" in f.read()
        assert not win.editor.is_dirty()

        # Closing with unsaved edits asks, Cancel keeps the window.
        _type_into(name_edit, "again")
        _answer(Btn.Cancel)
        ev = QtGui.QCloseEvent()
        win.closeEvent(ev)
        assert not ev.isAccepted()
        _answer(Btn.Discard)
        ev = QtGui.QCloseEvent()
        win.closeEvent(ev)
        assert ev.isAccepted()


def test_gui_legend_follows_overlays_and_speed_scale():
    if QtCore is None:
        print("  skip  PyQt6 not installed")
        return
    with tempfile.TemporaryDirectory() as d:
        _app, _gui, win = _window(os.path.join(d, "settings"))
        for chk in (win.chk_ref, win.chk_fix, win.chk_zupt, win.chk_ghost):
            chk.setChecked(False)
        assert win.legend._shown == []
        win.chk_zupt.setChecked(True)
        win.chk_ref.setChecked(True)
        assert win.legend._shown == ["ref", "zupt"]
        win.chk_zupt.setChecked(False)
        assert win.legend._shown == ["ref"]
        # The ramp end follows the trail's speed scale.
        win.pos_view.append_trail([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                                  [0.0, 30.0], [False, False])
        win.legend.set_scale(win.pos_view.speed_scale())
        assert win.legend._vmax == win.pos_view.speed_scale() > 30.0
        win.legend.resize(600, 22)
        assert not win.legend.grab().isNull()  # paints without raising


def test_gui_trail_speed_scale_is_adjustable():
    if QtCore is None:
        print("  skip  PyQt6 not installed")
        return
    with tempfile.TemporaryDirectory() as d:
        _app, _gui, win = _window(os.path.join(d, "settings"))
        pts, spd, brk = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], [0.0, 30.0], [False] * 2
        assert win.scale_combo.currentText() == "auto"
        # Auto widens to the fastest sample ...
        win.pos_view.append_trail(pts, spd, brk)
        assert win.pos_view.speed_scale() > 30.0
        # ... a typed value pins it, whatever the speeds are.
        win.scale_combo.setCurrentText("2")
        win._on_speed_scale_edited()
        assert win.pos_view.speed_scale() == 2.0
        assert win.scale_combo.currentText() == "2 m/s"
        assert win.legend._vmax == 2.0
        win.pos_view.append_trail(pts, spd, brk)
        assert win.pos_view.speed_scale() == 2.0
        win.pos_view.reset_trail()
        assert win.pos_view.speed_scale() == 2.0
        # Nonsense falls back to auto, "auto" restores the widening.
        win.scale_combo.setCurrentText("-3")
        win._on_speed_scale_edited()
        assert win.scale_combo.currentText() == "auto"
        assert win.pos_view.speed_scale() == 5.0
        win.pos_view.append_trail(pts, spd, brk)
        assert win.pos_view.speed_scale() > 30.0
        # The choice is remembered for the next start.
        win.scale_combo.setCurrentText("12.5 m/s")
        win._on_speed_scale_edited()
        _app2, _gui2, win2 = _window(os.path.join(d, "settings"))
        assert win2.pos_view.speed_scale() == 12.5
        assert win2.scale_combo.currentText() == "12.5 m/s"


def test_gui_shortcuts():
    if QtCore is None:
        print("  skip  PyQt6 not installed")
        return
    with tempfile.TemporaryDirectory() as d:
        _app, _gui, win = _window(os.path.join(d, "settings"))
        keys = {sc.key().toString() for sc in win.findChildren(QtGui.QShortcut)}
        for want in ("F5", "Space", "Esc", "Ctrl+O", "Ctrl+S",
                     "Ctrl+Shift+S", "Ctrl+1", "Ctrl+5"):
            assert want in keys, (want, sorted(keys))
        # Space pauses only while a run can be paused ...
        win._on_space()
        assert not win.pause_btn.isChecked()
        win.pause_btn.setEnabled(True)
        win._on_space()
        assert win.pause_btn.isChecked()
        win._on_space()
        assert not win.pause_btn.isChecked()
        # ... and presses a focused check box instead of pausing.
        win.show()
        win.activateWindow()
        QtTest.QTest.qWaitForWindowActive(win)
        win.chk_model.setFocus()
        before = win.chk_model.isChecked()
        win._on_space()
        assert win.chk_model.isChecked() != before
        assert not win.pause_btn.isChecked()
        win.chk_model.setChecked(before)
        # A space typed into a text field stays a space.
        win.outage_edit.setFocus()
        win.outage_edit.clear()
        QtTest.QTest.keyClicks(win.outage_edit, "60 30")
        assert win.outage_edit.text() == "60 30"
        assert not win.pause_btn.isChecked()
        win.hide()
        # Ctrl+<n> picks the tab.
        win.tabs.setCurrentIndex(win.TAB_REPLAY)
        for sc in win.findChildren(QtGui.QShortcut):
            if sc.key().toString() == "Ctrl+5":
                sc.activated.emit()
        assert win.tabs.currentIndex() == win.TAB_SUMMARY


def test_gui_run_result_panel_and_error_report():
    if QtCore is None:
        print("  skip  PyQt6 not installed")
        return
    with tempfile.TemporaryDirectory() as d:
        app, _gui, win = _window(os.path.join(d, "settings"))
        os.makedirs(os.path.join(d, "a"))
        ds = test_range_stream._dataset(os.path.join(d, "a"), 1e6, False)
        assert win.open_dataset(ds)
        assert not win.result_frame.isVisibleTo(win)

        _run_and_wait(app, win)
        assert win.results is not None
        assert win.result_frame.isVisibleTo(win)
        text = win.lbl_result.text()
        assert "pos rms" in text and "att rms r/p/y" in text, text
        assert "previous" not in text  # nothing to compare with yet
        assert win.lbl_findings.text().startswith("findings: ")

        # A second run knows the first as "previous", an unsaved edit is
        # called out, and starting a run hides the old result.
        _, _, name_edit = win.editor._widgets[("name",)]
        _type_into(name_edit, "edited")
        _run_and_wait(app, win)
        text = win.lbl_result.text()
        assert "previous" in text and "Δ" in text, text
        assert "unsaved config edits" in text, text

        # The error report: short text, traceback on demand.
        shown = {}

        class FakeBox:
            def __init__(self, _icon, title, text, *_a):
                shown.update(title=title, text=text)

            def setDetailedText(self, details):
                shown["details"] = details

            def exec(self):
                shown["exec"] = True

        real = QtWidgets.QMessageBox
        QtWidgets.QMessageBox = type("QMB", (), {
            "Icon": real.Icon, "StandardButton": real.StandardButton,
            "__new__": lambda cls, *a: FakeBox(*a)})
        try:
            win._on_worker_error("ValueError: boom", "Traceback ...\nboom")
            assert shown["text"] == "ValueError: boom"
            assert shown["details"].startswith("Traceback")
            shown.clear()
            win._on_worker_error("config problem", "")
            assert "details" not in shown and shown["exec"]
        finally:
            QtWidgets.QMessageBox = real


def test_gui_other_dataset_clears_view():
    if QtCore is None:
        print("  skip  PyQt6 not installed")
        return
    with tempfile.TemporaryDirectory() as d:
        app, _gui, win = _window(os.path.join(d, "settings"))
        assert win.run_btn.objectName() == "primary"  # the green Run button
        for sub in ("a", "b"):
            os.makedirs(os.path.join(d, sub))
        ds_a = test_range_stream._dataset(os.path.join(d, "a"), 1e6, False)
        ds_b = test_range_stream._dataset(os.path.join(d, "b"), 1e6, False)
        assert win.open_dataset(ds_a)
        _run_and_wait(app, win)
        assert win.results is not None and win.pos_view.trail_points

        # Another config of the same dataset keeps the run as comparison.
        cfg_a = win.cfg_path
        variant = os.path.join(os.path.dirname(cfg_a), "config_variant.yaml")
        shutil.copyfile(cfg_a, variant)
        assert win.open_dataset(variant)
        assert win.results is not None and win.pos_view.trail_points

        # Another dataset empties the views, the ghost included.
        assert win.open_dataset(ds_b)
        assert win.results is None and win._ghost is None
        assert not win.pos_view.trail_points
        assert not win.result_frame.isVisibleTo(win)
        assert not win.pdf_btn.isEnabled()

        # Opened while a replay runs: that replay is stopped and cut loose,
        # so it cannot draw the old dataset into the emptied views.
        win.chk_realtime.setChecked(True)
        win._on_run()
        t_end = time.time() + 30.0
        while not win.pos_view.trail_points and time.time() < t_end:
            app.processEvents()
            time.sleep(0.02)
        assert win.pos_view.trail_points
        assert win.open_dataset(ds_a)
        assert win.worker is None and win.run_btn.isEnabled()
        for _ in range(20):
            app.processEvents()
            time.sleep(0.02)
        assert win.results is None and not win.pos_view.trail_points
        win.close()  # waits for the stopped replay
        assert not any(w.isRunning() for w in win._stale_workers)


if __name__ == "__main__":
    fails = 0
    for fn in (test_gui_config_dirty_tracking_and_guards,
               test_gui_legend_follows_overlays_and_speed_scale,
               test_gui_trail_speed_scale_is_adjustable,
               test_gui_shortcuts,
               test_gui_run_result_panel_and_error_report,
               test_gui_other_dataset_clears_view):
        try:
            fn()
            print("  ok    %s" % fn.__name__)
        except AssertionError as e:
            fails += 1
            print("  FAIL  %s\n%s" % (fn.__name__, e))
    print("==== %d failures ====" % fails)
    sys.exit(1 if fails else 0)
