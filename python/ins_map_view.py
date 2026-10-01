#!/usr/bin/env python3
"""ins_map_view -- an OpenStreetMap background for a replayed lat/lon track.

The tile fetch machinery (identifying User-Agent, cache-first, on-screen
budget, no prefetch) is the same policy tools/inslib_gui.py's Track tab
follows, and TILE_CACHE points at the very same directory on disk, so a
dataset looked at here in inspostgui.py and a device looked at live there
do not each download the same square of the world twice.
See https://operations.osmfoundation.org/policies/tiles/

Kept as its own module rather than copied into different GUIs: tools/ and
python/ do not import across each other (separate scripts, separate
dependency sets), which is also why python/ins_map_frames.py keeps its
own plain-urllib copy of this same cache/policy rather than importing
Qt/pyqtgraph for a batch PNG renderer that has no window.

MapView itself does not know about INSLIB, replay.py or ECEF/ENU - it
takes plain (lat_deg, lon_deg) sequences, projects them to local metres
itself (Web Mercator, scaled at the origin, exactly like the Track tab)
and draws them over the tiles. That keeps it usable from anything that
can produce a list of positions.

(c) Jan Zwiener (jan@zwiener.org)
"""

import math
import os
import queue

from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg

import ins_gui_theme as theme

EARTH_R = 6378137.0

TILE_URL = "https://tile.openstreetmap.org/%d/%d/%d.png"
TILE_AGENT = "inslib_gui/1.0 (INSLIB device console; local diagnostics)"
TILE_BUDGET = 24
# Same cache root as tools/inslib_gui.py and python/ins_map_frames.py --
# "inslib_gui" names the shared cache, not this particular script.
TILE_CACHE = os.path.join(
    os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache"),
    "inslib_gui", "tiles")


def merc(lat_deg, lon_deg):
    """WGS84 -> Web Mercator metres, the frame the tiles are cut in."""
    lat = max(-85.05112878, min(85.05112878, lat_deg))
    return (EARTH_R * math.radians(lon_deg),
            EARTH_R * math.log(math.tan(math.pi / 4.0
                                        + math.radians(lat) / 2.0)))


class TileFetcher(QtCore.QThread):
    """Fetches map tiles off the UI thread, cache first.

    One thread, a queue, and no prefetching: a map view that scrolled
    would otherwise turn into exactly the bulk download the tile policy
    asks applications not to do.
    """

    ready = QtCore.pyqtSignal(int, int, int, bytes)

    def __init__(self):
        super().__init__()
        self.q = queue.Queue()
        # What is IN FLIGHT, so the same square is not queued twice while
        # it is being fetched. Not a record of everything ever asked for:
        # a view that comes back to a tile has to be able to ask again.
        self.pending = set()
        self._stop = False

    def want(self, z, x, y):
        if (z, x, y) in self.pending:
            return
        self.pending.add((z, x, y))
        self.q.put((z, x, y))

    def stop(self):
        self._stop = True
        self.q.put(None)

    def run(self):
        import urllib.request
        while not self._stop:
            item = self.q.get()
            if item is None:
                break
            z, x, y = item
            path = os.path.join(TILE_CACHE, str(z), str(x), "%d.png" % y)
            try:
                if os.path.exists(path):
                    with open(path, "rb") as fh:
                        data = fh.read()
                else:
                    req = urllib.request.Request(
                        TILE_URL % (z, x, y),
                        headers={"User-Agent": TILE_AGENT})
                    with urllib.request.urlopen(req, timeout=8) as r:
                        data = r.read()
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as fh:
                        fh.write(data)
                self.ready.emit(z, x, y, data)
            except Exception:                             # noqa: BLE001
                # A missing tile is a blank square, not an error dialog:
                # the track underneath is the point of the view and it is
                # still there.
                pass
            finally:
                self.pending.discard((z, x, y))


def _qimage_to_array(img):
    """QImage -> (h, w, 3) uint8, the orientation pyqtgraph draws upright."""
    img = img.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    ptr.setsize(h * img.bytesPerLine())
    arr = np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine())
    arr = arr[:, :w * 3].reshape(h, w, 3)
    # Tiles number their rows from the north down; the plot's y axis runs
    # the other way, so the image is flipped once here rather than by
    # inverting an axis and mirroring the track with it.
    #
    # The copy is the point of the last step and not a tidiness: everything
    # above is a VIEW into the QImage's own buffer, and that buffer is
    # released with the QImage when this function returns. What the plot
    # would keep is a pointer into freed memory, which paints as noise
    # until the process happens to fall over instead.
    return np.transpose(arr[::-1], (1, 0, 2)).copy()


class MapView(QtWidgets.QWidget):
    """A track, a ground-truth track and the GNSS fixes over an optional
    OSM background.

    Unlike the Track tab in tools/inslib_gui.py this is handed a whole
    replayed trajectory at once rather than growing one live, so there is
    no Ring buffer and no data-refresh timer -- set_tracks() replaces
    everything, and the only timer here re-tiles after a zoom or pan.

    The street map is optional and off until switched on. Tiles need a
    network, and asking for them tells somebody else's server which area
    is being looked at -- which is to say where this dataset was
    recorded. That is a decision for whoever is looking, not a default.
    """

    TILE_KEEP = 96

    def __init__(self, parent=None):
        super().__init__(parent)
        self.origin = None
        self.origin_merc = None
        self.scale = 1.0
        self.fetcher = None
        self.tiles = {}          # (z, x, y) -> the item on the plot
        self.images = {}         # (z, x, y) -> the decoded pixels
        self.want = set()        # what the current view is made of
        self._build()

    def _build(self):
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)

        bar = QtWidgets.QHBoxLayout()
        self.info = QtWidgets.QLabel("no track yet")

        bar.addWidget(self.info, 1)
        self.show_ref = QtWidgets.QCheckBox("Truth")
        self.show_ref.setChecked(True)
        self.show_ref.toggled.connect(lambda on: self.c_ref.setVisible(on))
        bar.addWidget(self.show_ref)
        self.show_fix = QtWidgets.QCheckBox("GNSS fixes")
        self.show_fix.setChecked(True)
        self.show_fix.toggled.connect(lambda on: self.c_fix.setVisible(on))
        bar.addWidget(self.show_fix)
        self.map_on = QtWidgets.QCheckBox("OpenStreetMap background")
        self.map_on.setToolTip(
            "Fetches map tiles from tile.openstreetmap.org for the area on "
            "screen. That tells their server where you are looking.")
        self.map_on.toggled.connect(self._map_toggled)
        bar.addWidget(self.map_on)
        v.addLayout(bar)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel('left', 'north of the first fix [m]')
        self.plot.setLabel('bottom', 'east of the first fix [m]')
        # Locked 1:1, or the shape of the track is a lie and the tiles
        # underneath it would not line up either.
        self.plot.setAspectLocked(True)
        self.plot.addLegend(offset=(10, 10))
        # The previous run's estimate, first so it draws behind.
        self.c_ghost = self.plot.plot([], [], name="previous run")
        self._ghost_wanted = True
        self.c_ref = self.plot.plot([], [], name="truth")
        # Dots, not a line: joining 1 Hz fixes draws chords nobody drove.
        self.c_fix = self.plot.plot(
            [], [], pen=None, symbol="o", symbolSize=4,
            symbolPen=None, name="GNSS fixes")
        self.c_est = self.plot.plot([], [], name="estimate")
        self.c_here = self.plot.plot(
            [], [], pen=None, symbol="+", symbolSize=14, name="last")
        v.addWidget(self.plot, 1)

        # A zoom or a pan changes which tiles are on screen, and this is
        # a static view: there is no next refresh() tick to catch up on
        # it, so it is driven off the plot's own range-changed signal
        # instead, coalesced since that fires once per wheel notch and
        # once per mouse move while dragging.
        self.view_timer = QtCore.QTimer(self)
        self.view_timer.setSingleShot(True)
        self.view_timer.setInterval(120)
        self.view_timer.timeout.connect(self._view_settled)
        self.plot.getPlotItem().sigRangeChanged.connect(
            lambda *_a: self.view_timer.start())

        self.note = QtWidgets.QLabel(
            "Metres east and north of the first estimated position, on "
            "the Web Mercator sphere so map tiles align.")
        self.note.setWordWrap(True)
        v.addWidget(self.note)
        theme.themed(self._restyle)

    def _restyle(self):
        self.info.setStyleSheet(theme.dim())
        self.note.setStyleSheet(theme.dim() + " font-size: 11px;")
        theme.style_plot(self.plot)
        self.c_ref.setPen(theme.pen("ref", 1.5, theme.DASH))
        self.c_ghost.setPen(theme.pen("ghost", 2))
        self.c_fix.setSymbolBrush(theme.T["fix"])
        self.c_est.setPen(theme.pen("map_est", 2))
        self.c_here.setSymbolBrush(theme.T["map_here"])
        self.c_here.setSymbolPen(theme.T["map_here"])

    # -- projection ----------------------------------------------------

    def _project(self, lat, lon):
        """(east, north) in metres against the first position projected."""
        if self.origin is None:
            self.origin = (lat, lon)
            self.origin_merc = merc(lat, lon)
            # Web Mercator stretches by 1/cos(latitude), isotropically over
            # a small area. Undoing that here is what lets one coordinate
            # system carry both true distances AND aligned tiles.
            self.scale = math.cos(math.radians(lat))
        mx, my = merc(lat, lon)
        return ((mx - self.origin_merc[0]) * self.scale,
                (my - self.origin_merc[1]) * self.scale)

    def _to_merc(self, east, north):
        return (self.origin_merc[0] + east / self.scale,
                self.origin_merc[1] + north / self.scale)

    def _view_rect(self):
        return self.plot.getViewBox().viewRect()

    # -- data ------------------------------------------------------------

    def clear(self):
        self.origin = None
        self.origin_merc = None
        self.scale = 1.0
        for c in (self.c_est, self.c_ref, self.c_fix, self.c_here,
                  self.c_ghost):
            c.setData([], [])
        # Every tile rect is measured against the origin, and the next
        # track sets a new one. Keeping the pictures would keep them
        # where the OLD origin put them.
        self._drop_tiles()
        self.images.clear()
        self.info.setText("no track yet")

    def set_ghost_visible(self, on):
        self._ghost_wanted = on
        self.c_ghost.setVisible(on)

    def set_tracks(self, est_latlon, ref_latlon, fix_latlon=(),
                   ghost_latlon=()):
        """Replace the view with a replayed trajectory.

        est_latlon/ref_latlon/fix_latlon/ghost_latlon: sequences of
        (lat_deg, lon_deg, ...) -- e.g. inspostgui.py's results["kml_est"]/
        ["kml_ref"] rows, extra columns (time, altitude, attitude) are
        ignored. ghost_latlon is a previous run's estimate. Any may be
        empty. The origin is the first point of whichever is projected
        first (est, then ref, then fix).
        """
        self.clear()
        if not est_latlon and not ref_latlon and not fix_latlon:
            return

        def project(rows):
            return (np.array([self._project(r[0], r[1]) for r in rows])
                    if rows else np.empty((0, 2)))
        est_xy = project(est_latlon)
        ref_xy = project(ref_latlon)
        fix_xy = project(fix_latlon)
        ghost_xy = project(ghost_latlon)
        if est_xy.size:
            self.c_est.setData(est_xy[:, 0], est_xy[:, 1])
            self.c_here.setData([est_xy[-1, 0]], [est_xy[-1, 1]])
        if ref_xy.size:
            self.c_ref.setData(ref_xy[:, 0], ref_xy[:, 1])
        if fix_xy.size:
            self.c_fix.setData(fix_xy[:, 0], fix_xy[:, 1])
        if ghost_xy.size:
            self.c_ghost.setData(ghost_xy[:, 0], ghost_xy[:, 1])
        self.c_ghost.setVisible(self._ghost_wanted and bool(ghost_xy.size))
        self.info.setText("estimate %d pts   truth %d pts   GNSS %d fixes"
                          % (len(est_latlon), len(ref_latlon),
                             len(fix_latlon)))
        if self.map_on.isChecked():
            self._update_tiles()

    # -- tiles -------------------------------------------------------------

    def _map_toggled(self, on):
        if on:
            m = QtWidgets.QMessageBox(self)
            m.setWindowTitle("Fetch map tiles?")
            m.setIcon(QtWidgets.QMessageBox.Icon.Question)
            m.setText(
                "This fetches map tiles from tile.openstreetmap.org for the "
                "area on screen.\n\nThat sends the area you are looking at - "
                "which is where this dataset was recorded - to their "
                "server, and needs a network. Tiles are cached on disk "
                "(shared with the device console's Track tab) and never "
                "fetched ahead of what is shown.")
            m.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes |
                                 QtWidgets.QMessageBox.StandardButton.Cancel)
            m.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Cancel)
            if m.exec() != QtWidgets.QMessageBox.StandardButton.Yes:
                self.map_on.blockSignals(True)
                self.map_on.setChecked(False)
                self.map_on.blockSignals(False)
                return
            if self.fetcher is None:
                self.fetcher = TileFetcher()
                self.fetcher.ready.connect(self._tile_ready)
                self.fetcher.start()
            self.note.setText(
                "Map data (C) OpenStreetMap contributors, "
                "openstreetmap.org/copyright - ODbL. Metres east and "
                "north of the first estimated position.")
            if self.origin_merc is not None:
                self._update_tiles()
        else:
            self._drop_tiles()

    def _drop_tiles(self):
        for item in self.tiles.values():
            self.plot.removeItem(item)
        self.tiles.clear()
        self.want = set()

    def _update_tiles(self):
        if self.origin_merc is None:
            return
        rect = self._view_rect()
        z = self._zoom_for(rect)
        x0, x1, y0, y1 = self._tile_range(rect, z)
        self.want = {(z, x, y)
                     for x in range(x0, x1 + 1)
                     for y in range(y0, y1 + 1)}
        # Whatever the view has left behind goes: the previous zoom level
        # covers the same ground at a different resolution and two of
        # those on top of each other is not a map, and a pan that keeps
        # every square it has ever shown is a leak with a picture on it.
        # The pixels stay in self.images, so coming back is free.
        for key in [k for k in self.tiles if k not in self.want]:
            self.plot.removeItem(self.tiles.pop(key))
        # Only what is on screen: no prefetch, no bulk.
        for key in sorted(self.want):
            if key in self.tiles:
                continue
            if key in self.images:
                self._show_tile(key)
            elif self.fetcher is not None:
                self.fetcher.want(*key)

    def _view_settled(self):
        if self.map_on.isChecked():
            self._update_tiles()

    def _tile_range(self, rect, z):
        """The inclusive tile index box a view rect covers.

        A QRectF counts y downwards and the view counts north upwards, so
        which corner gives which tile row depends on a convention that is
        easy to get backwards. Both corners, then min and max: the range
        is right whichever way round the rect came.
        """
        xa, ya = self._tile_of(rect.left(), rect.top(), z)
        xb, yb = self._tile_of(rect.right(), rect.bottom(), z)
        last = 2 ** z - 1
        x0, x1 = self._span(xa, xb, last)
        y0, y1 = self._span(ya, yb, last)
        return x0, x1, y0, y1

    @staticmethod
    def _span(a, b, last):
        """The inclusive tile index range two edges fall in, clamped to
        the world at this zoom."""
        lo = max(0, min(last, int(math.floor(min(a, b)))))
        hi = max(0, min(last, int(math.floor(max(a, b)))))
        return lo, hi

    def _zoom_for(self, rect):
        """The finest zoom whose visible tiles still fit the budget.

        Mercator metres, because that is what a tile measures - the view
        is in local metres, so the width is converted first. About two
        tiles across the view to begin with, then out a level at a time
        until the count fits: a tall window or a wide view then gets a
        coarser map instead of no map at all, which is what a hard cap on
        its own would give.
        """
        span = max(abs(rect.width()) / self.scale, 1.0)
        z = int(round(math.log2(2 * math.pi * EARTH_R * 2.0 / span)))
        z = max(1, min(19, z))
        while z > 1:
            x0, x1, y0, y1 = self._tile_range(rect, z)
            if (x1 - x0 + 1) * (y1 - y0 + 1) <= TILE_BUDGET:
                break
            z -= 1
        return z

    def _tile_of(self, east, north, z):
        mx, my = self._to_merc(east, north)
        n = 2 ** z
        half = math.pi * EARTH_R
        return ((mx + half) / (2 * half) * n,
                (half - my) / (2 * half) * n)

    def _tile_rect(self, z, x, y):
        """The tile's footprint in the plot's own metres."""
        n = 2 ** z
        half = math.pi * EARTH_R
        mx0 = x / n * 2 * half - half
        mx1 = (x + 1) / n * 2 * half - half
        my1 = half - y / n * 2 * half
        my0 = half - (y + 1) / n * 2 * half
        e0 = (mx0 - self.origin_merc[0]) * self.scale
        e1 = (mx1 - self.origin_merc[0]) * self.scale
        n0 = (my0 - self.origin_merc[1]) * self.scale
        n1 = (my1 - self.origin_merc[1]) * self.scale
        return QtCore.QRectF(e0, n0, e1 - e0, n1 - n0)

    def _tile_ready(self, z, x, y, data):
        if self.origin_merc is None:
            return
        img = QtGui.QImage()
        if not img.loadFromData(data):
            return
        key = (z, x, y)
        self._remember(key, _qimage_to_array(img))
        # The view may have moved on while this one was in flight. It is
        # kept either way and shown only if it is still wanted.
        if self.map_on.isChecked() and key in self.want:
            self._show_tile(key)

    def _remember(self, key, arr):
        """Keep the decoded pixels, bounded, on-screen ones kept last."""
        self.images.pop(key, None)
        self.images[key] = arr
        stale = [k for k in self.images if k not in self.tiles and k != key]
        for old in stale[:max(0, len(self.images) - self.TILE_KEEP)]:
            del self.images[old]

    def _show_tile(self, key):
        if key in self.tiles:
            return
        item = pg.ImageItem()
        item.setImage(self.images[key])
        item.setRect(self._tile_rect(*key))
        # Behind the tracks, and dimmed: a street map is context, and a
        # background at full contrast hides the two lines that matter.
        item.setZValue(-100)
        item.setOpacity(0.55)
        # ignoreBounds, or the tile just added enlarges the view, which asks
        # for tiles further out, which enlarge the view again: the map walks
        # away from the track it was fetched for.
        self.plot.addItem(item, ignoreBounds=True)
        self.tiles[key] = item

    def shutdown(self):
        if self.fetcher:
            self.fetcher.stop()
            self.fetcher.wait(2000)
            self.fetcher = None
