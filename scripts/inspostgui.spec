# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for tools/inspostgui.py (the post-processing GUI) as a
# self-contained Windows folder, no Python installation needed.
#
#   make pylib                       # the bundle ships python/INSLIB/libINSLIB.dll
#   pip install -r scripts/requirements-exe.txt
#   make inspostgui-exe              # or: pyinstaller scripts/inspostgui.spec ...
#
# One folder, not one file: a onefile exe unpacks itself on every start and
# is flagged by virus scanners far more often.

import os

REPO = os.path.dirname(os.path.abspath(SPECPATH))

a = Analysis(
    [os.path.join(REPO, 'tools', 'inspostgui.py')],
    # inspostgui.py puts these on sys.path itself at run time, PyInstaller's
    # static analysis needs to be told.
    pathex=[os.path.join(REPO, 'tools'),
            os.path.join(REPO, 'python'),
            os.path.join(REPO, 'datasets')],
    binaries=[(os.path.join(REPO, 'python', 'INSLIB', 'libINSLIB.dll'),
               'INSLIB')],
    datas=[(os.path.join(REPO, 'doc', 'figures', 'inslib_logo.png'),
            os.path.join('doc', 'figures')),
           (os.path.join(REPO, 'tools', 'kml_assets'), 'kml_assets')],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=['tkinter', 'PyQt5', 'PySide2', 'PySide6', 'IPython', 'pytest'],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='inspostgui',
    console=False,          # a GUI: no console window behind it
    icon=os.path.join(REPO, 'doc', 'figures', 'inslib_logo.png'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name='inspostgui',
)
