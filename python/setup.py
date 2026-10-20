"""Build hook of the INSLIB wheel: compiles libINSLIB next to the package.

All metadata lives in pyproject.toml. This file only adds what setuptools
cannot express there: the shared library that ``_core.py`` loads with
ctypes. It is a plain C library (no Python C API), so it is built with a
direct compiler call instead of a setuptools Extension, and the wheel is
tagged ``py3-none-<platform>``: one wheel per platform serves every
Python version.

The source list and the KFCore limits are read from the top-level
Makefile (SUITE_SRC, KFCORE_SRC, KFCORE_LIMITS), so the wheel and
``make pylib`` cannot drift apart. That needs the full repository
checkout, which is why only wheels are published, no sdist.

Compiler: ``$CC`` (gcc/clang style) if set, else cc on Linux/macOS. On
Windows MinGW-w64 gcc for x64 when it is on PATH, otherwise MSVC (always
for ARM64, which has no MinGW gcc on the runners). MSVC exports only what
it is told, so that build passes the export list from ``_msvc_exports()``.

(c) Jan Zwiener (jan@zwiener.org)
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig

from setuptools import Command, setup
from setuptools.command.build import build
from setuptools.dist import Distribution

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # setuptools < 70.1 ships it in the wheel package
    from wheel.bdist_wheel import bdist_wheel

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

if sys.platform == "win32":
    _LIBNAME = "libINSLIB.dll"
elif sys.platform == "darwin":
    _LIBNAME = "libINSLIB.dylib"
else:
    _LIBNAME = "libINSLIB.so"


def _makefile_vars(*names):
    """Read single-line ``NAME := value`` assignments from the Makefile."""
    path = os.path.join(_ROOT, "Makefile")
    if not os.path.exists(path):
        raise RuntimeError(
            "INSLIB's Makefile not found at %s. The wheel builds from the "
            "full repository checkout (python/ inside it)." % path)
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    values = {}
    for name in names:
        m = re.search(r"^%s\s*:=\s*(.*)$" % re.escape(name), text, re.M)
        if not m:
            raise RuntimeError("Makefile has no '%s :=' line" % name)
        values[name] = m.group(1).split()
    return values


_MSVC_EXTERNAL_FN = re.compile(
    r"\bSECT[0-9A-F]+\s+notype \(\)\s+External\s+\|\s+(\S+)")


def _msvc_exports(linker, objs):
    """Every external function the objects define, like gcc exports them.

    ``link /dump /symbols`` is dumpbin. A list derived from the Python
    callers would miss the tools in tools/, which call into the library
    through ``INSLIB._core._lib`` too.
    """
    names = set()
    for obj in objs:
        out = subprocess.run([linker, "/dump", "/symbols", obj], check=True,
                             capture_output=True, text=True).stdout
        names.update(_MSVC_EXTERNAL_FN.findall(out))
    # A format change of the dump must fail the build, not ship a DLL
    # without exports.
    if "ins_suite_create" not in names:
        raise RuntimeError("no INSLIB exports found in the MSVC objects")
    return sorted(names)


def _use_msvc(cc_env):
    if sys.platform != "win32" or cc_env:
        return False
    if sysconfig.get_platform() != "win-amd64":
        return True
    return shutil.which("gcc") is None


class build_inslib(Command):
    """Compile the C library into the package (``build_lib`` or in place)."""

    description = "compile libINSLIB (ctypes shared library)"
    user_options = []

    def initialize_options(self):
        self.build_lib = None
        self.build_temp = None
        self.editable_mode = False

    def finalize_options(self):
        self.set_undefined_options("build", ("build_lib", "build_lib"),
                                   ("build_temp", "build_temp"))

    def _target(self):
        if self.editable_mode:
            return os.path.join(_HERE, "INSLIB", _LIBNAME)
        return os.path.join(self.build_lib, "INSLIB", _LIBNAME)

    def _license(self):
        return os.path.join(os.path.dirname(self._target()), "LICENSE")

    def get_outputs(self):
        if self.editable_mode:
            return [self._target()]
        return [self._target(), self._license()]

    def get_output_mapping(self):
        return {}

    def get_source_files(self):
        return []

    @staticmethod
    def _includes():
        incs = ["src", "KFCore/c", "KFCore/c/navigation_tools", "KFCore/tests"]
        return [os.path.join(_HERE, "csrc")] + \
            [os.path.join(_ROOT, p) for p in incs]

    def _build_gcc(self, cc, srcs, limits, target):
        if shutil.which(cc[0]) is None:
            raise RuntimeError(
                "C compiler '%s' not found. Set CC or install gcc/clang."
                % cc[0])
        cmd = cc + ["-std=c11", "-O2", "-D_GNU_SOURCE", "-fPIC", "-shared"]
        cmd += limits
        cmd += ["-I" + p for p in self._includes()]
        cmd += srcs
        if sys.platform == "win32":
            # No libgcc DLL dependency next to the wheel's DLL.
            cmd.append("-static-libgcc")
        cmd += ["-lm", "-o", target]
        self.announce(" ".join(cmd), level=2)
        subprocess.check_call(cmd)

    def _build_msvc(self, srcs, limits, target):
        # setuptools' compiler driver locates MSVC (vswhere) and targets the
        # architecture of the running interpreter.
        from distutils.ccompiler import new_compiler

        msvc = new_compiler(compiler="msvc")
        msvc.initialize()
        # No whole-program optimization (/GL): its objects carry no symbol
        # table for _msvc_exports() to read.
        msvc.compile_options = [o for o in msvc.compile_options
                                if o != "/GL"]
        macros = [("_USE_MATH_DEFINES", None)]
        for flag in limits:
            name, _, value = flag[2:].partition("=")
            macros.append((name, value or None))
        tmp = os.path.join(self.build_temp, "inslib_msvc")
        objs = msvc.compile(srcs, output_dir=tmp, macros=macros,
                            include_dirs=self._includes(),
                            extra_preargs=["/std:c11", "/wd4068"])
        msvc.link(msvc.SHARED_OBJECT, objs, os.path.basename(target),
                  output_dir=os.path.dirname(target),
                  export_symbols=_msvc_exports(msvc.linker, objs),
                  build_temp=tmp)

    def run(self):
        mk = _makefile_vars("SUITE_SRC", "KFCORE_SRC", "KFCORE_LIMITS")
        srcs = [os.path.join(_ROOT, p) for p in
                mk["SUITE_SRC"] + mk["KFCORE_SRC"]]
        srcs.append(os.path.join(_HERE, "csrc", "ins_capi.c"))
        missing = [p for p in srcs if not os.path.exists(p)]
        if missing:
            raise RuntimeError(
                "missing C sources (KFCore submodule not initialized? run "
                "git submodule update --init):\n  " + "\n  ".join(missing))

        cc_env = shlex.split(os.environ.get("CC", ""))
        limits = mk["KFCORE_LIMITS"]
        target = self._target()
        os.makedirs(os.path.dirname(target), exist_ok=True)
        try:
            if _use_msvc(cc_env):
                self._build_msvc(srcs, limits, target)
            else:
                default = "gcc" if sys.platform == "win32" else "cc"
                self._build_gcc(cc_env or [default], srcs, limits, target)
        except Exception as exc:
            if self.editable_mode and os.path.exists(target):
                self.warn("building libINSLIB failed (%s), keeping the "
                          "existing %s" % (exc, target))
                return
            raise
        if not self.editable_mode:
            # The AGPL text travels with every wheel. It lives at the
            # repository root, outside the project directory that PEP 639
            # license-files may point into.
            shutil.copyfile(os.path.join(_ROOT, "LICENSE"), self._license())


class build_with_lib(build):
    sub_commands = build.sub_commands + [("build_inslib", None)]


class bdist_wheel_platform(bdist_wheel):
    """Platform wheel without a Python ABI tag (ctypes, no C extension)."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = super().get_tag()
        return "py3", "none", plat


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    distclass=BinaryDistribution,
    cmdclass={
        "build": build_with_lib,
        "build_inslib": build_inslib,
        "bdist_wheel": bdist_wheel_platform,
    },
)
