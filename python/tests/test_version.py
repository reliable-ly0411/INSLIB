"""INSLIB.__version__ (the PyPI package version) must match the release.

CLAUDE.md release step 2 bumps the version in CITATION.cff, sbom.cdx.json
and python/INSLIB/__init__.py together. This catches a forgotten one
before a wheel with a stale version number reaches PyPI.

(c) Jan Zwiener (jan@zwiener.org)
"""

import json
import os
import re
import sys

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(_ROOT, "python"))

import INSLIB  # noqa: E402


def _citation_version():
    with open(os.path.join(_ROOT, "CITATION.cff"), encoding="utf-8") as fh:
        m = re.search(r"^version:\s*['\"]?([0-9][^'\"\s]*)", fh.read(), re.M)
    assert m, "CITATION.cff has no version line"
    return m.group(1)


def _sbom_version():
    with open(os.path.join(_ROOT, "sbom.cdx.json"), encoding="utf-8") as fh:
        return json.load(fh)["metadata"]["component"]["version"]


def test_package_version_matches_release():
    assert INSLIB.__version__ == _citation_version()
    assert INSLIB.__version__ == _sbom_version()


if __name__ == "__main__":
    test_package_version_matches_release()
    print("OK")
