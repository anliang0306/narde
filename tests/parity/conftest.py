"""pytest path setup: make both `narde` (src/) and `laya` (reference oracle) importable."""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "reference", "laya")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
