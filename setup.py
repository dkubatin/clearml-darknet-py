import sys
from setuptools import setup


# Check python version
MINIMAL_PY_VERSION = (3, 6)
if sys.version_info < MINIMAL_PY_VERSION:
    raise RuntimeError(f"Minimum version of Python {'.'.join(map(str, MINIMAL_PY_VERSION))}+ required")


setup()
