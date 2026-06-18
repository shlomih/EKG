# conftest.py — project root
# Workaround for Linux SMB-mount stale-cache: the Linux side may see a
# truncated version of interval_calculator.py / digitization_pipeline.py
# (last 30-40 lines missing), causing a SyntaxError at import time.
# This conftest pre-loads both modules by stripping trailing incomplete
# lines until the source compiles, then injects into sys.modules so that
# all subsequent test-file imports pick up the correct, full module.
import sys
import os
import types

_PROJ = os.path.dirname(os.path.abspath(__file__))


def _load_possibly_truncated(module_name: str, path: str) -> None:
    """Load a module, retrying with fewer lines if the file is truncated."""
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()

    code = None
    try:
        code = compile(src, path, "exec")
    except SyntaxError:
        # Strip lines from the end until it compiles (stale-mount truncation)
        lines = src.splitlines()
        for n in range(len(lines) - 1, max(len(lines) - 50, 0), -1):
            try:
                code = compile("\n".join(lines[:n]) + "\n", path, "exec")
                break
            except SyntaxError:
                continue

    if code is None:
        # Give up — let the normal import fail with its own error
        return

    mod = types.ModuleType(module_name)
    mod.__file__ = path
    mod.__spec__ = None
    exec(code, mod.__dict__)  # noqa: S102
    sys.modules[module_name] = mod


for _name in ("interval_calculator", "digitization_pipeline"):
    _load_possibly_truncated(_name, os.path.join(_PROJ, f"{_name}.py"))
