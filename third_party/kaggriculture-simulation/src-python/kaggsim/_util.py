"""Small shared helpers (internal)."""
from __future__ import annotations

import importlib
import re

_RANGE = re.compile(r"^\s*(-?\d+)\s*(?:-|\.\.)\s*(-?\d+)\s*$")


def parse_seeds(spec) -> list:
    """``"0-9,15,20..22,-3"`` -> seeds (duplicates removed, order kept).

    Ranges are inclusive and written ``lo-hi`` or ``lo..hi``; a lone
    negative number is a single seed.
    """
    if isinstance(spec, (list, tuple, range)):
        items = [int(s) for s in spec]
    else:
        items = []
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            m = _RANGE.match(part)
            if m:
                lo, hi = int(m.group(1)), int(m.group(2))
                if hi < lo:
                    raise ValueError(f"empty seed range {part!r}")
                items.extend(range(lo, hi + 1))
            else:
                try:
                    items.append(int(part))
                except ValueError:
                    raise ValueError(f"bad seed {part!r}") from None
    seen, out = set(), []
    for s in items:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def resolve_ref(ref: str):
    """``"package.module:attr.sub"`` -> the object."""
    if not isinstance(ref, str) or ":" not in ref:
        raise ValueError(f"reference {ref!r} must look like 'module:name'")
    mod, attr = ref.split(":", 1)
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj
