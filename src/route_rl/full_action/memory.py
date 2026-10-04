"""Conservative host-memory availability, including Linux container limits."""
from __future__ import annotations

from pathlib import Path


def host_memory_status(proc: Path = Path("/proc"), cgroup: Path = Path("/sys/fs/cgroup")) -> dict:
    available = None
    if (proc / "meminfo").exists():
        for line in (proc / "meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
    candidates = [(cgroup, "memory.max", "memory.current"),
                  (cgroup / "memory", "memory.limit_in_bytes", "memory.usage_in_bytes")]
    membership = proc / "self/cgroup"
    if membership.exists():
        for line in membership.read_text().splitlines():
            _, controllers, path = line.split(":", 2)
            if controllers == "":
                root, limit, usage = cgroup, "memory.max", "memory.current"
            elif "memory" in controllers.split(","):
                root, limit, usage = cgroup / "memory", "memory.limit_in_bytes", "memory.usage_in_bytes"
            else:
                continue
            current = root / path.lstrip("/")
            while current != root:
                candidates.append((current, limit, usage))
                current = current.parent
    limits, seen = [], set()
    for root, limit_file, usage_file in candidates:
        path = root / limit_file
        if path in seen or not path.is_file() or not (root / usage_file).is_file():
            continue
        seen.add(path)
        value = path.read_text().strip()
        if value == "max":
            continue
        limit = int(value)
        if limit >= 2**60:  # Linux v1's effectively unlimited sentinel.
            continue
        used = int((root / usage_file).read_text().strip())
        limits.append({"path": str(path), "limit_bytes": limit, "used_bytes": used,
                       "available_bytes": max(0, limit - used)})
    choices = [row["available_bytes"] for row in limits]
    if available is not None:
        choices.append(available)
    return {"host_available_bytes": available, "cgroup_limits": limits,
            "effective_available_bytes": min(choices) if choices else None}
