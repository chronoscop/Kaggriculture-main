"""Submission archive limits and safe extraction for the local economic pipeline."""
from pathlib import Path, PurePosixPath
import tarfile

MAX_ARCHIVE = 100 * 1024 ** 2
MAX_DISK = 8 * 1024 ** 3
MAX_RAM = int(6.5 * 1024 ** 3)

def unpack(archive, root):
    archive, root = Path(archive), Path(root)
    if archive.stat().st_size > MAX_ARCHIVE:
        raise ValueError("archive exceeds 100 MiB")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        total = sum(m.size for m in members)
        names = [m.name for m in members]
        if total > MAX_DISK or len(names) != len(set(names)):
            raise ValueError("oversized archive or duplicate entries")
        if "main.py" not in names:
            raise ValueError("main.py must be at the archive root")
        for member in members:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError("archive must contain only safe relative regular files")
            target = root.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(member).read())
    return total
