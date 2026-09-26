"""Archive extraction checks; no simulator or training required."""
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from route_rl.local.archive import unpack

class LocalArchiveTests(unittest.TestCase):
    def make_archive(self, path, extra=None):
        with tarfile.open(path, "w:gz") as tar:
            main = tarfile.TarInfo("main.py")
            main.size = 4
            tar.addfile(main, io.BytesIO(b"pass"))
            if extra is not None:
                tar.addfile(extra, io.BytesIO(b""))

    def test_extracts_root_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, root = Path(tmp)/"agent.tar.gz", Path(tmp)/"agent"
            self.make_archive(archive)
            self.assertEqual(unpack(archive, root), 4)
            self.assertEqual((root/"main.py").read_bytes(), b"pass")

    def test_rejects_unsafe_and_duplicate_entries(self):
        for name, kind in [("../escape.py", tarfile.REGTYPE),
                           ("/escape.py", tarfile.REGTYPE),
                           ("link", tarfile.SYMTYPE),
                           ("main.py", tarfile.REGTYPE)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                archive, root = Path(tmp)/"agent.tar.gz", Path(tmp)/"agent"
                extra = tarfile.TarInfo(name)
                extra.type = kind
                extra.linkname = "/tmp"
                self.make_archive(archive, extra)
                with self.assertRaises(ValueError):
                    unpack(archive, root)
                self.assertFalse((Path(tmp)/"escape.py").exists())

if __name__ == "__main__":
    unittest.main()
