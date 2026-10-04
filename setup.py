"""Native-bearing wheels must carry the interpreter/platform compatibility tag."""
from pathlib import Path
from setuptools import Distribution, setup


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        root = Path(__file__).parent / 'src/route_rl'
        return any(root.rglob('*.so'))


setup(distclass=NativeDistribution)
