"""Shared pytest fixtures for the eddy smoke suite.

The tutorial FITS files in ``docs/tutorials/`` are reused as test
fixtures so the suite has no extra data dependency. All fixtures are
session-scoped because loading a cube takes several seconds and JAX
compilation dominates wall time once `disk_coords` is exercised.
"""

import os

import pytest


HERE = os.path.dirname(os.path.abspath(__file__))
TUTORIAL_DATA = os.path.join(HERE, os.pardir, "docs", "tutorials")


def _tutorial_path(name):
    path = os.path.normpath(os.path.join(TUTORIAL_DATA, name))
    if not os.path.exists(path):
        pytest.skip("Tutorial fixture {} not present".format(name))
    return path


@pytest.fixture(scope="session")
def hd163296_v0_path():
    return _tutorial_path("HD163296_CO_v0.fits")


@pytest.fixture(scope="session")
def hd163296_dv0_path():
    return _tutorial_path("HD163296_CO_dv0.fits")


@pytest.fixture(scope="session")
def twhya_cube_path():
    return _tutorial_path("TWHya_CO_cube.fits")


@pytest.fixture(scope="session")
def twhya_v0_path():
    return _tutorial_path("TWHya_CO_cube_v0.fits")


@pytest.fixture(scope="session")
def hd163296_rotationmap(hd163296_v0_path, hd163296_dv0_path):
    """A small, downsampled rotationmap for fit_map smoke tests."""
    from eddy import rotationmap
    return rotationmap(path=hd163296_v0_path,
                       uncertainty=hd163296_dv0_path,
                       FOV=8.0, downsample=4)


@pytest.fixture(scope="session")
def twhya_linecube(twhya_cube_path):
    """A small linecube for Annulus3D / get_velocity_profile smoke tests."""
    from eddy import linecube
    return linecube(path=twhya_cube_path, FOV=4.0)
