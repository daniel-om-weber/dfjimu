"""Smoke tests for dfjimu — lightweight CI-friendly tests using synthetic data."""

import numpy as np
import pytest

import dfjimu
from dfjimu import mekf_acc, map_acc, _CYTHON_AVAILABLE


def _synthetic_data(N=100, Fs=100.0):
    """Generate minimal synthetic IMU data for smoke testing."""
    rng = np.random.default_rng(42)
    gyr1 = rng.standard_normal((N, 3)) * 0.1
    gyr2 = rng.standard_normal((N, 3)) * 0.1
    # Accelerometers: gravity + small noise
    acc1 = np.tile([0.0, 0.0, 9.81], (N, 1)) + rng.standard_normal((N, 3)) * 0.05
    acc2 = np.tile([0.0, 0.0, 9.81], (N, 1)) + rng.standard_normal((N, 3)) * 0.05
    r1 = np.array([0.1, 0.0, 0.0])
    r2 = np.array([-0.1, 0.0, 0.0])
    q_init = np.array([1.0, 0.0, 0.0, 0.0])
    return gyr1, gyr2, acc1, acc2, r1, r2, Fs, q_init


def test_version():
    assert hasattr(dfjimu, '__version__')
    assert isinstance(dfjimu.__version__, str)


def test_cython_flag():
    assert isinstance(_CYTHON_AVAILABLE, bool)


def test_mekf_acc():
    gyr1, gyr2, acc1, acc2, r1, r2, Fs, q_init = _synthetic_data()
    N = gyr1.shape[0]

    q1, q2 = mekf_acc(gyr1, gyr2, acc1, acc2, r1, r2, Fs, q_init)

    assert q1.shape == (N, 4)
    assert q2.shape == (N, 4)
    # Quaternions should be approximately unit norm
    norms1 = np.linalg.norm(q1, axis=1)
    norms2 = np.linalg.norm(q2, axis=1)
    np.testing.assert_allclose(norms1, 1.0, atol=1e-6)
    np.testing.assert_allclose(norms2, 1.0, atol=1e-6)


def test_map_acc():
    gyr1, gyr2, acc1, acc2, r1, r2, Fs, q_init = _synthetic_data()
    N = gyr1.shape[0]

    cov_w = np.eye(6) * 1e-4
    cov_i = np.eye(3) * 1e-2
    cov_lnk = np.eye(3) * 1e-2

    q1, q2 = map_acc(gyr1, gyr2, acc1, acc2, r1, r2, Fs, q_init,
                     cov_w, cov_i, cov_lnk, iterations=3)

    assert q1.shape == (N, 4)
    assert q2.shape == (N, 4)
    # Quaternions should be approximately unit norm
    norms1 = np.linalg.norm(q1, axis=1)
    norms2 = np.linalg.norm(q2, axis=1)
    np.testing.assert_allclose(norms1, 1.0, atol=1e-6)
    np.testing.assert_allclose(norms2, 1.0, atol=1e-6)
