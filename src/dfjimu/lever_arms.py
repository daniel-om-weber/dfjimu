"""Public API for lever arm estimation from dual-IMU data."""

import numpy as np

try:
    from ._cython.lever_arms import estimate_lever_arms_cython
except ImportError:
    from ._python.lever_arms import estimate_lever_arms_cython

from .utils.common import approxDerivative, calculateK


def estimate_lever_arms(gyr1, gyr2, acc1, acc2, Fs, iterations=25, step=0.7,
                        x0=None):
    """
    Estimate lever arm vectors from dual-IMU data.

    Parameters:
    -----------
    gyr1, gyr2 : (N, 3) gyroscope data
    acc1, acc2 : (N, 3) accelerometer data
    Fs : float, sampling frequency in Hz
    iterations : int, Gauss-Newton iterations (default 25)
    step : float, step size for updates (default 0.7)
    x0 : (6,) initial estimate, or None for 0.1 * ones(6)

    Returns:
    --------
    r1, r2 : (3,) estimated lever arm vectors
    """
    gyr1 = np.asarray(gyr1, dtype=np.float64)
    gyr2 = np.asarray(gyr2, dtype=np.float64)
    acc1 = np.asarray(acc1, dtype=np.float64)
    acc2 = np.asarray(acc2, dtype=np.float64)

    if x0 is None:
        x0 = 0.1 * np.ones(6)
    else:
        x0 = np.asarray(x0, dtype=np.float64)
        if x0.shape != (6,):
            raise ValueError(f"x0 must have shape (6,), got {x0.shape}")

    # Precompute K matrices (N, 3, 3) via existing utilities
    dgyr1 = np.column_stack([approxDerivative(gyr1[:, i], Fs) for i in range(3)])
    dgyr2 = np.column_stack([approxDerivative(gyr2[:, i], Fs) for i in range(3)])
    K1 = calculateK(gyr1, dgyr1)
    K2 = calculateK(gyr2, dgyr2)

    return estimate_lever_arms_cython(acc1, acc2, K1, K2, iterations, step, x0)
