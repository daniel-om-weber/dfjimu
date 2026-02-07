import numpy as np

try:
    from ._cython.core import run_mekf_cython
except ImportError:
    from ._python.core import run_mekf_cython

from .utils.common import preprocess_acc_at_center


def mekf_acc(gyr1, gyr2, acc1, acc2, r1, r2, Fs, q_init, Q_cov=None,
             R_diag=0.011552, P_init_diag=0.1225):
    """
    Run the MEKF-acc filter (Weygers & Kok, 2020).

    Parameters:
    -----------
    gyr1, gyr2 : (N, 3) gyroscope data
    acc1, acc2 : (N, 3) accelerometer data
    r1, r2 : (3,) position vectors of sensors from joint center
    Fs : float, sampling frequency in Hz
    q_init : (4,) initial quaternion [w, x, y, z]
    Q_cov : (6,) gyroscope noise covariance, or None to auto-estimate
        from the first 50 samples of each gyroscope signal
    R_diag : float, measurement noise variance (default 0.011552)
    P_init_diag : float, initial error covariance diagonal (default 0.1225)

    Returns:
    --------
    q1, q2 : (N, 4) estimated orientation quaternions
    """
    C1 = preprocess_acc_at_center(gyr1, acc1, r1, Fs)
    C2 = preprocess_acc_at_center(gyr2, acc2, r2, Fs)

    if Q_cov is None:
        limit = min(50, gyr1.shape[0])
        var1 = np.var(gyr1[:limit], axis=0)
        var2 = np.var(gyr2[:limit], axis=0)
        Q_cov = np.concatenate([var1, var2])

    return run_mekf_cython(
        gyr1.astype(np.float64),
        gyr2.astype(np.float64),
        acc1.astype(np.float64),
        acc2.astype(np.float64),
        C1.astype(np.float64),
        C2.astype(np.float64),
        float(Fs),
        q_init.astype(np.float64),
        Q_cov.astype(np.float64),
        float(R_diag),
        float(P_init_diag),
    )


class MekfAcc:
    """
    MEKF-acc estimator (Weygers & Kok, 2020).

    Stores sensor data and configuration, delegates to mekf_acc().
    """

    def __init__(self, gyr1, gyr2, acc1, acc2, r1, r2, Fs, Q_cov=None,
                 R_diag=0.011552, P_init_diag=0.1225):
        self.gyr1 = gyr1
        self.gyr2 = gyr2
        self.acc1 = acc1
        self.acc2 = acc2
        self.r1 = r1
        self.r2 = r2
        self.Fs = Fs
        self.Q_cov = Q_cov
        self.R_diag = R_diag
        self.P_init_diag = P_init_diag

    def solve(self, q_init):
        return mekf_acc(
            self.gyr1, self.gyr2, self.acc1, self.acc2,
            self.r1, self.r2, self.Fs, q_init, self.Q_cov,
            self.R_diag, self.P_init_diag,
        )
