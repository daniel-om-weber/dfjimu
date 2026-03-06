import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
try:
    from ._cython.optimizer import build_system_stable_cython, update_lin_points_cython
except ImportError:
    from ._python.optimizer import build_system_stable as build_system_stable_cython
    from ._python.optimizer import update_lin_points_cython
from .lever_arms import estimate_lever_arms
from .utils.common import integrateGyr, preprocess_acc_at_center


def map_acc_stable(gyr1, gyr2, acc1, acc2, r1=None, r2=None, Fs=None, q_init=None,
                   cov_w=None, cov_i=None, cov_lnk=None, iterations=10, tol=1e-4):
    """
    Run the MAP-acc estimator with numerically stable log_q (atan2-based).

    Same interface as map_acc(). Uses atan2 instead of acos in the quaternion
    logarithm, which avoids precision loss near identity quaternions (static
    segments, high sampling rates, near convergence).

    Parameters:
    -----------
    gyr1, gyr2 : (N, 3) gyroscope data
    acc1, acc2 : (N, 3) accelerometer data
    r1, r2 : (3,) position vectors of sensors from joint center, or None
        to auto-estimate via estimate_lever_arms()
    Fs : float, sampling frequency
    q_init : (4,) initial quaternion [w, x, y, z]
    cov_w : (6, 6) gyroscope noise covariance
    cov_i : (3, 3) inclination noise covariance
    cov_lnk : (3, 3) link constraint covariance
    iterations : int, max Gauss-Newton iterations (default 10)
    tol : float, convergence threshold on update norm (default 1e-4)

    Returns:
    --------
    q1, q2 : (N, 4) estimated orientation quaternions
    """
    solver = MapAccStable(gyr1, gyr2, acc1, acc2, r1, r2, Fs, cov_w, cov_i, cov_lnk)
    return solver.solve(q_init, iterations=iterations, tol=tol)


class MapAccStable:
    """
    MAP-acc estimator with numerically stable log_q (atan2-based).

    Same interface as MapAcc.
    """
    def __init__(self, gyr1, gyr2, acc1, acc2, r1=None, r2=None, Fs=None,
                 cov_w=None, cov_i=None, cov_lnk=None):
        self.gyr1 = gyr1
        self.gyr2 = gyr2
        self.acc1 = acc1
        self.acc2 = acc2
        if r1 is None or r2 is None:
            r1, r2 = estimate_lever_arms(gyr1, gyr2, acc1, acc2, Fs)
        self.r1 = r1
        self.r2 = r2
        self.Fs = Fs
        self.T = 1.0 / Fs

        def inv_sqrt(cov):
            if np.isscalar(cov) or cov.size == 1:
                return np.eye(3) * (1.0/np.sqrt(cov))
            s, u = np.linalg.eigh(cov)
            return u @ np.diag(1.0/np.sqrt(s)) @ u.T

        self.icov_w1 = inv_sqrt(cov_w[:3, :3])
        self.icov_w2 = inv_sqrt(cov_w[3:6, 3:6])
        self.icov_i = inv_sqrt(cov_i)
        self.icov_lnk = inv_sqrt(cov_lnk)

        self.C1 = preprocess_acc_at_center(gyr1, acc1, r1, self.Fs)
        self.C2 = preprocess_acc_at_center(gyr2, acc2, r2, self.Fs)

    def solve(self, q_init, iterations=10, tol=1e-4):
        q_lin_s1 = integrateGyr(self.gyr1, q_init, self.T)
        q_lin_s2 = integrateGyr(self.gyr2, q_init, self.T)

        N = self.gyr1.shape[0]
        n_rows = 9 * N
        n_cols = 6 * N

        for k in range(iterations):
            epsilon, j_data, j_rows, j_cols = build_system_stable_cython(
                q_lin_s1, q_lin_s2, self.gyr1, self.gyr2, self.C1, self.C2,
                q_init, self.Fs, self.icov_w1, self.icov_w2, self.icov_i, self.icov_lnk
            )

            J = coo_matrix((j_data, (j_rows, j_cols)), shape=(n_rows, n_cols))

            Jt = J.T
            H = Jt @ J
            g = Jt @ epsilon

            n = spsolve(H, -g)

            n_s1 = n[0:3*N].reshape((N, 3))
            n_s2 = n[3*N:].reshape((N, 3))

            q_lin_s1 = update_lin_points_cython(q_lin_s1, n_s1)
            q_lin_s2 = update_lin_points_cython(q_lin_s2, n_s2)

            if np.linalg.norm(n) < tol:
                break

        return q_lin_s1, q_lin_s2
