import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
try:
    from ._cython.optimizer import build_system_cython, update_lin_points_cython
except ImportError:
    from ._python.optimizer import build_system_cython, update_lin_points_cython
from .utils.common import integrateGyr, preprocess_acc_at_center


def map_acc(gyr1, gyr2, acc1, acc2, r1, r2, Fs, q_init,
            cov_w, cov_i, cov_lnk, iterations=10):
    """
    Run the MAP-acc estimator (Weygers & Kok, 2020).

    Parameters:
    -----------
    gyr1, gyr2 : (N, 3) gyroscope data
    acc1, acc2 : (N, 3) accelerometer data
    r1, r2 : (3,) position vectors of sensors from joint center
    Fs : float, sampling frequency
    q_init : (4,) initial quaternion [w, x, y, z]
    cov_w : (6, 6) gyroscope noise covariance
    cov_i : (3, 3) inclination noise covariance
    cov_lnk : (3, 3) link constraint covariance
    iterations : int, max Gauss-Newton iterations (default 10)

    Returns:
    --------
    q1, q2 : (N, 4) estimated orientation quaternions
    """
    solver = MapAcc(gyr1, gyr2, acc1, acc2, r1, r2, Fs, cov_w, cov_i, cov_lnk)
    return solver.solve(q_init, iterations=iterations)


class MapAcc:
    """
    MAP-acc estimator (Weygers & Kok, 2020).

    Uses Gauss-Newton optimization on a sparse system.
    """
    def __init__(self, gyr1, gyr2, acc1, acc2, r1, r2, Fs, cov_w, cov_i, cov_lnk):
        self.gyr1 = gyr1
        self.gyr2 = gyr2
        self.acc1 = acc1
        self.acc2 = acc2
        self.r1 = r1
        self.r2 = r2
        self.Fs = Fs
        self.T = 1.0 / Fs

        # Precompute inv sqrt covs
        def inv_sqrt(cov):
            if np.isscalar(cov) or cov.size == 1:
                return np.eye(3) * (1.0/np.sqrt(cov))
            s, u = np.linalg.eigh(cov)
            return u @ np.diag(1.0/np.sqrt(s)) @ u.T

        self.icov_w1 = inv_sqrt(cov_w[:3, :3])
        self.icov_w2 = inv_sqrt(cov_w[3:6, 3:6])
        self.icov_i = inv_sqrt(cov_i)
        self.icov_lnk = inv_sqrt(cov_lnk)

        # Preprocessing
        self.C1 = preprocess_acc_at_center(gyr1, acc1, r1, self.Fs)
        self.C2 = preprocess_acc_at_center(gyr2, acc2, r2, self.Fs)

    def solve(self, q_init, iterations=10):
        # Init points
        q_lin_s1 = integrateGyr(self.gyr1, q_init, self.T)
        q_lin_s2 = integrateGyr(self.gyr2, q_init, self.T)

        N = self.gyr1.shape[0]
        n_rows = 9 * N
        n_cols = 6 * N

        for k in range(iterations):
            # Cython Assembly
            epsilon, j_data, j_rows, j_cols = build_system_cython(
                q_lin_s1, q_lin_s2, self.gyr1, self.gyr2, self.C1, self.C2,
                q_init, self.Fs, self.icov_w1, self.icov_w2, self.icov_i, self.icov_lnk
            )

            # Solve Sparse System
            J = coo_matrix((j_data, (j_rows, j_cols)), shape=(n_rows, n_cols))

            # Normal Equations: J.T @ J @ n = -J.T @ epsilon
            Jt = J.T
            H = Jt @ J
            g = Jt @ epsilon

            # Solve H n = -g
            n = spsolve(H, -g)

            # Update
            n_s1 = n[0:3*N].reshape((N, 3))
            n_s2 = n[3*N:].reshape((N, 3))

            # Cython Update
            q_lin_s1 = update_lin_points_cython(q_lin_s1, n_s1)
            q_lin_s2 = update_lin_points_cython(q_lin_s2, n_s2)

            if np.linalg.norm(n) < 1e-4: # Simple convergence
                break

        return q_lin_s1, q_lin_s2
