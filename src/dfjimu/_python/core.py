"""Pure Python fallback for core.pyx (MEKF filter)."""

import numpy as np
from ..utils.common import EXPq, EXPr, quat2matrix, crossM, quatmultiply


def run_mekf_cython(gyr1, gyr2, acc1, acc2, C1, C2, Fs, q_init, Q_in,
                    R_diag=0.011552, P_init_diag=0.1225):
    """
    Pure Python MEKF filter matching the Cython implementation signature.

    Parameters
    ----------
    gyr1, gyr2 : (N, 3) gyroscope data
    acc1, acc2 : (N, 3) accelerometer data (unused, kept for API compat)
    C1, C2 : (N, 3) acceleration at joint center
    Fs : float, sampling frequency
    q_init : (4,) initial quaternion [w, x, y, z]
    Q_in : (6,) diagonal process noise

    Returns
    -------
    q_s1_out, q_s2_out : (N, 4) estimated orientations
    """
    N = gyr1.shape[0]
    dt = 1.0 / Fs
    dt_half = 0.5 * dt

    q_s1_out = np.zeros((N, 4))
    q_s2_out = np.zeros((N, 4))

    q1_curr = q_init.copy()
    q2_curr = q_init.copy()
    q_s1_out[0] = q_init
    q_s2_out[0] = q_init

    # Covariance P (6x6)
    P = np.eye(6) * P_init_diag

    # Process noise
    GQGt = dt * dt * Q_in.copy()
    GQGt[GQGt < 1e-12] = 1e-8

    R_val = R_diag

    for t in range(1, N):
        # --- Time Update ---
        # F matrix (block diagonal rotation matrices)
        R_rot1 = EXPr(-dt * gyr1[t - 1])
        R_rot2 = EXPr(-dt * gyr2[t - 1])

        F = np.zeros((6, 6))
        F[:3, :3] = R_rot1
        F[3:, 3:] = R_rot2

        # Predict quaternions
        dq1 = EXPq(dt_half * gyr1[t - 1])
        q1_pred = quatmultiply(q1_curr, dq1)

        dq2 = EXPq(dt_half * gyr2[t - 1])
        q2_pred = quatmultiply(q2_curr, dq2)

        # Predict covariance
        P = F @ P @ F.T + np.diag(GQGt)

        # --- Measurement Update ---
        R1 = quat2matrix(q1_pred)
        R2 = quat2matrix(q2_pred)

        # Innovation
        e = R1 @ C1[t] - R2 @ C2[t]

        # H matrix
        H = np.zeros((3, 6))
        H[:, :3] = R1 @ crossM(C1[t])
        H[:, 3:] = -R2 @ crossM(C2[t])

        # Innovation covariance
        S = H @ P @ H.T + R_val * np.eye(3)

        # Kalman gain
        invS = np.linalg.inv(S)
        K = P @ H.T @ invS

        # State update
        n = K @ e

        # Covariance update
        P_tilde = P - K @ S @ K.T

        # --- Relinearize ---
        dq1 = EXPq(0.5 * n[:3])
        q1_curr = quatmultiply(q1_pred, dq1)

        dq2 = EXPq(0.5 * n[3:])
        q2_curr = quatmultiply(q2_pred, dq2)

        # J matrix for covariance relinearization
        J = np.zeros((6, 6))
        J[:3, :3] = EXPr(-n[:3])
        J[3:, 3:] = EXPr(-n[3:])

        P = J @ P_tilde @ J.T

        # Normalize quaternions
        q1_curr = q1_curr / np.linalg.norm(q1_curr)
        q2_curr = q2_curr / np.linalg.norm(q2_curr)

        q_s1_out[t] = q1_curr
        q_s2_out[t] = q2_curr

    return q_s1_out, q_s2_out
