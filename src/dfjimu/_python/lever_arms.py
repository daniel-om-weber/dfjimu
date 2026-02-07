"""Pure Python fallback for lever arm estimation (vectorized with einsum)."""

import numpy as np


def estimate_lever_arms_cython(acc1, acc2, K1, K2, iterations, step, x0):
    """Gauss-Newton lever arm estimation from precomputed K matrices, returns (r1, r2)."""
    N = acc1.shape[0]
    x = x0.copy()

    for _ in range(iterations):
        # Residuals: acc - K @ x  (vectorized over N samples)
        e1 = acc1 - np.einsum('nij,j->ni', K1, x[:3])
        e2 = acc2 - np.einsum('nij,j->ni', K2, x[3:])

        n1 = np.linalg.norm(e1, axis=1)
        n2 = np.linalg.norm(e2, axis=1)
        eps = n1 - n2

        # Jacobian columns: d(||e||)/dx = -(K^T e) / ||e||
        inv_n1 = 1.0 / (n1 + 1e-9)
        inv_n2 = 1.0 / (n2 + 1e-9)

        # K^T @ e per sample: einsum('nji,ni->nj', K, e)
        J = np.empty((N, 6))
        J[:, :3] = -np.einsum('nji,ni->nj', K1, e1) * inv_n1[:, None]
        J[:, 3:] = np.einsum('nji,ni->nj', K2, e2) * inv_n2[:, None]

        H = J.T @ J
        G = J.T @ eps
        try:
            x -= step * np.linalg.solve(H + 1e-8 * np.eye(6), G)
        except np.linalg.LinAlgError:
            break

    return x[:3].copy(), x[3:].copy()
