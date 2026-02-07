# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""Cython lever arm estimation via Gauss-Newton optimization."""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs

ctypedef np.float64_t DTYPE_t


cdef void mat3_vec_mul(double* A, double* x, double* y) nogil:
    """Compute y = A @ x for contiguous 3x3 A (row-major) and 3-vector x."""
    cdef int i, k
    for i in range(3):
        y[i] = 0.0
        for k in range(3):
            y[i] += A[i * 3 + k] * x[k]


cdef void mat3T_vec_mul(double* A, double* x, double* y) nogil:
    """Compute y = A^T @ x for contiguous 3x3 A (row-major) and 3-vector x."""
    cdef int i, k
    for i in range(3):
        y[i] = 0.0
        for k in range(3):
            y[i] += A[k * 3 + i] * x[k]


cdef int solve6x6(double H[6][6], double G[6], double dx[6]) nogil:
    """Solve H @ dx = G via Gaussian elimination with partial pivoting, returns 1 on success."""
    cdef double A[6][7]
    cdef double maxval, tmp
    cdef int i, j, k, pivot

    # Augmented matrix [H | G]
    for i in range(6):
        for j in range(6):
            A[i][j] = H[i][j]
        A[i][6] = G[i]

    # Forward elimination with partial pivoting
    for i in range(6):
        # Find pivot
        pivot = i
        maxval = fabs(A[i][i])
        for k in range(i + 1, 6):
            if fabs(A[k][i]) > maxval:
                maxval = fabs(A[k][i])
                pivot = k
        if maxval < 1e-15:
            return 0

        # Swap rows
        if pivot != i:
            for j in range(7):
                tmp = A[i][j]
                A[i][j] = A[pivot][j]
                A[pivot][j] = tmp

        # Eliminate below
        for k in range(i + 1, 6):
            tmp = A[k][i] / A[i][i]
            for j in range(i, 7):
                A[k][j] -= tmp * A[i][j]

    # Back substitution
    for i in range(5, -1, -1):
        dx[i] = A[i][6]
        for j in range(i + 1, 6):
            dx[i] -= A[i][j] * dx[j]
        dx[i] /= A[i][i]

    return 1


def estimate_lever_arms_cython(
    np.ndarray[DTYPE_t, ndim=2] acc1,
    np.ndarray[DTYPE_t, ndim=2] acc2,
    np.ndarray[DTYPE_t, ndim=3] K1,
    np.ndarray[DTYPE_t, ndim=3] K2,
    int iterations,
    double step,
    np.ndarray[DTYPE_t, ndim=1] x0,
):
    """Gauss-Newton lever arm estimation from precomputed K matrices, returns (r1, r2)."""
    cdef int N = acc1.shape[0]
    cdef int it, i, j, k, n

    cdef double x[6]
    for i in range(6):
        x[i] = x0[i]

    cdef double e1[3]
    cdef double e2[3]
    cdef double Kt_e1[3]
    cdef double Kt_e2[3]
    cdef double n1, n2, eps_i, inv_n1, inv_n2
    cdef double jrow[6]

    cdef double H[6][6]
    cdef double G[6]
    cdef double dx[6]

    for it in range(iterations):
        # Zero H and G
        for i in range(6):
            G[i] = 0.0
            for j in range(6):
                H[i][j] = 0.0

        for n in range(N):
            # e1 = acc1[n] - K1[n] @ x[:3]
            mat3_vec_mul(&K1[n, 0, 0], x, e1)
            for i in range(3):
                e1[i] = acc1[n, i] - e1[i]

            # e2 = acc2[n] - K2[n] @ x[3:]
            mat3_vec_mul(&K2[n, 0, 0], &x[3], e2)
            for i in range(3):
                e2[i] = acc2[n, i] - e2[i]

            n1 = sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2])
            n2 = sqrt(e2[0]*e2[0] + e2[1]*e2[1] + e2[2]*e2[2])
            eps_i = n1 - n2

            inv_n1 = 1.0 / (n1 + 1e-9)
            inv_n2 = 1.0 / (n2 + 1e-9)

            # Jacobian row: jrow[:3] = -(K1^T @ e1) / n1, jrow[3:] = (K2^T @ e2) / n2
            mat3T_vec_mul(&K1[n, 0, 0], e1, Kt_e1)
            mat3T_vec_mul(&K2[n, 0, 0], e2, Kt_e2)

            for i in range(3):
                jrow[i] = -Kt_e1[i] * inv_n1
                jrow[i + 3] = Kt_e2[i] * inv_n2

            # Accumulate H += jrow^T @ jrow (outer product) and G += jrow * eps
            for i in range(6):
                G[i] += jrow[i] * eps_i
                for j in range(6):
                    H[i][j] += jrow[i] * jrow[j]

        # Regularize
        for i in range(6):
            H[i][i] += 1e-8

        # Solve H @ dx = G
        if solve6x6(H, G, dx) == 0:
            break

        for i in range(6):
            x[i] -= step * dx[i]

    # Return as numpy arrays
    cdef np.ndarray[DTYPE_t, ndim=1] r1 = np.empty(3, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] r2 = np.empty(3, dtype=np.float64)
    for i in range(3):
        r1[i] = x[i]
        r2[i] = x[i + 3]

    return r1, r2
