"""Pure Python fallback for optimizer.pyx (GN sparse system builder)."""

import numpy as np
from ..utils.common import (
    quatmultiply, quatconj, LOGq, quat2matrix, crossM, quatL, quatR,
    update_linPoints,
)


def build_system_cython(q_lin_s1, q_lin_s2, gyr1, gyr2, C1, C2, q_init,
                         Fs, icov_w1, icov_w2, icov_i, icov_lnk):
    """
    Build error vector and sparse Jacobian (COO format) for GN optimization.

    Matches the Cython build_system_cython signature and output format.

    Returns
    -------
    epsilon : (9*N,) error vector
    j_data, j_rows, j_cols : sparse Jacobian in COO format
    """
    N = q_lin_s1.shape[0]
    dt = 1.0 / Fs

    n_err = 9 * N
    epsilon = np.zeros(n_err)

    j_data = []
    j_rows = []
    j_cols = []

    q_init_inv = quatconj(q_init)

    # --- Sensor 1: Init cost ---
    q_tmp = quatmultiply(q_init_inv, q_lin_s1[0])
    v_tmp = 2.0 * LOGq(q_tmp)
    epsilon[0:3] = icov_i @ v_tmp

    L = quatL(q_tmp)
    L_br = L[1:, 1:]  # bottom-right 3x3
    J_block = icov_i @ L_br
    _append_block(j_data, j_rows, j_cols, J_block, 0, 0)

    # --- Sensor 1: Motion cost ---
    row_offset = 3
    for t in range(1, N):
        q_prev = q_lin_s1[t - 1]
        q_curr = q_lin_s1[t]

        q_tmp = quatmultiply(quatconj(q_prev), q_curr)
        v_tmp = (2.0 / dt) * LOGq(q_tmp) - gyr1[t - 1]
        epsilon[row_offset + (t - 1) * 3: row_offset + t * 3] = icov_w1 @ v_tmp

        # Jacobian w.r.t. t-1: -1/dt * R[bottom-right]
        R_mat = quatR(q_tmp)
        R_br = R_mat[1:, 1:]
        J_tm1 = icov_w1 @ (-1.0 / dt * R_br)
        _append_block(j_data, j_rows, j_cols, J_tm1,
                      row_offset + (t - 1) * 3, (t - 1) * 3)

        # Jacobian w.r.t. t: 1/dt * L[bottom-right]
        L_mat = quatL(q_tmp)
        L_br = L_mat[1:, 1:]
        J_t = icov_w1 @ (1.0 / dt * L_br)
        _append_block(j_data, j_rows, j_cols, J_t,
                      row_offset + (t - 1) * 3, t * 3)

    # --- Sensor 2: Init cost ---
    row_offset_s2 = 3 * N
    col_offset = 3 * N

    q_tmp = quatmultiply(q_init_inv, q_lin_s2[0])
    v_tmp = 2.0 * LOGq(q_tmp)
    epsilon[row_offset_s2: row_offset_s2 + 3] = icov_i @ v_tmp

    L = quatL(q_tmp)
    L_br = L[1:, 1:]
    J_block = icov_i @ L_br
    _append_block(j_data, j_rows, j_cols, J_block, row_offset_s2, col_offset)

    # --- Sensor 2: Motion cost ---
    row_offset_s2 += 3
    for t in range(1, N):
        q_prev = q_lin_s2[t - 1]
        q_curr = q_lin_s2[t]

        q_tmp = quatmultiply(quatconj(q_prev), q_curr)
        v_tmp = (2.0 / dt) * LOGq(q_tmp) - gyr2[t - 1]
        epsilon[row_offset_s2 + (t - 1) * 3: row_offset_s2 + t * 3] = icov_w2 @ v_tmp

        R_mat = quatR(q_tmp)
        R_br = R_mat[1:, 1:]
        J_tm1 = icov_w2 @ (-1.0 / dt * R_br)
        _append_block(j_data, j_rows, j_cols, J_tm1,
                      row_offset_s2 + (t - 1) * 3, col_offset + (t - 1) * 3)

        L_mat = quatL(q_tmp)
        L_br = L_mat[1:, 1:]
        J_t = icov_w2 @ (1.0 / dt * L_br)
        _append_block(j_data, j_rows, j_cols, J_t,
                      row_offset_s2 + (t - 1) * 3, col_offset + t * 3)

    # --- Link constraint ---
    row_offset_lnk = 6 * N
    for t in range(N):
        R1 = quat2matrix(q_lin_s1[t])
        R2 = quat2matrix(q_lin_s2[t])

        err_lnk = R1 @ C1[t] - R2 @ C2[t]
        epsilon[row_offset_lnk + t * 3: row_offset_lnk + (t + 1) * 3] = icov_lnk @ err_lnk

        # J S1: -icov * R1 * [c1]x
        J_s1 = icov_lnk @ R1 @ crossM(C1[t])
        _append_block_neg(j_data, j_rows, j_cols, J_s1,
                          row_offset_lnk + t * 3, t * 3)

        # J S2: icov * R2 * [c2]x
        J_s2 = icov_lnk @ R2 @ crossM(C2[t])
        _append_block(j_data, j_rows, j_cols, J_s2,
                      row_offset_lnk + t * 3, 3 * N + t * 3)

    return (
        epsilon,
        np.array(j_data, dtype=np.float64),
        np.array(j_rows, dtype=np.int32),
        np.array(j_cols, dtype=np.int32),
    )


def update_lin_points_cython(q_lin, n):
    """Update linearization points. Wraps utils.common.update_linPoints."""
    return update_linPoints(q_lin, n)


def _append_block(data, rows, cols, block, row_start, col_start):
    """Append non-zero entries of a 3x3 block to COO lists."""
    for i in range(3):
        for j in range(3):
            val = block[i, j]
            if val != 0:
                data.append(val)
                rows.append(row_start + i)
                cols.append(col_start + j)


def _append_block_neg(data, rows, cols, block, row_start, col_start):
    """Append negated non-zero entries of a 3x3 block to COO lists."""
    for i in range(3):
        for j in range(3):
            val = block[i, j]
            if val != 0:
                data.append(-val)
                rows.append(row_start + i)
                cols.append(col_start + j)
