# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, sin, cos, acos, fabs

ctypedef np.float64_t DTYPE_t

# --- Inline Math Helpers (Pointer based) ---

cdef inline void mat3_mul(double* A, double* B, double* C) nogil:
    cdef int i, j, k
    for i in range(3):
        for j in range(3):
            C[i*3+j] = 0.0
            for k in range(3):
                C[i*3+j] += A[i*3+k] * B[k*3+j]

cdef inline void mat3_vec_mul(double* A, double* x, double* y) nogil:
    cdef int i, k
    for i in range(3):
        y[i] = 0.0
        for k in range(3):
            y[i] += A[i*3+k] * x[k]

cdef inline void quat_mul(double* q, double* r, double* res) nogil:
    res[0] = q[0]*r[0] - q[1]*r[1] - q[2]*r[2] - q[3]*r[3]
    res[1] = q[0]*r[1] + q[1]*r[0] + q[2]*r[3] - q[3]*r[2]
    res[2] = q[0]*r[2] - q[1]*r[3] + q[2]*r[0] + q[3]*r[1]
    res[3] = q[0]*r[3] + q[1]*r[2] - q[2]*r[1] + q[3]*r[0]

cdef inline void quat_conj(double* q, double* res) nogil:
    res[0] = q[0]
    res[1] = -q[1]
    res[2] = -q[2]
    res[3] = -q[3]

cdef inline void quat2rot(double* q, double* R) nogil:
    cdef double w=q[0], x=q[1], y=q[2], z=q[3]
    cdef double x2=x*x, y2=y*y, z2=z*z
    # R is 3x3 (9 doubles)
    # Row 0
    R[0] = 1.0 - 2.0*(y2 + z2); R[1] = 2.0*(x*y - z*w); R[2] = 2.0*(x*z + y*w)
    # Row 1
    R[3] = 2.0*(x*y + z*w);     R[4] = 1.0 - 2.0*(x2 + z2); R[5] = 2.0*(y*z - x*w)
    # Row 2
    R[6] = 2.0*(x*z - y*w);     R[7] = 2.0*(y*z + x*w);     R[8] = 1.0 - 2.0*(x2 + y2)

cdef inline void cross_mat(double* v, double* M) nogil:
    M[0] = 0.0;   M[1] = -v[2]; M[2] = v[1]
    M[3] = v[2];  M[4] = 0.0;   M[5] = -v[0]
    M[6] = -v[1]; M[7] = v[0];  M[8] = 0.0

cdef inline void exp_q(double* v, double* q) nogil:
    cdef double nv = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    cdef double s
    if nv < 1e-12:
        q[0]=1.0; q[1]=0.0; q[2]=0.0; q[3]=0.0
    else:
        s = sin(nv) / nv
        q[0] = cos(nv); q[1]=v[0]*s; q[2]=v[1]*s; q[3]=v[2]*s

cdef inline void log_q(double* q, double* v) nogil:
    cdef double nv = sqrt(q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    cdef double scale
    if nv < 1e-12:
        v[0]=0.0; v[1]=0.0; v[2]=0.0
    else:
        if q[0] > 1.0: q[0] = 1.0
        elif q[0] < -1.0: q[0] = -1.0
        scale = acos(q[0]) / nv
        v[0]=q[1]*scale; v[1]=q[2]*scale; v[2]=q[3]*scale

# Matrices for derivatives
cdef inline void get_quatL(double* q, double* L) nogil:
    # L is 4x4 (16 doubles)
    cdef double w=q[0], x=q[1], y=q[2], z=q[3]
    L[0]=w; L[1]=-x; L[2]=-y; L[3]=-z
    L[4]=x; L[5]=w;  L[6]=-z; L[7]=y
    L[8]=y; L[9]=z;  L[10]=w; L[11]=-x
    L[12]=z; L[13]=-y; L[14]=x; L[15]=w

cdef inline void get_quatR(double* q, double* R) nogil:
    cdef double w=q[0], x=q[1], y=q[2], z=q[3]
    R[0]=w; R[1]=-x; R[2]=-y; R[3]=-z
    R[4]=x; R[5]=w;  R[6]=z;  R[7]=-y
    R[8]=y; R[9]=-z; R[10]=w; R[11]=x
    R[12]=z; R[13]=y; R[14]=-x; R[15]=w

# --- Main System Builder ---

def build_system_cython(
    np.ndarray[DTYPE_t, ndim=2] q_lin_s1, # (N, 4)
    np.ndarray[DTYPE_t, ndim=2] q_lin_s2, # (N, 4)
    np.ndarray[DTYPE_t, ndim=2] gyr1, # (N, 3)
    np.ndarray[DTYPE_t, ndim=2] gyr2, # (N, 3)
    np.ndarray[DTYPE_t, ndim=2] C1, # (N, 3)
    np.ndarray[DTYPE_t, ndim=2] C2, # (N, 3)
    np.ndarray[DTYPE_t, ndim=1] q_init, # (4,)
    double Fs,
    np.ndarray[DTYPE_t, ndim=2] icov_w1, # (3, 3) sensor 1
    np.ndarray[DTYPE_t, ndim=2] icov_w2, # (3, 3) sensor 2
    np.ndarray[DTYPE_t, ndim=2] icov_i, # (3, 3)
    np.ndarray[DTYPE_t, ndim=2] icov_lnk # (3, 3)
):
    cdef int N = q_lin_s1.shape[0]
    cdef double dt = 1.0 / Fs
    
    # Pre-allocate Error Vector
    cdef int n_err = 9 * N
    cdef np.ndarray[DTYPE_t, ndim=1] epsilon = np.zeros(n_err, dtype=np.float64)
    
    # Estimate max non-zeros
    cdef int max_nnz = 60 * N 
    cdef np.ndarray[DTYPE_t, ndim=1] j_data = np.zeros(max_nnz, dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] j_rows = np.zeros(max_nnz, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] j_cols = np.zeros(max_nnz, dtype=np.int32)
    
    cdef int idx_nnz = 0
    cdef int idx_err = 0
    
    # Temps
    cdef double q_tmp[4], q_conj[4], q_res[4], v_tmp[3], v_res[3]
    cdef double mat3_tmp[9], mat3_res[9], mat4_tmp[16]
    cdef double L[16], R[16]
    cdef int t, i, j
    
    # Pointers to cov matrices for speed
    cdef double* p_icov_i = &icov_i[0,0]
    cdef double* p_icov_w1 = &icov_w1[0,0]
    cdef double* p_icov_w2 = &icov_w2[0,0]
    cdef double* p_icov_lnk = &icov_lnk[0,0]
    
    # --- Sensor 1 Loop ---
    # t=0: Init Cost
    cdef double q_init_inv[4]
    quat_conj(&q_init[0], q_init_inv)
    quat_mul(q_init_inv, &q_lin_s1[0, 0], q_tmp)
    log_q(q_tmp, v_tmp)
    for i in range(3): v_res[i] = 2.0 * v_tmp[i]
    mat3_vec_mul(p_icov_i, v_res, &epsilon[0])
    
    # Jacobian Init
    get_quatL(q_tmp, L)
    # Extract bottom-right 3x3 of L -> mat3_tmp
    # L indices: 0..15.
    # L[i+1][j+1] -> L[(i+1)*4 + (j+1)]
    for i in range(3):
        for j in range(3):
            mat3_tmp[i*3+j] = L[(i+1)*4 + (j+1)]
    
    mat3_mul(p_icov_i, mat3_tmp, mat3_res)
    
    for i in range(3):
        for j in range(3):
            if mat3_res[i*3+j] != 0:
                j_data[idx_nnz] = mat3_res[i*3+j]
                j_rows[idx_nnz] = i
                j_cols[idx_nnz] = j
                idx_nnz += 1
                
    # Motion Loop S1
    cdef int row_offset = 3
    cdef double q_prev[4], q_curr[4]
    
    for t in range(1, N):
        for i in range(4): q_prev[i] = q_lin_s1[t-1, i]
        for i in range(4): q_curr[i] = q_lin_s1[t, i]
        
        quat_conj(q_prev, q_conj)
        quat_mul(q_conj, q_curr, q_tmp)
        log_q(q_tmp, v_tmp)
        for i in range(3): v_res[i] = (2.0/dt)*v_tmp[i] - gyr1[t-1, i]
        mat3_vec_mul(p_icov_w1, v_res, &epsilon[row_offset + (t-1)*3])

        # J t-1: -1/dt * R[bottom-right]
        get_quatR(q_tmp, R)
        for i in range(3):
            for j in range(3):
                mat3_tmp[i*3+j] = -1.0 * R[(i+1)*4 + (j+1)] * (1.0/dt)
        mat3_mul(p_icov_w1, mat3_tmp, mat3_res)

        for i in range(3):
            for j in range(3):
                if mat3_res[i*3+j] != 0:
                    j_data[idx_nnz] = mat3_res[i*3+j]
                    j_rows[idx_nnz] = row_offset + (t-1)*3 + i
                    j_cols[idx_nnz] = (t-1)*3 + j
                    idx_nnz += 1

        # J t: 1/dt * L[bottom-right]
        get_quatL(q_tmp, L)
        for i in range(3):
            for j in range(3):
                mat3_tmp[i*3+j] = L[(i+1)*4 + (j+1)] * (1.0/dt)
        mat3_mul(p_icov_w1, mat3_tmp, mat3_res)
        
        for i in range(3):
            for j in range(3):
                if mat3_res[i*3+j] != 0:
                    j_data[idx_nnz] = mat3_res[i*3+j]
                    j_rows[idx_nnz] = row_offset + (t-1)*3 + i
                    j_cols[idx_nnz] = t*3 + j
                    idx_nnz += 1
                    
    # --- Sensor 2 Loop ---
    row_offset = 3 * N
    cdef int col_offset = 3 * N
    
    # S2 Init
    quat_conj(&q_init[0], q_init_inv)
    quat_mul(q_init_inv, &q_lin_s2[0, 0], q_tmp)
    log_q(q_tmp, v_tmp)
    for i in range(3): v_res[i] = 2.0 * v_tmp[i]
    mat3_vec_mul(p_icov_i, v_res, &epsilon[row_offset])
    
    get_quatL(q_tmp, L)
    for i in range(3):
        for j in range(3):
            mat3_tmp[i*3+j] = L[(i+1)*4 + (j+1)]
    mat3_mul(p_icov_i, mat3_tmp, mat3_res)
    
    for i in range(3):
        for j in range(3):
            if mat3_res[i*3+j] != 0:
                j_data[idx_nnz] = mat3_res[i*3+j]
                j_rows[idx_nnz] = row_offset + i
                j_cols[idx_nnz] = col_offset + j
                idx_nnz += 1
                
    row_offset += 3
    
    # Motion S2
    for t in range(1, N):
        for i in range(4): q_prev[i] = q_lin_s2[t-1, i]
        for i in range(4): q_curr[i] = q_lin_s2[t, i]
        
        quat_conj(q_prev, q_conj)
        quat_mul(q_conj, q_curr, q_tmp)
        log_q(q_tmp, v_tmp)
        for i in range(3): v_res[i] = (2.0/dt)*v_tmp[i] - gyr2[t-1, i]
        mat3_vec_mul(p_icov_w2, v_res, &epsilon[row_offset + (t-1)*3])

        get_quatR(q_tmp, R)
        for i in range(3):
            for j in range(3):
                mat3_tmp[i*3+j] = -1.0 * R[(i+1)*4 + (j+1)] * (1.0/dt)
        mat3_mul(p_icov_w2, mat3_tmp, mat3_res)

        for i in range(3):
            for j in range(3):
                if mat3_res[i*3+j] != 0:
                    j_data[idx_nnz] = mat3_res[i*3+j]
                    j_rows[idx_nnz] = row_offset + (t-1)*3 + i
                    j_cols[idx_nnz] = col_offset + (t-1)*3 + j
                    idx_nnz += 1

        get_quatL(q_tmp, L)
        for i in range(3):
            for j in range(3):
                mat3_tmp[i*3+j] = L[(i+1)*4 + (j+1)] * (1.0/dt)
        mat3_mul(p_icov_w2, mat3_tmp, mat3_res)
        
        for i in range(3):
            for j in range(3):
                if mat3_res[i*3+j] != 0:
                    j_data[idx_nnz] = mat3_res[i*3+j]
                    j_rows[idx_nnz] = row_offset + (t-1)*3 + i
                    j_cols[idx_nnz] = col_offset + t*3 + j
                    idx_nnz += 1
                    
    # --- Link Constraint ---
    row_offset = 6 * N
    cdef double R1[9], R2[9], c1[3], c2[3], term1[3], term2[3], err_lnk[3], mat3_cross[9]
    
    for t in range(N):
        for i in range(4): q_prev[i] = q_lin_s1[t, i]
        for i in range(4): q_curr[i] = q_lin_s2[t, i]
        
        quat2rot(q_prev, R1)
        quat2rot(q_curr, R2)
        
        for i in range(3): c1[i] = C1[t, i]
        for i in range(3): c2[i] = C2[t, i]
        
        mat3_vec_mul(R1, c1, term1)
        mat3_vec_mul(R2, c2, term2)
        for i in range(3): err_lnk[i] = term1[i] - term2[i]
        mat3_vec_mul(p_icov_lnk, err_lnk, &epsilon[row_offset + t*3])
        
        # J S1: - icov * R1 * [c1]x
        cross_mat(c1, mat3_cross)
        mat3_mul(R1, mat3_cross, mat3_tmp)
        mat3_mul(p_icov_lnk, mat3_tmp, mat3_res)
        
        for i in range(3):
            for j in range(3):
                if mat3_res[i*3+j] != 0:
                    j_data[idx_nnz] = -mat3_res[i*3+j]
                    j_rows[idx_nnz] = row_offset + t*3 + i
                    j_cols[idx_nnz] = t*3 + j
                    idx_nnz += 1
                    
        # J S2: icov * R2 * [c2]x
        cross_mat(c2, mat3_cross)
        mat3_mul(R2, mat3_cross, mat3_tmp)
        mat3_mul(p_icov_lnk, mat3_tmp, mat3_res)
        
        for i in range(3):
            for j in range(3):
                if mat3_res[i*3+j] != 0:
                    j_data[idx_nnz] = mat3_res[i*3+j]
                    j_rows[idx_nnz] = row_offset + t*3 + i
                    j_cols[idx_nnz] = 3*N + t*3 + j
                    idx_nnz += 1
                    
    return (
        epsilon, 
        j_data[:idx_nnz], 
        j_rows[:idx_nnz], 
        j_cols[:idx_nnz]
    )

def update_lin_points_cython(
    np.ndarray[DTYPE_t, ndim=2] q_lin,
    np.ndarray[DTYPE_t, ndim=2] n
):
    cdef int N = q_lin.shape[0]
    cdef int t, i
    cdef double v[3], dq[4], q_new[4], q_old[4]
    cdef np.ndarray[DTYPE_t, ndim=2] q_out = np.zeros_like(q_lin)
    
    for t in range(N):
        for i in range(3): v[i] = 0.5 * n[t, i]
        exp_q(v, dq)
        for i in range(4): q_old[i] = q_lin[t, i]
        quat_mul(q_old, dq, q_new)
        for i in range(4): q_out[t, i] = q_new[i]
        
    return q_out