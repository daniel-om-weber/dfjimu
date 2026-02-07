# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, sin, cos, fabs

ctypedef np.float64_t DTYPE_t

cdef double dot_product(double[:] a, double[:] b, int n) nogil:
    cdef int i
    cdef double s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s

cdef void mat3_mul(double A[3][3], double B[3][3], double C[3][3]) nogil:
    cdef int i, j, k
    for i in range(3):
        for j in range(3):
            C[i][j] = 0.0
            for k in range(3):
                C[i][j] += A[i][k] * B[k][j]

cdef void mat3_vec_mul(double A[3][3], double x[3], double y[3]) nogil:
    cdef int i, k
    for i in range(3):
        y[i] = 0.0
        for k in range(3):
            y[i] += A[i][k] * x[k]

cdef void mat3_transpose(double A[3][3], double B[3][3]) nogil:
    cdef int i, j
    for i in range(3):
        for j in range(3):
            B[i][j] = A[j][i]

cdef int inv_3x3(double A[3][3], double invA[3][3]) nogil:
    cdef double det
    
    invA[0][0] = A[1][1]*A[2][2] - A[1][2]*A[2][1]
    invA[0][1] = A[0][2]*A[2][1] - A[0][1]*A[2][2]
    invA[0][2] = A[0][1]*A[1][2] - A[0][2]*A[1][1]
    
    invA[1][0] = A[1][2]*A[2][0] - A[1][0]*A[2][2]
    invA[1][1] = A[0][0]*A[2][2] - A[0][2]*A[2][0]
    invA[1][2] = A[0][2]*A[1][0] - A[0][0]*A[1][2]
    
    invA[2][0] = A[1][0]*A[2][1] - A[1][1]*A[2][0]
    invA[2][1] = A[0][1]*A[2][0] - A[0][0]*A[2][1]
    invA[2][2] = A[0][0]*A[1][1] - A[0][1]*A[1][0]
    
    det = A[0][0]*invA[0][0] + A[0][1]*invA[1][0] + A[0][2]*invA[2][0]
    
    if fabs(det) < 1e-12:
        return 0 
        
    cdef double invDet = 1.0 / det
    cdef int i, j
    for i in range(3):
        for j in range(3):
            invA[i][j] *= invDet
            
    return 1

# Quaternion Helper Functions
cdef void quat_mul(double q[4], double r[4], double res[4]) nogil:
    res[0] = q[0]*r[0] - q[1]*r[1] - q[2]*r[2] - q[3]*r[3]
    res[1] = q[0]*r[1] + q[1]*r[0] + q[2]*r[3] - q[3]*r[2]
    res[2] = q[0]*r[2] - q[1]*r[3] + q[2]*r[0] + q[3]*r[1]
    res[3] = q[0]*r[3] + q[1]*r[2] - q[2]*r[1] + q[3]*r[0]

cdef void exp_q(double v[3], double q[4]) nogil:
    cdef double nv = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    cdef double s
    if nv < 1e-12:
        q[0] = 1.0
        q[1] = 0.0
        q[2] = 0.0
        q[3] = 0.0
    else:
        s = sin(nv) / nv
        q[0] = cos(nv)
        q[1] = v[0]*s
        q[2] = v[1]*s
        q[3] = v[2]*s

cdef void exp_r(double v[3], double R[3][3]) nogil:
    cdef double nv = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    cdef double s, c, one_minus_c
    cdef double u0, u1, u2
    
    if nv < 1e-12:
        R[0][0] = 1.0; R[0][1] = 0.0; R[0][2] = 0.0
        R[1][0] = 0.0; R[1][1] = 1.0; R[1][2] = 0.0
        R[2][0] = 0.0; R[2][1] = 0.0; R[2][2] = 1.0
        return
        
    u0 = v[0]/nv; u1 = v[1]/nv; u2 = v[2]/nv
    s = sin(nv)
    c = cos(nv)
    one_minus_c = 1.0 - c
    
    # R = I + s*K + (1-c)*K^2
    # K = [[0, -u2, u1], [u2, 0, -u0], [-u1, u0, 0]]
    # K^2 = u*u^T - I
    
    # Diagonal
    R[0][0] = c + u0*u0*one_minus_c
    R[1][1] = c + u1*u1*one_minus_c
    R[2][2] = c + u2*u2*one_minus_c
    
    # Off-diagonal
    R[0][1] = u0*u1*one_minus_c - u2*s
    R[0][2] = u0*u2*one_minus_c + u1*s
    
    R[1][0] = u0*u1*one_minus_c + u2*s
    R[1][2] = u1*u2*one_minus_c - u0*s
    
    R[2][0] = u0*u2*one_minus_c - u1*s
    R[2][1] = u1*u2*one_minus_c + u0*s

cdef void quat2rot(double q[4], double R[3][3]) nogil:
    # R_nb (Body to Nav) from q_nb
    cdef double w=q[0], x=q[1], y=q[2], z=q[3]
    cdef double x2=x*x, y2=y*y, z2=z*z
    
    R[0][0] = 1.0 - 2.0*(y2 + z2)
    R[0][1] = 2.0*(x*y - z*w)
    R[0][2] = 2.0*(x*z + y*w)
    
    R[1][0] = 2.0*(x*y + z*w)
    R[1][1] = 1.0 - 2.0*(x2 + z2)
    R[1][2] = 2.0*(y*z - x*w)
    
    R[2][0] = 2.0*(x*z - y*w)
    R[2][1] = 2.0*(y*z + x*w)
    R[2][2] = 1.0 - 2.0*(x2 + y2)

cdef void cross_mat(double v[3], double M[3][3]) nogil:
    M[0][0] = 0.0;   M[0][1] = -v[2]; M[0][2] = v[1]
    M[1][0] = v[2];  M[1][1] = 0.0;   M[1][2] = -v[0]
    M[2][0] = -v[1]; M[2][1] = v[0];  M[2][2] = 0.0

def run_mekf_cython(np.ndarray[DTYPE_t, ndim=2] gyr1,
                    np.ndarray[DTYPE_t, ndim=2] gyr2,
                    np.ndarray[DTYPE_t, ndim=2] acc1,
                    np.ndarray[DTYPE_t, ndim=2] acc2,
                    np.ndarray[DTYPE_t, ndim=2] C1,
                    np.ndarray[DTYPE_t, ndim=2] C2,
                    double Fs,
                    np.ndarray[DTYPE_t, ndim=1] q_init,
                    np.ndarray[DTYPE_t, ndim=1] Q_in,
                    double R_diag=0.011552,
                    double P_init_diag=0.1225):
    
    cdef int N = gyr1.shape[0]
    cdef double dt = 1.0 / Fs
    cdef double dt_half = 0.5 * dt
    cdef int t, i, j, k
    
    # Outputs
    cdef np.ndarray[DTYPE_t, ndim=2] q_s1_out = np.zeros((N, 4), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] q_s2_out = np.zeros((N, 4), dtype=np.float64)
    
    # Initialization
    cdef double q1_curr[4]
    cdef double q2_curr[4]
    for i in range(4):
        q1_curr[i] = q_init[i]
        q2_curr[i] = q_init[i]
        q_s1_out[0, i] = q_init[i]
        q_s2_out[0, i] = q_init[i]

    # Covariance P (6x6)
    cdef double P[6][6]
    for i in range(6):
        for j in range(6):
            P[i][j] = 0.0
    for i in range(6):
        P[i][i] = P_init_diag
        
    # GQGt = dt^2 * Q
    cdef double GQGt[6]
    for i in range(6):
        GQGt[i] = (dt*dt) * Q_in[i]
        if GQGt[i] < 1e-12: GQGt[i] = 1e-8 # floor
        
    cdef double R_val = R_diag
    
    # Temps
    cdef double F[6][6]
    cdef double R_rot1[3][3]
    cdef double R_rot2[3][3]
    cdef double dq1[4]
    cdef double dq2[4]
    cdef double q1_pred[4]
    cdef double q2_pred[4]
    cdef double v_tmp[3]
    
    cdef double P_new[6][6]
    cdef double H[3][6]
    cdef double S[3][3]
    cdef double invS[3][3]
    cdef double K[6][3]
    cdef double e[3]
    cdef double n[6]
    cdef double P_tilde[6][6]
    cdef double J[6][6]
    
    # Helper vars
    cdef double tmp_R1[3][3]
    cdef double tmp_R2[3][3]
    cdef double cross_c1[3][3]
    cdef double cross_c2[3][3]
    cdef double term1[3], term2[3]
    cdef double val
    cdef double FP[6][6]
    cdef double HP[3][6]
    cdef double KS[6][3]
    cdef double JP[6][6]
    
    # Main Loop
    for t in range(1, N):
        # --- Time Update ---
        # 1. F Matrix (Diagonal blocks)
        for i in range(3): v_tmp[i] = -dt * gyr1[t-1, i]
        exp_r(v_tmp, R_rot1)
        for i in range(3): v_tmp[i] = -dt * gyr2[t-1, i]
        exp_r(v_tmp, R_rot2)
        
        # Reset F/J/H to 0
        for i in range(6):
            for j in range(6):
                F[i][j] = 0.0
                J[i][j] = 0.0
        for i in range(3):
            for j in range(6):
                H[i][j] = 0.0
        
        for i in range(3):
            for j in range(3):
                F[i][j] = R_rot1[i][j]
                F[i+3][j+3] = R_rot2[i][j]
                
        # 2. Predict Q
        for i in range(3): v_tmp[i] = dt_half * gyr1[t-1, i]
        exp_q(v_tmp, dq1)
        quat_mul(q1_curr, dq1, q1_pred)
        
        for i in range(3): v_tmp[i] = dt_half * gyr2[t-1, i]
        exp_q(v_tmp, dq2)
        quat_mul(q2_curr, dq2, q2_pred)
        
        # 3. Predict P = F P F' + Q
        # Manual matrix multiplication 6x6. Sparse optimized.
        # Temp P_new
        for i in range(6):
            for j in range(6):
                val = 0.0
                # P * F_T (since F is block diag, we can optimize but full mul is fine for 6x6)
                # Actually P * F.T
                # Let's do tmp = P * F.T
                # Then F * tmp
                # Optimized for block structure:
                pass 
        
        # Simple 6x6 matmul for F*P*F.T
        # F is block diagonal.
        # P = [P11 P12; P21 P22]
        # F = [R1 0; 0 R2]
        # F P F.T = [R1 P11 R1.T,  R1 P12 R2.T; R2 P21 R1.T, R2 P22 R2.T]
        
        # Helper to compute R A R.T for 3x3
        # We need a function for this but inline is messy.
        # Fallback to naive 6x6 loop for simplicity and correctness.
        # It's only 6x6.
        
        # FP = F * P
        for i in range(6):
            for j in range(6):
                FP[i][j] = 0.0
                for k in range(6):
                    FP[i][j] += F[i][k] * P[k][j]
        
        # P_pred = FP * F.T + Q
        for i in range(6):
            for j in range(6):
                val = 0.0
                for k in range(6):
                    val += FP[i][k] * F[j][k] # F[j][k] is F.T[k][j]
                P[i][j] = val
                if i == j: P[i][j] += GQGt[i]
                
        # --- Measurement Update ---
        # R_nb from q_pred
        quat2rot(q1_pred, tmp_R1)
        quat2rot(q2_pred, tmp_R2)
        
        # e = R1 c1 - R2 c2
        for i in range(3): v_tmp[i] = C1[t, i]
        mat3_vec_mul(tmp_R1, v_tmp, term1)
        
        for i in range(3): v_tmp[i] = C2[t, i]
        mat3_vec_mul(tmp_R2, v_tmp, term2)
        
        for i in range(3):
            e[i] = term1[i] - term2[i]
            
        # H matrix
        # H1 = R1 * [c1]x
        for i in range(3): v_tmp[i] = C1[t, i]
        cross_mat(v_tmp, cross_c1)
        mat3_mul(tmp_R1, cross_c1, R_rot1) # Reuse R_rot1 as temp result
        
        for i in range(3): v_tmp[i] = C2[t, i]
        cross_mat(v_tmp, cross_c2)
        mat3_mul(tmp_R2, cross_c2, R_rot2)
        
        for i in range(3):
            for j in range(3):
                H[i][j] = R_rot1[i][j]
                H[i][j+3] = -R_rot2[i][j]
                
        # S = H P H.T + R
        # HP = H * P
        for i in range(3):
            for j in range(6):
                HP[i][j] = 0.0
                for k in range(6):
                    HP[i][j] += H[i][k] * P[k][j]
                    
        # S = HP * H.T + R
        for i in range(3):
            for j in range(3):
                val = 0.0
                for k in range(6):
                    val += HP[i][k] * H[j][k]
                S[i][j] = val
                if i == j: S[i][j] += R_val
                
        # invS
        if inv_3x3(S, invS) == 0:
            # Singularity fallback
            invS[0][0]=1; invS[1][1]=1; invS[2][2]=1
        
        # K = P * H.T * invS
        # PHt = (HP).T = P * H.T
        # K = PHt * invS
        for i in range(6):
            for j in range(3):
                K[i][j] = 0.0
                for k in range(3):
                    # PHt[i][k] is HP[k][i]
                    K[i][j] += HP[k][i] * invS[k][j]
                    
        # n = K * e
        for i in range(6):
            n[i] = 0.0
            for j in range(3):
                n[i] += K[i][j] * e[j]
                
        # P_tilde = P - K S K.T
        # K S
        for i in range(6):
            for j in range(3):
                KS[i][j] = 0.0
                for k in range(3):
                    KS[i][j] += K[i][k] * S[k][j]
                    
        for i in range(6):
            for j in range(6):
                val = 0.0
                for k in range(3):
                    val += KS[i][k] * K[j][k]
                P_tilde[i][j] = P[i][j] - val
                
        # --- Relinearize ---
        # q = q * exp(0.5 * n)
        for i in range(3): v_tmp[i] = 0.5 * n[i]
        exp_q(v_tmp, dq1)
        quat_mul(q1_pred, dq1, q1_curr)
        
        for i in range(3): v_tmp[i] = 0.5 * n[i+3]
        exp_q(v_tmp, dq2)
        quat_mul(q2_pred, dq2, q2_curr)
        
        # J matrix
        for i in range(3): v_tmp[i] = -n[i]
        exp_r(v_tmp, R_rot1)
        for i in range(3): v_tmp[i] = -n[i+3]
        exp_r(v_tmp, R_rot2)
        
        for i in range(3):
            for j in range(3):
                J[i][j] = R_rot1[i][j]
                J[i+3][j+3] = R_rot2[i][j]
                
        # P = J P_tilde J.T
        # JP = J * P_tilde
        for i in range(6):
            for j in range(6):
                JP[i][j] = 0.0
                for k in range(6):
                    JP[i][j] += J[i][k] * P_tilde[k][j]
                    
        for i in range(6):
            for j in range(6):
                val = 0.0
                for k in range(6):
                    val += JP[i][k] * J[j][k]
                P[i][j] = val
        
        # Normalize
        val = sqrt(q1_curr[0]*q1_curr[0] + q1_curr[1]*q1_curr[1] + q1_curr[2]*q1_curr[2] + q1_curr[3]*q1_curr[3])
        for i in range(4): q1_curr[i] /= val
        val = sqrt(q2_curr[0]*q2_curr[0] + q2_curr[1]*q2_curr[1] + q2_curr[2]*q2_curr[2] + q2_curr[3]*q2_curr[3])
        for i in range(4): q2_curr[i] /= val
        
        # Store
        for i in range(4):
            q_s1_out[t, i] = q1_curr[i]
            q_s2_out[t, i] = q2_curr[i]
            
    return q_s1_out, q_s2_out