import numpy as np

def quatmultiply(q, r):
    """
    Multiply two quaternions.
    Expects q and r to be of shape (N, 4) or (4,).
    Convention: [w, x, y, z]
    """
    q = np.atleast_2d(q)
    r = np.atleast_2d(r)
    
    # If one is a single quaternion and the other is N, broadcast
    if q.shape[0] == 1 and r.shape[0] > 1:
        q = np.tile(q, (r.shape[0], 1))
    elif r.shape[0] == 1 and q.shape[0] > 1:
        r = np.tile(r, (q.shape[0], 1))

    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r0, r1, r2, r3 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]

    out = np.zeros_like(q)
    out[:, 0] = q0*r0 - q1*r1 - q2*r2 - q3*r3
    out[:, 1] = q0*r1 + q1*r0 + q2*r3 - q3*r2
    out[:, 2] = q0*r2 - q1*r3 + q2*r0 + q3*r1
    out[:, 3] = q0*r3 + q1*r2 - q2*r1 + q3*r0
    
    if out.shape[0] == 1:
        return out.flatten()
    return out

def quatconj(q):
    """
    Conjugate of quaternion [w, x, y, z] -> [w, -x, -y, -z]
    """
    q = np.atleast_2d(q)
    out = q.copy()
    out[:, 1:] = -out[:, 1:]
    if out.shape[0] == 1:
        return out.flatten()
    return out

def quatinv(q):
    """
    Inverse of quaternion. For unit quaternions, same as conjugate.
    """
    return quatconj(q)

def quatnormalize(q):
    q = np.atleast_2d(q)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    out = q / norm
    if out.shape[0] == 1:
        return out.flatten()
    return out

def quat2matrix(q):
    """
    Converts quaternion to rotation matrix (DCM).
    Matches MATLAB's quat2matrix.m
    q: [w, x, y, z]
    """
    # q must be normalized
    if q.ndim == 1:
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        return np.array([
            [w**2 + x**2 - y**2 - z**2, 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     w**2 - x**2 + y**2 - z**2, 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     w**2 - x**2 - y**2 + z**2]
        ])
    else:
        # Vectorized version if needed, but the original function seems to process one by one or return 3x3 for one input.
        # The usage in code typically iterates or processes one.
        # For simplicity, if input is (N, 4), return (N, 3, 3)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
        
        R = np.zeros((q.shape[0], 3, 3))
        
        R[:, 0, 0] = w**2 + x**2 - y**2 - z**2
        R[:, 0, 1] = 2*(x*y - z*w)
        R[:, 0, 2] = 2*(x*z + y*w)
        
        R[:, 1, 0] = 2*(x*y + z*w)
        R[:, 1, 1] = w**2 - x**2 + y**2 - z**2
        R[:, 1, 2] = 2*(y*z - x*w)
        
        R[:, 2, 0] = 2*(x*z - y*w)
        R[:, 2, 1] = 2*(y*z + x*w)
        R[:, 2, 2] = w**2 - x**2 - y**2 + z**2
        return R

def angDist(s, r):
    """
    Angular distance between two quaternions in degrees.
    Matches MATLAB implementation.
    """
    s = np.atleast_2d(s)
    r = np.atleast_2d(r)
    
    q_inv_r = quatconj(r) # assuming unit quaternions
    quat_Diff = quatmultiply(q_inv_r, s)
    quat_Diff = np.atleast_2d(quat_Diff)
    
    # Check sign of w component
    w = quat_Diff[:, 0]
    flip_mask = w < 0
    quat_Diff[flip_mask] = -quat_Diff[flip_mask]
    
    ang_d = np.degrees(2 * np.real(np.arccos(np.clip(quat_Diff[:, 0], -1.0, 1.0))))
    return ang_d

def approxDerivative(y, Fs=50):
    """
    Finite difference approximation (5-point stencil).
    y: (N,) or (N, M)
    """
    y = np.atleast_1d(y)
    dy = np.zeros_like(y)
    
    # 5-point stencil
    # dy(3:end-2)= (y(1:end-4) - 8*y(2:end-3) +8*y(4:end-1)-y(5:end))*(Fs/12);
    
    dy[2:-2] = (y[:-4] - 8*y[1:-3] + 8*y[3:-1] - y[4:]) * (Fs/12.0)
    
    return dy

def crossM(q):
    """
    Cross product matrix of a quaternion or vector.
    q: (3,) or (4,) or (N, 3) or (N, 4)
    Returns (3, 3) or (N, 3, 3)
    """
    q = np.atleast_1d(q)
    
    if q.ndim == 1:
        if q.shape[0] == 3:
            qv = q
            w = 0
        else:
            w = q[0]
            qv = q[1:]
        
        return np.array([
            [0, -qv[2], qv[1]],
            [qv[2], 0, -qv[0]],
            [-qv[1], qv[0], 0]
        ])
    else:
        # Vectorized
        if q.shape[1] == 3:
            qv = q
            w = np.zeros(q.shape[0])
        else:
            w = q[:, 0]
            qv = q[:, 1:]
            
        qx = np.zeros((q.shape[0], 3, 3))
        qx[:, 0, 1] = -qv[:, 2]
        qx[:, 0, 2] =  qv[:, 1]
        qx[:, 1, 0] =  qv[:, 2]
        qx[:, 1, 2] = -qv[:, 0]
        qx[:, 2, 0] = -qv[:, 1]
        qx[:, 2, 1] =  qv[:, 0]
        
        return qx 

def calculateK(w, wd):
    """
    w, wd: (3, N) or (N, 3) ?
    MATLAB uses (3, N) for gyr.
    Returns K: (3, 3, N)
    """
    # Prefer (N, 3) in Python.
    # If input is (3, N), transpose.
    if w.shape[0] == 3 and w.shape[1] > 3:
        w = w.T
        wd = wd.T
    
    N = w.shape[0]
    K = np.zeros((N, 3, 3))
    
    cw = crossM(w) # (N, 3, 3)
    cwd = crossM(wd) # (N, 3, 3)
    
    # K = crossM(w)^2 + crossM(wd)
    # Python matmul @
    K = cw @ cw + cwd
    return K

def calcAccatCenter(yg, dyg, ya, r):
    """
    yg, dyg, ya: (3, N) or (N, 3)
    r: (3,) or (1, 3)
    """
    if yg.shape[0] == 3 and yg.shape[1] > 3:
        yg = yg.T
        dyg = dyg.T
        ya = ya.T
    
    r = r.flatten() # (3,)
    
    K = calculateK(yg, dyg) # (N, 3, 3)
    
    # D = K * r'
    # C = ya - K * r'
    
    Kr = np.einsum('nij,j->ni', K, r)
    
    D = Kr
    C = ya - Kr
    
    return C, D

def preprocess_acc_at_center(gyr, acc, r, Fs):
    """Compute acceleration at joint center from raw IMU data.

    Parameters
    ----------
    gyr : (N, 3) gyroscope data
    acc : (N, 3) accelerometer data
    r : (3,) position vector from sensor to joint center
    Fs : float, sampling frequency

    Returns
    -------
    C : (N, 3) acceleration at joint center
    """
    dgyr = np.column_stack([approxDerivative(gyr[:, i], Fs) for i in range(3)])
    C, _ = calcAccatCenter(gyr, dgyr, acc, r)
    if C.shape[0] == 3 and C.shape[1] > 3:
        C = C.T
    return C


def EXPr(v):
    """
    Returns DCM R for orientation deviation parameterization.
    v: (3,)
    """
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return np.eye(3)
        
    vX = crossM(v / nv)
    R = np.eye(3) + np.sin(nv)*vX + (1 - np.cos(nv))*(vX @ vX)
    return R

def EXPq(v):
    """
    Exponential map from vector to quaternion.
    v: (3,)
    """
    v = np.atleast_1d(v)
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    s = np.sin(nv) / nv
    q = np.array([np.cos(nv), v[0]*s, v[1]*s, v[2]*s])
    return q

def LOGq(q):
    """
    Logarithm map from quaternion to vector.
    q: (4,)
    """
    q = np.atleast_1d(q)
    w = q[0]
    v = q[1:]
    nv = np.linalg.norm(v)
    
    if nv < 1e-12:
        return np.zeros(3)
    
    theta = np.arccos(np.clip(w, -1.0, 1.0))
    scale = theta / nv
    return v * scale

def quatL(q):
    """
    Left multiplication matrix of quaternion q.
    Returns 4x4.
    """
    q = np.atleast_1d(q)
    if len(q) == 3:
        q = np.concatenate(([0], q))
        
    w = q[0]
    v = q[1:]
    
    qL = np.zeros((4, 4))
    qL[0, 0] = w
    qL[0, 1:] = -v
    qL[1:, 0] = v
    # qL(2:4, 2:4) = q0*eye(3) + crossM(q)
    qL[1:, 1:] = w * np.eye(3) + crossM(v)
    return qL

def quatR(q):
    """
    Right multiplication matrix of quaternion q.
    Returns 4x4.
    """
    q = np.atleast_1d(q)
    if len(q) == 3:
        q = np.concatenate(([0], q))
        
    w = q[0]
    v = q[1:]
    
    qR = np.zeros((4, 4))
    qR[0, 0] = w
    qR[0, 1:] = -v
    qR[1:, 0] = v
    # qR(2:4, 2:4) = q0*eye(3) - crossM(q)
    qR[1:, 1:] = w * np.eye(3) - crossM(v)
    return qR

def integrateGyr(gyr, q_1, T):
    """
    gyr: (N, 3)
    q_1: (4,)
    T: sampling period
    """
    N = gyr.shape[0]
    orientation = np.zeros((N, 4))
    orientation[0] = q_1
    
    q = q_1
    for i in range(1, N):
        dq = EXPq((T/2) * gyr[i])
        q = quatmultiply(q, dq)
        orientation[i] = q
        
    return orientation

def update_linPoints(q_lin, n):
    """
    q_lin: (N, 4)
    n: (N, 3)
    """
    # Vectorized update
    # q_lin_ = q_lin * EXPq(n/2)
    
    norms = np.linalg.norm(n/2, axis=1, keepdims=True)
    mask = (norms.flatten() < 1e-12)
    
    sin_term = np.sin(norms) / norms
    sin_term[mask] = 0 # Dummy value
    
    exp_n = np.zeros((n.shape[0], 4))
    exp_n[:, 0] = np.cos(norms).flatten()
    exp_n[:, 1:] = (n/2) * sin_term
    
    exp_n[mask, 0] = 1.0
    exp_n[mask, 1:] = 0.0
    
    return quatmultiply(q_lin, exp_n)

# Constant matrices
def get_dlogdq():
    M = np.zeros((3, 4))
    M[:, 1:] = np.eye(3)
    return M

def get_dexpndn():
    return get_dlogdq().T

def get_dexpnCdexpn():
    M = -np.eye(4)
    M[0, 0] = 1
    return M
