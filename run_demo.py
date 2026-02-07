import numpy as np
import scipy.io
import time
import sys
import os

from dfjimu import map_acc, mekf_acc
from dfjimu.utils.common import quatmultiply, quatconj, angDist, approxDerivative, calcAccatCenter

def calculate_rmse(q_ref, q_est, shift_est=0):
    if q_ref.shape[0] == 4 and q_ref.shape[1] > 4:
        q_ref = q_ref.T
    N_ref = q_ref.shape[0]
    N_est = q_est.shape[0]
    limit = min(N_ref, N_est - shift_est)
    if limit > N_ref - 1: limit = N_ref - 1
    dists = [angDist(q_ref[t], q_est[t + shift_est]) for t in range(limit)]
    return np.sqrt(np.mean(np.array(dists)**2))

def suppress_stdout():
    return open(os.devnull, 'w')

def main():
    file_path = 'data/data_2D_07.mat'
    print(f"Benchmarking MEKF implementations on {file_path}...")

    mat = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    data = mat['data']
    r1 = np.atleast_1d(-data.r_12).flatten()
    r2 = np.atleast_1d(-data.r_21).flatten()
    sensor_data = data.sensorData

    acc1 = sensor_data[:, 0:3]; gyr1 = sensor_data[:, 3:6]
    acc2 = sensor_data[:, 6:9]; gyr2 = sensor_data[:, 9:12]

    qGS1_ref = data.ref[:, 0:4]; qGS2_ref = data.ref[:, 4:8]
    qREF = quatmultiply(quatconj(qGS1_ref), qGS2_ref)

    Fs = 50.0
    q_init = np.array([1.0, 0.0, 0.0, 0.0])

    # 1. MAP-acc (optimization-based smoothing)
    print("1. MAP-acc (optimization)...")
    cov_w = np.eye(6) * 0.01; cov_i = np.eye(3)*0.1; cov_lnk = np.eye(3)
    start = time.time()
    old = sys.stdout; sys.stdout = suppress_stdout()
    try:
        q1, q2 = map_acc(gyr1, gyr2, acc1, acc2, r1, r2, Fs, q_init,
                         cov_w, cov_i, cov_lnk, iterations=10)
    finally:
        sys.stdout.close(); sys.stdout = old
    time_opt = time.time() - start
    rmse_opt = calculate_rmse(qREF, quatmultiply(quatconj(q1), q2), 1)

    # 2. MEKF-acc (online filtering)
    print("2. MEKF-acc (filtering)...")
    start = time.time()
    q1_fast, q2_fast = mekf_acc(gyr1, gyr2, acc1, acc2, r1, r2, Fs, q_init)
    time_mekf = time.time() - start
    rmse_mekf = calculate_rmse(qREF, quatmultiply(quatconj(q1_fast), q2_fast), 1)


    print("\n" + "="*80)
    print(f"{ 'Method':<20} { 'RMSE (deg)':<12} { 'Time (s)':<12} { 'Speedup':<10}")
    print("-" * 80)
    print(f"{ 'MAP-acc':<20} {rmse_opt:<12.4f} {time_opt:<12.4f} {'1.0x':<10}")
    print(f"{ 'MEKF-acc':<20} {rmse_mekf:<12.4f} {time_mekf:<12.4f} {time_opt/time_mekf:.2f}x")
    print("-" * 80)

if __name__ == '__main__':
    main()
