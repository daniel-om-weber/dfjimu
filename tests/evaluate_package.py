import numpy as np
import scipy.io
import glob
import os
import time

from dfjimu import map_acc, mekf_acc
from dfjimu.utils.common import quatmultiply, quatconj, angDist

def calculate_rmse(q_ref, q_est, shift_est=0):
    if q_ref.shape[0] == 4 and q_ref.shape[1] > 4: q_ref = q_ref.T
    N_ref = q_ref.shape[0]
    N_est = q_est.shape[0]
    limit = min(N_ref, N_est - shift_est)
    if limit > N_ref - 1: limit = N_ref - 1
    dists = [angDist(q_ref[t], q_est[t + shift_est]) for t in range(limit)]
    return np.sqrt(np.mean(np.array(dists)**2))

def process_file(file_path):
    try:
        mat = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        data = mat['data']
        r1 = np.atleast_1d(-data.r_12).flatten()
        r2 = np.atleast_1d(-data.r_21).flatten()
        sensor_data = data.sensorData
        acc1 = sensor_data[:, 0:3]; gyr1 = sensor_data[:, 3:6]
        acc2 = sensor_data[:, 6:9]; gyr2 = sensor_data[:, 9:12]
        qGS1_ref = data.ref[:, 0:4]; qGS2_ref = data.ref[:, 4:8]
        qREF = quatmultiply(quatconj(qGS1_ref), qGS2_ref)
        Fs = 50.0; q_init = np.array([1.0, 0.0, 0.0, 0.0])

        # Setup Covariances
        limit = 50
        var1 = np.var(gyr1[:limit], axis=0)
        var2 = np.var(gyr2[:limit], axis=0)
        cov_w = np.eye(6) * 0.01
        cov_i = np.eye(3) * 0.1
        cov_lnk = np.eye(3)

        # 1. MAP-acc
        start = time.time()
        q1, q2 = map_acc(gyr1, gyr2, acc1, acc2, r1, r2, Fs, q_init,
                         cov_w, cov_i, cov_lnk, iterations=10)
        time_gn = time.time() - start
        rmse_gn = calculate_rmse(qREF, quatmultiply(quatconj(q1), q2), 1)

        # 2. MEKF-acc
        start = time.time()
        q1, q2 = mekf_acc(gyr1, gyr2, acc1, acc2, r1, r2, Fs, q_init)
        time_mekf = time.time() - start
        rmse_mekf = calculate_rmse(qREF, quatmultiply(quatconj(q1), q2), 1)

        return {
            'file': os.path.basename(file_path).replace('data_', '').replace('.mat', ''),
            'rmse_gn': rmse_gn, 'time_gn': time_gn,
            'rmse_mekf': rmse_mekf, 'time_mekf': time_mekf
        }
    except Exception as e:
        return {'file': os.path.basename(file_path), 'error': str(e)}

def main():
    files = sorted(glob.glob('data/data_*.mat'))
    print(f"Comparing Solvers on {len(files)} datasets...")
    print(f"{'Dataset':<8} {'Err(MAP)':<9} {'Err(MK)':<9} {'T(MAP)':<9} {'T(MK)':<9} {'Sp(MK)':<6}")
    print("-" * 55)

    sp_mk = []

    for f in files:
        r = process_file(f)
        if 'error' in r:
            print(f"{r['file']:<8} ERROR: {r['error']}")
            continue

        s_mk = r['time_gn'] / r['time_mekf']
        sp_mk.append(s_mk)

        print(f"{r['file']:<8} {r['rmse_gn']:.2f}     {r['rmse_mekf']:.2f}     {r['time_gn']:.3f}s   {r['time_mekf']:.4f}s  {s_mk:.1f}x")

    print("-" * 55)
    if sp_mk:
        print(f"Average MEKF-acc speedup vs MAP-acc: {np.mean(sp_mk):.1f}x")

if __name__ == '__main__':
    main()
