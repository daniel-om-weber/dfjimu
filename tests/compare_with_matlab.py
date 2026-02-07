"""
Compare Cython implementation against MATLAB reference outputs.

Usage:
    python tests/compare_with_matlab.py                     # All datasets
    python tests/compare_with_matlab.py --dataset data_2D_07  # Single dataset
    python tests/compare_with_matlab.py --verbose             # Detailed output

Prerequisites:
    Run tests/matlab_export/export_reference.m in MATLAB first to generate
    reference .mat files in tests/reference_outputs/.
"""

import argparse
import glob
import os
import sys

import numpy as np
import scipy.io

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dfjimu import MapAcc, run_mekf_cython
from dfjimu.utils.common import (
    approxDerivative,
    calcAccatCenter,
    quatconj,
    quatmultiply,
)


def angular_distance(q1, q2):
    """
    Compute angular distance between two quaternion arrays in degrees.
    Handles sign ambiguity (q and -q represent same rotation).

    q1, q2: (N, 4) arrays
    Returns: (N,) array of distances in degrees
    """
    # q_diff = conj(q1) * q2
    q_diff = quatmultiply(quatconj(q1), q2)
    q_diff = np.atleast_2d(q_diff)

    # Handle sign ambiguity
    w = q_diff[:, 0]
    flip = w < 0
    q_diff[flip] = -q_diff[flip]

    # Angular distance = 2 * arccos(|w|)
    w_clipped = np.clip(q_diff[:, 0], -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(w_clipped))


def max_abs_quat_error(q1, q2):
    """
    Max absolute element-wise quaternion error with sign ambiguity handling.
    For each row, compare both q and -q, pick the closer one.
    """
    diff_pos = np.abs(q1 - q2)
    diff_neg = np.abs(q1 + q2)

    # Per-row: pick the sign that gives smaller max error
    max_pos = np.max(diff_pos, axis=1)
    max_neg = np.max(diff_neg, axis=1)

    per_row = np.minimum(max_pos, max_neg)
    return np.max(per_row)


def compare_preprocessing(ref_file, verbose=False):
    """Stage 1: Compare C1/C2 preprocessing between MATLAB and Python."""
    mat = ref_file

    # Extract raw data (MATLAB stores as (3, N))
    gyr = mat['gyr']       # (3, N)
    gyr_2 = mat['gyr_2']   # (3, N)
    acc = mat['acc']        # (3, N)
    acc_2 = mat['acc_2']    # (3, N)
    r1 = mat['r1'].flatten()
    r2 = mat['r2'].flatten()
    Fs = 50.0

    # MATLAB reference C1, C2 are (3, N)
    C1_matlab = mat['C1']   # (3, N)
    C2_matlab = mat['C2']   # (3, N)

    # Python preprocessing (expects (N, 3) inputs)
    gyr_py = gyr.T          # (N, 3)
    gyr2_py = gyr_2.T
    acc_py = acc.T
    acc2_py = acc_2.T

    dgyr1 = np.column_stack([approxDerivative(gyr_py[:, i], Fs) for i in range(3)])
    dgyr2 = np.column_stack([approxDerivative(gyr2_py[:, i], Fs) for i in range(3)])
    C1_py, _ = calcAccatCenter(gyr_py, dgyr1, acc_py, r1)
    C2_py, _ = calcAccatCenter(gyr2_py, dgyr2, acc2_py, r2)

    # C1_py is (N, 3), C1_matlab is (3, N) — transpose for comparison
    C1_matlab_T = C1_matlab.T  # (N, 3)
    C2_matlab_T = C2_matlab.T

    c1_err = np.max(np.abs(C1_py - C1_matlab_T))
    c2_err = np.max(np.abs(C2_py - C2_matlab_T))
    max_err = max(c1_err, c2_err)

    passed = max_err < 1e-10

    if verbose:
        print(f"    C1 max error: {c1_err:.2e}")
        print(f"    C2 max error: {c2_err:.2e}")

    return passed, max_err


def compare_mekf(ref_file, verbose=False):
    """Stage 2: Compare MEKF outputs."""
    mat = ref_file

    # MATLAB reference
    mekf_q_s1_matlab = mat['mekf_q_s1']  # (N, 4)
    mekf_q_s2_matlab = mat['mekf_q_s2']  # (N, 4)

    # Raw data (MATLAB format: (3, N))
    gyr = mat['gyr'].T       # -> (N, 3)
    gyr_2 = mat['gyr_2'].T
    acc = mat['acc'].T
    acc_2 = mat['acc_2'].T

    # Use MATLAB-computed C1, C2 to isolate algorithm from preprocessing
    C1 = mat['C1'].T         # (N, 3)
    C2 = mat['C2'].T

    # Use MATLAB-computed cov_w for Q
    cov_w = mat['cov_w']     # (6, 6)
    Q_vec = np.diag(cov_w)   # Extract diagonal as vector

    Fs = 50.0
    q_init = np.array([1.0, 0.0, 0.0, 0.0])

    # Run Cython MEKF
    q_s1_py, q_s2_py = run_mekf_cython(
        gyr.astype(np.float64),
        gyr_2.astype(np.float64),
        acc.astype(np.float64),
        acc_2.astype(np.float64),
        C1.astype(np.float64),
        C2.astype(np.float64),
        Fs,
        q_init.astype(np.float64),
        Q_vec.astype(np.float64),
    )

    # Compare
    ang_s1 = angular_distance(q_s1_py, mekf_q_s1_matlab)
    ang_s2 = angular_distance(q_s2_py, mekf_q_s2_matlab)

    max_ang_s1 = np.max(ang_s1)
    max_ang_s2 = np.max(ang_s2)
    mean_ang_s1 = np.mean(ang_s1)
    mean_ang_s2 = np.mean(ang_s2)

    max_elem_s1 = max_abs_quat_error(q_s1_py, mekf_q_s1_matlab)
    max_elem_s2 = max_abs_quat_error(q_s2_py, mekf_q_s2_matlab)

    max_ang = max(max_ang_s1, max_ang_s2)
    mean_ang = max(mean_ang_s1, mean_ang_s2)
    max_elem = max(max_elem_s1, max_elem_s2)

    # Threshold: 0.01 degrees
    passed = max_ang < 0.01

    if verbose:
        print(f"    S1: max_ang={max_ang_s1:.4f}deg, mean={mean_ang_s1:.4f}deg, max_elem={max_elem_s1:.2e}")
        print(f"    S2: max_ang={max_ang_s2:.4f}deg, mean={mean_ang_s2:.4f}deg, max_elem={max_elem_s2:.2e}")

    return passed, max_ang, mean_ang, max_elem


def compare_opt(ref_file, verbose=False):
    """Stage 3: Compare GN optimizer outputs."""
    mat = ref_file

    # MATLAB reference
    opt_q_s1_matlab = mat['opt_q_s1']  # (N, 4)
    opt_q_s2_matlab = mat['opt_q_s2']  # (N, 4)

    # Raw data
    gyr = mat['gyr'].T       # -> (N, 3)
    gyr_2 = mat['gyr_2'].T
    acc = mat['acc'].T
    acc_2 = mat['acc_2'].T
    r1 = mat['r1'].flatten()
    r2 = mat['r2'].flatten()

    # Use MATLAB-computed covariances
    cov_w = mat['cov_w']     # (6, 6)
    cov_i = mat['cov_i']     # (3, 3)
    cov_lnk = mat['cov_lnk'] # (3, 3)

    Fs = 50.0
    q_init = np.array([1.0, 0.0, 0.0, 0.0])

    # Run MAP-acc solver (uses MATLAB covariances)
    solver = MapAcc(gyr, gyr_2, acc, acc_2, r1, r2, Fs, cov_w, cov_i, cov_lnk)
    q_s1_py, q_s2_py = solver.solve(q_init, iterations=10)

    # Compare
    ang_s1 = angular_distance(q_s1_py, opt_q_s1_matlab)
    ang_s2 = angular_distance(q_s2_py, opt_q_s2_matlab)

    max_ang_s1 = np.max(ang_s1)
    max_ang_s2 = np.max(ang_s2)
    mean_ang_s1 = np.mean(ang_s1)
    mean_ang_s2 = np.mean(ang_s2)

    max_elem_s1 = max_abs_quat_error(q_s1_py, opt_q_s1_matlab)
    max_elem_s2 = max_abs_quat_error(q_s2_py, opt_q_s2_matlab)

    max_ang = max(max_ang_s1, max_ang_s2)
    mean_ang = max(mean_ang_s1, mean_ang_s2)
    max_elem = max(max_elem_s1, max_elem_s2)

    # Threshold: 0.01 degrees
    passed = max_ang < 0.01

    if verbose:
        print(f"    S1: max_ang={max_ang_s1:.4f}deg, mean={mean_ang_s1:.4f}deg, max_elem={max_elem_s1:.2e}")
        print(f"    S2: max_ang={max_ang_s2:.4f}deg, mean={mean_ang_s2:.4f}deg, max_elem={max_elem_s2:.2e}")

    return passed, max_ang, mean_ang, max_elem


def process_dataset(ref_path, verbose=False):
    """Run all three comparison stages on a single dataset."""
    dataset_name = os.path.basename(ref_path).replace('ref_', '').replace('.mat', '')

    mat = scipy.io.loadmat(ref_path, squeeze_me=True)

    results = {'name': dataset_name}

    # Stage 1: Preprocessing
    try:
        passed, max_err = compare_preprocessing(mat, verbose)
        results['preproc'] = {'passed': passed, 'max_err': max_err}
    except Exception as e:
        results['preproc'] = {'passed': False, 'error': str(e)}

    # Stage 2: MEKF
    try:
        passed, max_ang, mean_ang, max_elem = compare_mekf(mat, verbose)
        results['mekf'] = {
            'passed': passed,
            'max_ang': max_ang,
            'mean_ang': mean_ang,
            'max_elem': max_elem,
        }
    except Exception as e:
        results['mekf'] = {'passed': False, 'error': str(e)}

    # Stage 3: GN Optimizer
    try:
        passed, max_ang, mean_ang, max_elem = compare_opt(mat, verbose)
        results['opt'] = {
            'passed': passed,
            'max_ang': max_ang,
            'mean_ang': mean_ang,
            'max_elem': max_elem,
        }
    except Exception as e:
        results['opt'] = {'passed': False, 'error': str(e)}

    return results


def print_results(all_results):
    """Print a summary table of all results."""
    header = (
        f"{'Dataset':<12} "
        f"{'Preproc':<10} "
        f"{'MEKF max°':<12} {'MEKF mean°':<12} {'MEKF':<6} "
        f"{'OPT max°':<12} {'OPT mean°':<12} {'OPT':<6}"
    )
    print()
    print(header)
    print("-" * len(header))

    n_pass = {'preproc': 0, 'mekf': 0, 'opt': 0}
    n_total = 0

    for r in all_results:
        n_total += 1
        name = r['name']

        # Preprocessing
        pp = r['preproc']
        if 'error' in pp:
            pp_str = f"ERR"
        else:
            pp_str = f"{'PASS' if pp['passed'] else 'FAIL'} {pp['max_err']:.0e}"
            if pp['passed']:
                n_pass['preproc'] += 1

        # MEKF
        mk = r['mekf']
        if 'error' in mk:
            mk_max = mk_mean = "ERR"
            mk_status = "ERR"
        else:
            mk_max = f"{mk['max_ang']:.4f}"
            mk_mean = f"{mk['mean_ang']:.4f}"
            mk_status = "PASS" if mk['passed'] else "FAIL"
            if mk['passed']:
                n_pass['mekf'] += 1

        # OPT
        op = r['opt']
        if 'error' in op:
            op_max = op_mean = "ERR"
            op_status = "ERR"
        else:
            op_max = f"{op['max_ang']:.4f}"
            op_mean = f"{op['mean_ang']:.4f}"
            op_status = "PASS" if op['passed'] else "FAIL"
            if op['passed']:
                n_pass['opt'] += 1

        print(
            f"{name:<12} "
            f"{pp_str:<10} "
            f"{mk_max:<12} {mk_mean:<12} {mk_status:<6} "
            f"{op_max:<12} {op_mean:<12} {op_status:<6}"
        )

    print("-" * len(header))
    print(
        f"{'TOTALS':<12} "
        f"{n_pass['preproc']}/{n_total:<8} "
        f"{'':12} {'':12} {n_pass['mekf']}/{n_total:<4} "
        f"{'':12} {'':12} {n_pass['opt']}/{n_total:<4}"
    )


def main():
    parser = argparse.ArgumentParser(description='Compare Cython vs MATLAB reference')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset name (e.g. data_2D_07)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed per-sensor metrics')
    parser.add_argument('--ref-dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'reference_outputs'),
                        help='Directory containing reference .mat files')
    args = parser.parse_args()

    ref_dir = args.ref_dir

    if not os.path.isdir(ref_dir):
        print(f"Error: Reference directory not found: {ref_dir}")
        print("Run tests/matlab_export/export_reference.m in MATLAB first.")
        sys.exit(1)

    if args.dataset:
        ref_files = [os.path.join(ref_dir, f'ref_{args.dataset}.mat')]
        if not os.path.exists(ref_files[0]):
            print(f"Error: Reference file not found: {ref_files[0]}")
            sys.exit(1)
    else:
        ref_files = sorted(glob.glob(os.path.join(ref_dir, 'ref_data_*.mat')))

    if not ref_files:
        print(f"Error: No reference files found in {ref_dir}")
        print("Run tests/matlab_export/export_reference.m in MATLAB first.")
        sys.exit(1)

    print(f"Comparing Cython vs MATLAB on {len(ref_files)} dataset(s)...")
    print(f"Tolerance: < 0.01 degrees angular distance")

    all_results = []
    for ref_path in ref_files:
        dataset_name = os.path.basename(ref_path).replace('ref_', '').replace('.mat', '')
        if args.verbose:
            print(f"\n--- {dataset_name} ---")

        results = process_dataset(ref_path, verbose=args.verbose)
        all_results.append(results)

    print_results(all_results)

    # Exit code: 0 if all pass, 1 if any fail
    all_pass = all(
        r['preproc'].get('passed', False)
        and r['mekf'].get('passed', False)
        and r['opt'].get('passed', False)
        for r in all_results
    )
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
