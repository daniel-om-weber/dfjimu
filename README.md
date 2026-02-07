# Drift-Free Joint Kinematics from Dual IMUs

High-performance Python/Cython implementation of the drift-free joint kinematics estimation method from [Weygers & Kok (2020), "Drift-Free Inertial Sensor-Based Joint Kinematics for Long-Term Arbitrary Movements"](https://ieeexplore.ieee.org/document/9044292/).

It includes:
1.  **MAP-acc** (optimization-based smoothing): Gauss-Newton optimizer for high-accuracy offline joint kinematics.
2.  **MEKF-acc** (Multiplicative Extended Kalman Filter): Online filtering.

## Package Structure

The core logic is in the `dfjimu` package.

-   **`dfjimu/`**: The Python package source.
    -   `map_acc()` / `MapAcc`: MAP-acc optimizer (uses Cython).
    -   `mekf_acc()` / `MekfAcc`: MEKF-acc filter (uses Cython).
    -   `_python/`: Pure Python reference implementations for educational/debugging purposes.

## Installation

You need a C compiler (GCC/Clang) for the Cython extensions.

```bash
# Core package
uv pip install -e .

# With Jupyter and matplotlib for running the example notebook
uv pip install -e ".[examples]"

# With Cython build tools for development
uv pip install -e ".[dev]"
```

## Running Benchmarks

Evaluate accuracy and speed of the solvers on the dataset:

```bash
uv run tests/evaluate_package.py
```

## Usage Example

```python
from dfjimu import mekf_acc, map_acc

# MEKF-acc (online filtering)
q1, q2 = mekf_acc(gyr1, gyr2, acc1, acc2, r1, r2, Fs, q_init)

# MAP-acc (optimization-based smoothing)
q1, q2 = map_acc(gyr1, gyr2, acc1, acc2, r1, r2, Fs, q_init,
                 cov_w, cov_i, cov_lnk)
```

See [`examples/demo.ipynb`](examples/demo.ipynb) for a full walkthrough with plots.

## Reference

Weygers, I., & Kok, M. (2020). Drift-Free Inertial Sensor-Based Joint Kinematics for Long-Term Arbitrary Movements. *IEEE Sensors Journal*, 20(14), 7969-7979.
