# Drift-Free Joint Kinematics from Dual IMUs

High-performance Python/Cython implementation of the drift-free joint kinematics estimation method from [Weygers & Kok (2020), "Drift-Free Inertial Sensor-Based Joint Kinematics for Long-Term Arbitrary Movements"](https://ieeexplore.ieee.org/document/9044292/).

Two algorithms are provided:

- **MEKF-acc** — online Multiplicative Extended Kalman Filter
- **MAP-acc** — offline Gauss-Newton optimizer for higher accuracy

## Installation

Pre-built wheels with the fast Cython backend are available for Linux, macOS, and Windows (Python 3.10+):

```bash
pip install dfjimu
```

When installing from the source distribution (no matching wheel), a C compiler is required for the Cython extension. If compilation fails, the package falls back to a pure Python backend automatically.

## Usage

```python
from dfjimu import mekf_acc, map_acc

# MEKF-acc — lever arms r1, r2 are auto-estimated when omitted
q1, q2 = mekf_acc(gyr1, gyr2, acc1, acc2, Fs=Fs, q_init=q_init)

# Or provide lever arms explicitly
q1, q2 = mekf_acc(gyr1, gyr2, acc1, acc2, r1, r2, Fs, q_init)

# MAP-acc (optimization-based smoothing)
q1, q2 = map_acc(gyr1, gyr2, acc1, acc2, Fs=Fs, q_init=q_init,
                 cov_w=cov_w, cov_i=cov_i, cov_lnk=cov_lnk)
```

See [`examples/demo.ipynb`](examples/demo.ipynb) for a full walkthrough with plots.

## API

| Function | Description |
|---|---|
| `mekf_acc()` / `MekfAcc` | MEKF-acc filter (online) |
| `map_acc()` / `MapAcc` | MAP-acc optimizer (offline, higher accuracy) |
| `estimate_lever_arms()` | Lever arm estimation from dual-IMU data |

Both `mekf_acc` and `map_acc` accept optional `r1`, `r2` lever arm vectors. When omitted (`None`), they are auto-estimated via `estimate_lever_arms()`.

## Development

```bash
# Install in editable mode with dev tools
pip install -e ".[dev]"

# Build Cython extensions in-place
python setup.py build_ext --inplace

# Run tests
pytest tests/

# Install with Jupyter and matplotlib for the example notebook
pip install -e ".[examples]"
```

## Benchmarks

Evaluate accuracy and speed of the solvers across all datasets:

```bash
python tests/evaluate_package.py
```

## Reference

Weygers, I., & Kok, M. (2020). Drift-Free Inertial Sensor-Based Joint Kinematics for Long-Term Arbitrary Movements. *IEEE Sensors Journal*, 20(14), 7969-7979.
