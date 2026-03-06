"""Tests for numerical stability of quaternion logarithm implementations.

LOGq (acos-based) loses precision near identity quaternions — the normal
operating region of the MAP-acc optimizer. LOGq_stable (atan2-based) fixes this.

Tests are structured so that:
- LOGq failures are expected (xfail)
- LOGq_stable must pass everywhere
"""

import numpy as np
import pytest

from dfjimu.utils.common import LOGq, LOGq_stable, quatmultiply, quatconj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def axis_angle_to_quat(axis, angle):
    """Create quaternion from axis and angle (radians)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    ha = angle / 2.0
    return np.array([np.cos(ha), *(np.sin(ha) * axis)])


def exact_log(axis, angle):
    """Exact quaternion log result for a given axis-angle rotation."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return (angle / 2.0) * axis


# ---------------------------------------------------------------------------
# 1. LOGq (acos): expected failures near identity
# ---------------------------------------------------------------------------

class TestLogQAcos:
    """Document known failures of the acos-based LOGq."""

    @pytest.mark.xfail(reason="acos returns zero below nv < 1e-12 threshold")
    @pytest.mark.parametrize("angle", [1e-9, 1e-12, 1e-13])
    def test_returns_zero_for_small_angles(self, angle):
        """LOGq snaps to zero for angles where nv < 1e-12."""
        q = axis_angle_to_quat([0.0, 0.0, 1.0], angle)
        result = LOGq(q)
        exact = exact_log([0.0, 0.0, 1.0], angle)
        np.testing.assert_allclose(result, exact, atol=1e-15, rtol=1e-10)

    @pytest.mark.xfail(reason="acos precision loss at 1e-6")
    def test_precision_loss_at_1e6(self):
        """LOGq has ~4e-5 relative error at angle=1e-6."""
        angle = 1e-6
        q = axis_angle_to_quat([1.0, 1.0, 1.0], angle)
        result = LOGq(q)
        exact = exact_log([1.0, 1.0, 1.0], angle)
        np.testing.assert_allclose(result, exact, rtol=1e-10)

    @pytest.mark.xfail(reason="acos threshold causes non-monotonic output")
    def test_non_monotonic_near_identity(self):
        """LOGq output norm is not monotonically increasing near identity."""
        axis = np.array([1.0, 0.0, 0.0])
        angles = np.logspace(-14, -10, 50)
        norms = np.array([np.linalg.norm(LOGq(axis_angle_to_quat(axis, a)))
                          for a in angles])
        diffs = np.diff(norms)
        assert np.all(diffs > 0), (
            f"Non-monotonic at {np.sum(diffs <= 0)} points"
        )

    def test_accurate_for_large_angles(self):
        """LOGq is fine for angles above ~1e-3."""
        for angle in [1e-3, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, np.pi]:
            q = axis_angle_to_quat([1.0, 1.0, 1.0], angle)
            exact = exact_log([1.0, 1.0, 1.0], angle)
            np.testing.assert_allclose(LOGq(q), exact, atol=1e-12, rtol=1e-8,
                                       err_msg=f"Failed at angle={angle}")


# ---------------------------------------------------------------------------
# 2. LOGq_stable (atan2): must pass everywhere
# ---------------------------------------------------------------------------

class TestLogQStable:
    """Validate that LOGq_stable handles all cases correctly."""

    @pytest.mark.parametrize("angle", [
        0.0, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3,
        0.01, 0.1, 0.5, 1.0, 2.0, 3.0, np.pi - 0.01, np.pi,
    ])
    def test_accuracy_across_full_range(self, angle):
        """LOGq_stable matches exact result for all rotation angles."""
        axis = [1.0, 1.0, 1.0]
        if angle == 0.0:
            q = np.array([1.0, 0.0, 0.0, 0.0])
            exact = np.zeros(3)
        else:
            q = axis_angle_to_quat(axis, angle)
            exact = exact_log(axis, angle)

        result = LOGq_stable(q)
        np.testing.assert_allclose(result, exact, atol=1e-15, rtol=1e-10)

    @pytest.mark.parametrize("angle", [1e-3, 1e-6, 1e-9, 1e-12, 1e-13])
    def test_small_angle_nonzero(self, angle):
        """LOGq_stable returns nonzero for any nonzero rotation."""
        q = axis_angle_to_quat([0.0, 0.0, 1.0], angle)
        result = LOGq_stable(q)
        expected_norm = angle / 2.0

        assert np.linalg.norm(result) > 0, (
            f"angle={angle}: returned zero"
        )
        np.testing.assert_allclose(
            np.linalg.norm(result), expected_norm, rtol=1e-8,
        )

    def test_monotonicity_near_identity(self):
        """Output norm increases monotonically with angle."""
        axis = np.array([1.0, 0.0, 0.0])
        angles = np.logspace(-14, -1, 100)

        norms = np.array([np.linalg.norm(LOGq_stable(axis_angle_to_quat(axis, a)))
                          for a in angles])

        diffs = np.diff(norms)
        assert np.all(diffs > 0), (
            f"Non-monotonic at {np.sum(diffs <= 0)} points"
        )

    def test_agrees_with_LOGq_for_large_angles(self):
        """LOGq_stable and LOGq agree where LOGq is accurate."""
        for angle in [0.01, 0.1, 0.5, 1.0, 2.0, 3.0]:
            q = axis_angle_to_quat([1.0, 0.0, 0.0], angle)
            np.testing.assert_allclose(
                LOGq_stable(q), LOGq(q), atol=1e-14,
                err_msg=f"Disagreement at angle={angle}",
            )


# ---------------------------------------------------------------------------
# 3. Jacobian accuracy (finite-difference vs analytical)
# ---------------------------------------------------------------------------

class TestMotionModelJacobian:
    """Motion model Jacobian accuracy near identity using LOGq_stable."""

    def _motion_residual(self, q_prev, q_curr, gyr, dt):
        q_rel = quatmultiply(quatconj(q_prev), q_curr)
        return (2.0 / dt) * LOGq_stable(q_rel) - gyr

    def _perturb_quat(self, q, delta):
        ha = delta / 2.0
        norm = np.linalg.norm(ha)
        if norm < 1e-15:
            dq = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            dq = np.array([np.cos(norm), *(np.sin(norm) / norm * ha)])
        return quatmultiply(q, dq)

    @pytest.mark.parametrize("dt,omega", [
        (0.01, 0.1),    # 100 Hz, slow motion
        (0.01, 0.001),  # 100 Hz, near-static
        (0.002, 0.05),  # 500 Hz, moderate motion
    ])
    def test_finite_difference_jacobian(self, dt, omega):
        """Finite-difference Jacobian is diagonal-dominant near identity."""
        gyr = np.array([omega, 0.0, 0.0])

        q_prev = np.array([1.0, 0.0, 0.0, 0.0])
        ha = dt / 2.0 * gyr
        norm_ha = np.linalg.norm(ha)
        dq = np.array([np.cos(norm_ha), *(np.sin(norm_ha) / norm_ha * ha)])
        q_curr = quatmultiply(q_prev, dq)

        eps_fd = 1e-7
        r0 = self._motion_residual(q_prev, q_curr, gyr, dt)

        J_fd = np.zeros((3, 3))
        for j in range(3):
            delta = np.zeros(3)
            delta[j] = eps_fd
            q_pert = self._perturb_quat(q_curr, delta)
            r_pert = self._motion_residual(q_prev, q_pert, gyr, dt)
            J_fd[:, j] = (r_pert - r0) / eps_fd

        diag = np.abs(np.diag(J_fd))
        off_diag = np.abs(J_fd) - np.diag(diag)

        assert np.all(diag > 0), (
            f"Diagonal elements should be positive, got {diag}"
        )
        assert np.max(off_diag) < 0.1 * np.min(diag), (
            f"Not diagonal-dominant. diag={diag}, max_off={np.max(off_diag)}"
        )


# ---------------------------------------------------------------------------
# 4. Full MAP-acc: stable variant on static segment
# ---------------------------------------------------------------------------

class TestMapAccStaticSegment:
    """MAP-acc on data with a long static segment."""

    @staticmethod
    def _make_static_then_moving(N_static=500, N_moving=200, Fs=100.0):
        rng = np.random.default_rng(123)
        N = N_static + N_moving
        gyr_noise = 1e-4

        gyr_static = rng.standard_normal((N_static, 3)) * gyr_noise
        acc_static = np.tile([0.0, 0.0, 9.81], (N_static, 1))

        t_move = np.arange(N_moving) / Fs
        omega_x = 2.0 * np.sin(2 * np.pi * 0.5 * t_move)
        gyr_moving = np.column_stack([omega_x,
                                       np.zeros(N_moving),
                                       np.zeros(N_moving)])
        acc_moving = np.tile([0.0, 0.0, 9.81], (N_moving, 1))

        gyr = np.vstack([gyr_static, gyr_moving])
        acc = np.vstack([acc_static, acc_moving])

        gyr1 = gyr + rng.standard_normal((N, 3)) * gyr_noise
        gyr2 = gyr + rng.standard_normal((N, 3)) * gyr_noise
        acc1 = acc + rng.standard_normal((N, 3)) * 0.01
        acc2 = acc + rng.standard_normal((N, 3)) * 0.01

        r1 = np.array([0.1, 0.0, 0.0])
        r2 = np.array([-0.1, 0.0, 0.0])
        q_init = np.array([1.0, 0.0, 0.0, 0.0])

        return gyr1, gyr2, acc1, acc2, r1, r2, Fs, q_init, N

    def test_stable_convergence_with_static_segment(self):
        """map_acc_stable should converge on data with a static segment."""
        from dfjimu import map_acc_stable

        gyr1, gyr2, acc1, acc2, r1, r2, Fs, q_init, N = (
            self._make_static_then_moving()
        )

        cov_w = np.eye(6) * 1e-4
        cov_i = np.eye(3) * 0.01
        cov_lnk = np.eye(3) * 0.1

        q1, q2 = map_acc_stable(gyr1, gyr2, acc1, acc2, r1, r2, Fs, q_init,
                                cov_w, cov_i, cov_lnk, iterations=15, tol=1e-8)

        norms1 = np.linalg.norm(q1, axis=1)
        norms2 = np.linalg.norm(q2, axis=1)
        np.testing.assert_allclose(norms1, 1.0, atol=1e-6)
        np.testing.assert_allclose(norms2, 1.0, atol=1e-6)

        # Static segment: both sensors should stay near identity
        static_end = 500
        for t in range(static_end):
            angle1 = 2 * np.arccos(np.clip(abs(q1[t, 0]), 0, 1))
            angle2 = 2 * np.arccos(np.clip(abs(q2[t, 0]), 0, 1))
            assert angle1 < np.radians(10), (
                f"t={t}: sensor1 drifted {np.degrees(angle1):.1f} deg"
            )
            assert angle2 < np.radians(10), (
                f"t={t}: sensor2 drifted {np.degrees(angle2):.1f} deg"
            )
