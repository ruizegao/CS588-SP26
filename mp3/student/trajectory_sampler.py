"""
Task 1: Frenet Trajectory Sampler
==================================

Implement a Werling-style sampling-based trajectory generator that operates in
the Frenet frame of a lane-level reference path.

Each candidate trajectory is defined by two independent polynomial profiles:
  - A quintic polynomial d(t) for the lateral Frenet offset.
  - A quartic polynomial s(t) for the longitudinal arc-length.

The candidate set is the Cartesian product of
    DEFAULT_LANE_OFFSETS × TARGET_SPEED_DELTAS × TARGET_ACCELS
producing 9 × 8 × 7 = 504 candidates by default.

Workflow for each candidate:
  1. Fit a QuinticPolynomial to match d(0), d'(0), d''(0) at the ego state and
     d(T)=d_target, d'(T)=0, d''(T)=0 at the terminal condition.
  2. Fit a QuarticPolynomial to match s(0), s'(0), s''(0) at the ego state and
     s'(T)=v_target, s''(T)=a_target at the terminal condition.
  3. Evaluate both polynomials at num_samples time steps to get (s, d) profiles.
  4. Convert to Cartesian (x, y) using frenet_to_cartesian.
  5. Numerically differentiate to fill speed, accel, curvature, s_jerk, d_jerk.

Reference: Werling et al., "Optimal Trajectory Generation for Dynamic Street
Scenarios in a Frenet Frame", ICRA 2010.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from waymax import datatypes

from planner.frenet_utils import frenet_to_cartesian
from planner.frenet_utils import project_to_frenet
from planner.types import ReferencePath
from planner.types import TrajectorySample


TARGET_SPEED_DELTAS = np.array(
    [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0],
    dtype=float,
)
TARGET_ACCELS = np.array([-8.0, -6.0, -4.0, -2.5, -1.0, 0.0, 1.5], dtype=float)


# ── TODO 1.1 — Quintic polynomial ─────────────────────────────────────────────

@dataclass
class QuinticPolynomial:
    """Quintic polynomial matching position, velocity, and acceleration at both endpoints.

    The polynomial has the form:
        p(t) = a0 + a1·t + a2·t² + a3·t³ + a4·t⁴ + a5·t⁵

    Given boundary conditions at t=0 and t=T, three coefficients (a0, a1, a2)
    are determined directly from the initial state, and (a3, a4, a5) are found
    by solving a 3×3 linear system from the terminal conditions.

    Attributes:
        coeffs: Coefficients [a0, a1, a2, a3, a4, a5], ordered low-to-high.
    """

    coeffs: np.ndarray

    @staticmethod
    def fit(
        start:     tuple[float, float, float],
        end:       tuple[float, float, float],
        horizon_s: float,
    ) -> "QuinticPolynomial":
        """TODO(Task 1.1a): Fit a quintic polynomial to six boundary conditions.

        The polynomial p(t) must satisfy:
            p(0)   = p0,   p'(0)  = v0,   p''(0) = a0   (from `start`)
            p(T)   = p1,   p'(T)  = v1,   p''(T) = a1   (from `end`)

        The first three coefficients follow directly:
            a0 = p0,   a1 = v0,   a2 = a0_start / 2

        The remaining three (a3, a4, a5) satisfy the linear system
            M · [a3, a4, a5]ᵀ = r
        where M and r are derived by substituting the terminal conditions into
        p(T), p'(T), and p''(T) and isolating the unknown coefficients.

        Args:
            start: (position, velocity, acceleration) at t=0.
            end:   (position, velocity, acceleration) at t=T.
            horizon_s: Duration T in seconds.

        Returns:
            QuinticPolynomial with fitted coefficients.
        """
        # ======= STUDENT TODO START (edit only inside this block) =======
        # TODO(student): implement QuinticPolynomial.fit
        #
        # Steps:
        #   1. Unpack start into (p0, v0, a0_start) and end into (p1, v1, a1_end).
        #   2. Set a0 = p0, a1 = v0, a2 = a0_start / 2 directly.
        #   3. Build the 3×3 matrix M whose rows correspond to the p(T), p'(T),
        #      and p''(T) equations for the unknowns [a3, a4, a5].
        #   4. Build the right-hand side vector r by subtracting the known a0, a1,
        #      a2 contributions from the terminal conditions p1, v1, a1_end.
        #   5. Solve M · [a3, a4, a5]ᵀ = r with a linear solver.
        #   6. Return QuinticPolynomial with coefficients [a0, a1, a2, a3, a4, a5].

        p0, v0, a0_start = start
        p1, v1, a1_start = end

        a0 = p0
        a1 = v0
        a2 = a0_start / 2.

        M = np.array([
            [horizon_s**3, horizon_s**4, horizon_s**5],
            [3*horizon_s**2, 4*horizon_s**3, 5*horizon_s**4],
            [6*horizon_s, 12*horizon_s**2, 20*horizon_s**3],
        ])
        r = np.array([
            [p1 - a0 - a1 * horizon_s - a2 * horizon_s**2],
            [0. - a1 - 2*a2*horizon_s],
            [0. - 2*a2],
        ])
        a3, a4, a5 = np.linalg.solve(M, r).squeeze(-1)
        return QuinticPolynomial(np.array([a0, a1, a2, a3, a4, a5]))
        # ======= STUDENT TODO END (do not change code outside this block) =======

    def evaluate(self, t: np.ndarray, order: int = 0) -> np.ndarray:
        """TODO(Task 1.1b): Evaluate the polynomial or one of its derivatives.

        The polynomial and its first three derivatives are:
            p(t)    = a0 + a1·t + a2·t² + a3·t³ + a4·t⁴ + a5·t⁵
            p'(t)   = a1 + 2a2·t + 3a3·t² + 4a4·t³ + 5a5·t⁴
            p''(t)  = 2a2 + 6a3·t + 12a4·t² + 20a5·t³
            p'''(t) = 6a3 + 24a4·t + 60a5·t²

        Args:
            t:     Time samples of shape (N,) in seconds.
            order: Derivative order in {0, 1, 2, 3}.

        Returns:
            Values of shape (N,).
        """
        # ======= STUDENT TODO START (edit only inside this block) =======
        # TODO(student): implement QuinticPolynomial.evaluate
        #
        # Steps:
        #   1. Extract coefficients from self.coeffs.
        #   2. Return the polynomial (order=0) or the appropriate derivative
        #      (order=1, 2, or 3) evaluated at all time samples t.

        a0, a1, a2, a3, a4, a5 = self.coeffs
        if order == 0:
            return a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
        if order == 1:
            return a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
        if order == 2:
            return 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
        if order == 3:
            return 6*a3 + 24*a4*t + 60*a5*t**2
        # ======= STUDENT TODO END (do not change code outside this block) =======


# ── TODO 1.2 — Quartic polynomial ─────────────────────────────────────────────

@dataclass
class QuarticPolynomial:
    """Quartic polynomial matching start state and terminal speed/acceleration.

    The polynomial has the form:
        p(t) = a0 + a1·t + a2·t² + a3·t³ + a4·t⁴

    Unlike the quintic, the terminal position p(T) is free; only the terminal
    velocity and acceleration are constrained.  This leaves a 2×2 system for
    (a3, a4).

    Attributes:
        coeffs: Coefficients [a0, a1, a2, a3, a4], ordered low-to-high.
    """

    coeffs: np.ndarray

    @staticmethod
    def fit(
        start:      tuple[float, float, float],
        end_speed:  float,
        end_accel:  float,
        horizon_s:  float,
    ) -> "QuarticPolynomial":
        """TODO(Task 1.2a): Fit a quartic polynomial to five boundary conditions.

        The polynomial p(t) must satisfy:
            p(0)   = p0,   p'(0)  = v0,   p''(0) = a0_start   (from `start`)
            p'(T)  = end_speed,            p''(T) = end_accel

        The first three coefficients follow directly:
            a0 = p0,   a1 = v0,   a2 = a0_start / 2

        The remaining two (a3, a4) satisfy the 2×2 linear system
            M · [a3, a4]ᵀ = r
        derived from the terminal velocity and acceleration constraints.

        Args:
            start:      (position, velocity, acceleration) at t=0.
            end_speed:  Desired p'(T) in m/s.
            end_accel:  Desired p''(T) in m/s².
            horizon_s:  Duration T in seconds.

        Returns:
            QuarticPolynomial with fitted coefficients.
        """
        # ======= STUDENT TODO START (edit only inside this block) =======
        # TODO(student): implement QuarticPolynomial.fit
        #
        # Steps:
        #   1. Unpack start into (p0, v0, a0_start).
        #   2. Set a0 = p0, a1 = v0, a2 = a0_start / 2 directly.
        #   3. Build the 2×2 matrix M for the unknowns [a3, a4] from the
        #      terminal p'(T) and p''(T) equations.
        #   4. Build the right-hand side r by subtracting known contributions
        #      from end_speed and end_accel.
        #   5. Solve M · [a3, a4]ᵀ = r with a linear solver.
        #   6. Return QuarticPolynomial with coefficients [a0, a1, a2, a3, a4].

        p0, v0, a0_start = start
        a0 = p0
        a1 = v0
        a2 = a0_start / 2.

        M = np.array([
            [3 * horizon_s**2, 4 * horizon_s**3],
            [6 * horizon_s, 12 * horizon_s**2]
        ])

        r = np.array([
            [end_speed - a1 - 2*a2*horizon_s],
            [end_accel - 2*a2]
        ])

        a3, a4 = np.linalg.solve(M, r).squeeze(-1)
        return QuarticPolynomial(np.array([a0, a1, a2, a3, a4]))
        # ======= STUDENT TODO END (do not change code outside this block) =======

    def evaluate(self, t: np.ndarray, order: int = 0) -> np.ndarray:
        """TODO(Task 1.2b): Evaluate the polynomial or one of its derivatives.

        The polynomial and its first three derivatives are:
            p(t)    = a0 + a1·t + a2·t² + a3·t³ + a4·t⁴
            p'(t)   = a1 + 2a2·t + 3a3·t² + 4a4·t³
            p''(t)  = 2a2 + 6a3·t + 12a4·t²
            p'''(t) = 6a3 + 24a4·t

        Args:
            t:     Time samples of shape (N,) in seconds.
            order: Derivative order in {0, 1, 2, 3}.

        Returns:
            Values of shape (N,).
        """
        # ======= STUDENT TODO START (edit only inside this block) =======
        # TODO(student): implement QuarticPolynomial.evaluate
        #
        # Steps:
        #   1. Extract coefficients from self.coeffs.
        #   2. Return the polynomial (order=0) or the appropriate derivative
        #      (order=1, 2, or 3) evaluated at all time samples t.

        a0, a1, a2, a3, a4 = self.coeffs

        if order == 0:
            return a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4
        if order == 1:
            return a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3
        if order == 2:
            return 2*a2 + 6*a3*t + 12*a4*t**2
        if order == 3:
            return 6*a3 + 24*a4*t
        # ======= STUDENT TODO END (do not change code outside this block) =======


# ── TODO 1.3 — Assemble the full candidate set ────────────────────────────────

def sample_trajectories(
    state:          datatypes.SimulatorState,
    reference_path: ReferencePath,
    horizon_s:      float = 3.0,
    num_samples:    int   = 31,
) -> list[TrajectorySample]:
    """TODO(Task 1.3): Generate all 504 Frenet candidate trajectories.

    Read the ego state, decompose its velocity into Frenet components, then
    iterate over all (d_target, v_target, a_target) combinations.  For each
    combination, fit a QuinticPolynomial for the lateral profile and a
    QuarticPolynomial for the longitudinal profile, evaluate them at
    num_samples time steps, convert to Cartesian, and numerically differentiate
    to populate all TrajectorySample fields.

    Steps:
        1. Read ego x, y, yaw, speed from state.sim_trajectory.
        2. Project ego position to Frenet (s0, d0) via project_to_frenet.
        3. Get the reference heading at s0 via frenet_to_cartesian at offset 0.
           Decompose ego speed into Frenet components:
               heading_error = ego_yaw - ref_heading
               s_dot0 = ego_speed * cos(heading_error)
               d_dot0 = ego_speed * sin(heading_error)
        4. Build the shared time array: times = linspace(0, horizon_s, num_samples).
        5. Compute target speeds as max(0, ego_speed + TARGET_SPEED_DELTAS).
        6. For each d_target in reference_path.lane_offsets:
               Fit QuinticPolynomial(start=(d0, d_dot0, 0), end=(d_target, 0, 0), T).
               Evaluate d = poly.evaluate(times, order=0).
               Evaluate d_dot = poly.evaluate(times, order=1).
               Evaluate d_jerk = poly.evaluate(times, order=3).
               For each v_target and a_target:
                   Fit QuarticPolynomial(start=(s0, s_dot0, 0), end_speed=v_target,
                                         end_accel=a_target, T).
                   Evaluate s = poly.evaluate(times, order=0).
                   Clip s_dot = max(poly.evaluate(times, order=1), 0) (no reversing).
                   Evaluate s_jerk = poly.evaluate(times, order=3).
                   Convert to Cartesian: (xy, center_heading) = frenet_to_cartesian(s, d).
                   Compute yaw = center_heading + arctan2(d_dot, max(s_dot, 1e-3)).
                   Numerically differentiate xy to get speed and accel.
                   Compute curvature = yaw_rate / max(speed, 1e-3).
                   Append TrajectorySample with all fields.
        7. Return the list of TrajectorySample objects.

    Args:
        state:          Current simulator state containing the ego state.
        reference_path: Lane-level path used as Frenet reference.
        horizon_s:      Planning horizon in seconds.
        num_samples:    Number of discrete time samples over the horizon.

    Returns:
        List of up to 504 TrajectorySample objects.
    """

    # read current SDV state (step 1)
    sdc_mask = np.asarray(state.object_metadata.is_sdc).astype(bool)
    if not sdc_mask.any():
      raise ValueError("Trajectory sampling assumes an SDC is present.")
    ego_idx = int(np.flatnonzero(sdc_mask)[0])
    timestep = int(state.timestep)
    ego_xy = np.asarray(state.sim_trajectory.xy[ego_idx, timestep])[None, :]
    ego_speed = float(np.asarray(state.sim_trajectory.speed[ego_idx, timestep]))
    ego_yaw = float(np.asarray(state.sim_trajectory.yaw[ego_idx, timestep]))

    # step 4
    times = np.linspace(0.0, horizon_s, num_samples)
    target_speeds = np.maximum(0.0, ego_speed + TARGET_SPEED_DELTAS)
    target_accels = TARGET_ACCELS
    target_offsets = reference_path.lane_offsets

    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): implement sample_trajectories
    s0, d0 = project_to_frenet(reference_path, ego_xy)
    _, ref_heading = frenet_to_cartesian(reference_path, s0, np.array([0.0]))
    heading_error = ego_yaw - ref_heading
    s_dot0 = ego_speed * np.cos(heading_error)
    d_dot0 = ego_speed * np.sin(heading_error)

    output_traj_samples = []
    for d_target in target_offsets:
        quint_poly = QuinticPolynomial.fit(start=(d0[0], d_dot0[0], 0), end=(d_target, 0, 0), horizon_s=horizon_s)
        d = quint_poly.evaluate(times, order=0)
        d_dot = quint_poly.evaluate(times, order=1)
        d_jerk = quint_poly.evaluate(times, order=3)
        for v_target in target_speeds:
            for a_target in target_accels:
                quart_poly = QuarticPolynomial.fit(start=(s0[0], s_dot0[0], 0), end_speed=v_target,
                                             end_accel=a_target, horizon_s=horizon_s)
                s = quart_poly.evaluate(times, order=0)
                s_dot = np.maximum(quart_poly.evaluate(times, order=1), 0.0)
                s_jerk = quart_poly.evaluate(times, order=3)
                xy, center_heading = frenet_to_cartesian(reference_path, s, d)
                yaw = center_heading + np.atan2(d_dot, np.maximum(s_dot, 1e-3))

                speed_xy = np.gradient(xy, horizon_s/(num_samples-1), axis=0)
                speed = np.linalg.norm(speed_xy, axis=1)

                accel_xy = np.gradient(speed_xy, horizon_s/(num_samples-1), axis=0)
                accel = np.linalg.norm(accel_xy, axis=1)

                yaw_rate = np.gradient(yaw, horizon_s/(num_samples-1))
                curvature = yaw_rate / np.maximum(speed, 1e-3)
                new_traj = TrajectorySample(
                    times=times,
                    s=s,
                    d=d,
                    x=xy[:, 0],
                    y=xy[:, 1],
                    yaw=yaw,
                    speed=s_dot,
                    accel=accel,
                    curvature=curvature,
                    s_jerk=s_jerk,
                    d_jerk=d_jerk,
                    target_offset=d_target,
                    target_speed=v_target,
                    target_accel=a_target,
                )
                output_traj_samples.append(new_traj)

    return output_traj_samples
    # ======= STUDENT TODO END (do not change code outside this block) =======
