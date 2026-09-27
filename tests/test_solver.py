import numpy as np

import kkt_solver as solver


def test_solve_unconstrained_case_matches_hand_derived_formula():
    V0 = 10.0
    r_star = (V0 / (2 * np.pi)) ** (1 / 3)
    h_star = 2 * r_star
    sol = solver.solve_unconstrained_case(V0)
    assert sol.converged
    assert abs(sol.r - r_star) < 1e-6
    assert abs(sol.h - h_star) < 1e-6


def test_solve_active_case_respects_height_limit():
    sol = solver.solve_active_case(V0=10.0, h_max=1.5)
    assert sol.h == 1.5
    assert sol.mu > 0


def test_solve_kkt_picks_inactive_case_when_limit_is_generous():
    sol = solver.solve_kkt(V0=10.0, h_max=5.0)
    assert sol.case == "inactive"
    assert sol.mu == 0.0


def test_solve_kkt_picks_active_case_when_limit_binds():
    sol = solver.solve_kkt(V0=10.0, h_max=1.5)
    assert sol.case == "active"
    assert sol.h == 1.5
    assert sol.mu > 0


def test_complementary_slackness_transition_at_h_star():
    V0 = 10.0
    h_star = 2 * (V0 / (2 * np.pi)) ** (1 / 3)
    sol_above = solver.solve_kkt(V0, h_star + 0.01)
    sol_below = solver.solve_kkt(V0, h_star - 0.01)
    assert sol_above.mu == 0.0
    assert sol_below.mu > 0
