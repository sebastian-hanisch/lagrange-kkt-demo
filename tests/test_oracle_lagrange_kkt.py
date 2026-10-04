"""Unabhaengige Orakel-Tests (anderer Rechenweg als der Code):
- Loesung: Reduktion auf eine Variable. Auf der Volumenkurve h = V0/(pi r^2) ist die Zielfunktion
  f(r) = 2 pi r^2 + 2 V0 / r strikt konvex; das Hoehenlimit wird zu r >= sqrt(V0/(pi h_max)).
  scipy.optimize.minimize_scalar liefert (r*, h*, f*).
- Multiplikatoren: Sensitivitaet des Optimalwerts, lambda = -d f*/d V0, mu = -d f*/d h_max (zentrale
  Differenzen der 1D-Loesung), nicht die Stationaritaetsformel des Codes.
- KKT-System fuer r>0 hat nur die Loesung h = 2r (scipy.optimize.fsolve aus Zufallsstarts), also
  kann es keinen 'falschen stationaeren Punkt' geben; der SLSQP-Fallstrick-Endpunkt ist - wenn er
  schlechter als das Optimum ist - kein stationaerer Punkt."""
import numpy as np
import pytest

import kkt_evaluation as ev
import kkt_functions as fn
import kkt_solver as sv

scipy_opt = pytest.importorskip("scipy.optimize")


def _ref(V0, h_max):
    r_min = np.sqrt(V0 / (np.pi * h_max))
    res = scipy_opt.minimize_scalar(lambda r: 2 * np.pi * r * r + 2 * V0 / r,
                                    bounds=(r_min, 1e3), method="bounded",
                                    options=dict(xatol=1e-12))
    return res.x, V0 / (np.pi * res.x ** 2), res.fun


def test_solution_case_and_multipliers_match_one_dimensional_reduction():
    rng = np.random.default_rng(1)
    cases = [(float(rng.uniform(1, 100)), float(rng.uniform(0.5, 6))) for _ in range(120)]
    cases += [(10.0, 5.0), (10.0, 2.335), (10.0, 2.0), (10.0, 1.0), (1.0, 0.5), (100.0, 6.0)]
    for V0, h_max in cases:
        s = sv.solve_kkt(V0, h_max)
        r, h, f = _ref(V0, h_max)
        assert s.converged
        assert s.r == pytest.approx(r, rel=1e-5) and s.h == pytest.approx(h, rel=1e-5)
        h_star = 2 * (V0 / (2 * np.pi)) ** (1 / 3)
        assert s.case == ("inactive" if h_star <= h_max + 1e-9 else "active")
        e = 1e-4 * V0
        lam_fd = -(_ref(V0 + e, h_max)[2] - _ref(V0 - e, h_max)[2]) / (2 * e)
        assert s.lam == pytest.approx(lam_fd, rel=2e-4, abs=2e-4)
        if s.case == "active":
            eh = 1e-5
            mu_fd = -(_ref(V0, h_max + eh)[2] - _ref(V0, h_max - eh)[2]) / (2 * eh)
            assert s.mu == pytest.approx(mu_fd, rel=2e-3, abs=2e-3)
        else:
            assert s.mu == 0.0


def test_kkt_system_has_only_the_height_equals_diameter_solution():
    rng = np.random.default_rng(2)
    for _ in range(60):
        V0 = float(rng.uniform(0.5, 100))
        x0 = list(np.exp(rng.uniform(-2, 3, size=2))) + [-rng.uniform(0.1, 5)]

        def F(v, V0=V0):
            r, h, lam = v
            return [4 * np.pi * r + 2 * np.pi * h + lam * 2 * np.pi * r * h,
                    2 * np.pi * r + lam * np.pi * r * r, np.pi * r * r * h - V0]

        sol, _info, ier, _msg = scipy_opt.fsolve(F, x0, full_output=True)
        if ier == 1 and sol[0] > 0:
            assert sol[1] / sol[0] == pytest.approx(2.0, rel=1e-6)


def test_scipy_pitfall_endpoint_is_never_a_second_stationary_point():
    out = ev.scipy_local_optimum_pitfall()
    gap = out["gap_percent"]
    x = out["naive_x"]
    g, gg = fn.grad_surface_area(x), fn.grad_volume_constraint(x)
    lam = -(g @ gg) / (gg @ gg)
    residual = np.linalg.norm(g + lam * gg)
    if gap > 1e-3:
        # schlechter als das Optimum => nicht stationaer (das Optimum auf der Volumenkurve ist eindeutig)
        assert residual > 1e-3
    assert out["informed_f"] == pytest.approx(_ref(10.0, 5.0)[2], rel=1e-6)
