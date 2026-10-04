"""Kennzahlen: Settings-Dataclass, analyse()-Einstiegspunkt, Hand-Ableitungs-Check,
SciPy-Kreuzpruefung (inkl. eines ehrlichen Nebenbefunds: SLSQP findet nicht immer das globale
Optimum), komplementaerer-Schlupf-Sweep, Gradienten-Check."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

import kkt_functions as fn
import kkt_solver as solver


@dataclass(frozen=True)
class Settings:
    V0: float
    h_max: float


def analyse(settings: Settings) -> dict:
    solution = solver.solve_kkt(settings.V0, settings.h_max)
    f_val = fn.surface_area(np.array([solution.r, solution.h]))
    return {"solution": solution, "f_val": f_val}


def hand_derived_check(V0: float = 10.0) -> dict:
    """Fuer ein sehr grosses Hoehenlimit (nie bindend) muss die numerische Loesung exakt der
    von Hand hergeleiteten Formel r*=(V0/(2*pi))^(1/3), h*=2r* entsprechen."""
    r_star = (V0 / (2 * np.pi)) ** (1 / 3)
    h_star = 2 * r_star
    solution = solver.solve_unconstrained_case(V0)
    return {"r_star": r_star, "h_star": h_star, "r_numeric": solution.r, "h_numeric": solution.h,
            "n_iter": solution.n_iter, "converged": solution.converged,
            "max_abs_err": float(max(abs(solution.r - r_star), abs(solution.h - h_star)))}


def unconstrained_case_robustness(V0_values=(0.5, 1.0, 5.0, 10.0, 50.0, 100.0)) -> list:
    """Der informierte Startwert (aus Dimensionsanalyse) konvergiert zuverlaessig ueber
    verschiedene Groessenordnungen von V0 - im Gegensatz zu einem neutralen Startwert wie
    (1,1,-1), bei dem das gedaempfte Newton-Verfahren fuer die meisten V0 stagniert (kein falscher
    stationaerer Punkt: fuer r>0 hat das KKT-System nur die Loesung h=2r)."""
    rows = []
    for V0 in V0_values:
        r_star = (V0 / (2 * np.pi)) ** (1 / 3)
        h_star = 2 * r_star
        sol = solver.solve_unconstrained_case(V0)
        matches = abs(sol.r - r_star) < 1e-6 and abs(sol.h - h_star) < 1e-6
        rows.append({"V0": V0, "n_iter": sol.n_iter, "converged": sol.converged,
                    "matches_analytic": matches})
    return rows


def complementary_slackness_sweep(V0: float = 10.0,
                                  h_max_values=(5.0, 3.0, 2.5, 2.4, 2.34, 2.335, 2.0, 1.5, 1.0)) -> list:
    """mu springt von 0 (Hoehenlimit inaktiv) auf einen positiven Wert (aktiv), exakt am
    Uebergang h_max = h_star."""
    rows = []
    for h_max in h_max_values:
        solution = solver.solve_kkt(V0, h_max)
        f_val = fn.surface_area(np.array([solution.r, solution.h]))
        rows.append({"h_max": h_max, "r": solution.r, "h": solution.h, "mu": solution.mu,
                    "case": solution.case, "f": f_val})
    return rows


def scipy_cross_check(V0: float = 10.0, h_max: float = 5.0, x0=(1.2, 2.3)) -> dict:
    """Kreuzpruefung gegen scipy.optimize.minimize(method='SLSQP') mit einem informierten
    Startwert - erste externe Kreuzpruefung der Linie."""

    def f(x):
        return fn.surface_area(x)

    def g1(x):
        return fn.volume_constraint(x, V0)

    cons = [{"type": "eq", "fun": g1}, {"type": "ineq", "fun": lambda x: h_max - x[1]}]
    bounds = [(1e-6, None), (1e-6, None)]
    res = minimize(f, x0=list(x0), bounds=bounds, constraints=cons, method="SLSQP")
    own = analyse(Settings(V0=V0, h_max=h_max))
    own_x = np.array([own["solution"].r, own["solution"].h])
    return {"scipy_x": res.x, "scipy_f": float(res.fun), "scipy_success": bool(res.success),
            "own_x": own_x, "own_f": own["f_val"],
            "max_abs_diff": float(np.max(np.abs(res.x - own_x)))}


def scipy_local_optimum_pitfall(V0: float = 10.0, h_max: float = 5.0) -> dict:
    """Ehrlicher Nebenbefund: SLSQP von einem 'naiven' Startwert (1,1) konvergiert zu einem
    schlechteren Punkt (kein weiteres lokales Optimum: auf der Volumenkurve ist f strikt konvex)
    und meldet trotzdem success=True - vorzeitiger SLSQP-Abbruch mit Gradienten per Differenzen."""

    def f(x):
        return fn.surface_area(x)

    def g1(x):
        return fn.volume_constraint(x, V0)

    cons = [{"type": "eq", "fun": g1}, {"type": "ineq", "fun": lambda x: h_max - x[1]}]
    bounds = [(1e-6, None), (1e-6, None)]
    res_naive = minimize(f, x0=[1.0, 1.0], bounds=bounds, constraints=cons, method="SLSQP")
    res_informed = minimize(f, x0=[1.2, 2.3], bounds=bounds, constraints=cons, method="SLSQP")
    return {"naive_x": res_naive.x, "naive_f": float(res_naive.fun),
            "naive_success": bool(res_naive.success), "informed_x": res_informed.x,
            "informed_f": float(res_informed.fun),
            "gap_percent": float((res_naive.fun - res_informed.fun) / res_informed.fun * 100)}


def gradient_check(eps: float = 1e-6) -> dict:
    x = np.array([1.3, 0.7])
    V0 = 10.0
    results = {}
    for name, func, gradf in (
        ("f", fn.surface_area, fn.grad_surface_area),
        ("g1", lambda xx: fn.volume_constraint(xx, V0), fn.grad_volume_constraint),
    ):
        analytic = gradf(x)
        numeric = np.zeros(2)
        for i in range(2):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            numeric[i] = (func(xp) - func(xm)) / (2 * eps)
        results[f"{name}_max_rel_err"] = float(
            np.max(np.abs(analytic - numeric) / np.maximum(np.abs(analytic), 1e-8)))
    return results
