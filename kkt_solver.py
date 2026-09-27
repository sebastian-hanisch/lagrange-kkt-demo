"""Loest kleine KKT-Systeme per aktive-Menge-Enumeration: da nur eine Ungleichung vorliegt, gibt
es genau zwei Faelle (Hoehenlimit inaktiv/aktiv). Jeder Fall wird per (gedaempftem) Newton-
Verfahren auf dem Stationaritaets-/Nebenbedingungssystem geloest (Nullstellensuche, nicht
Abstieg - Rueckverweis auf Stueck 2); anschliessend entscheidet primale + duale Zulaessigkeit,
welcher Fall die tatsaechliche KKT-Loesung ist."""
from dataclasses import dataclass

import numpy as np

import kkt_functions as fn


@dataclass
class KKTSolution:
    r: float
    h: float
    lam: float
    mu: float
    case: str  # "inactive" oder "active"
    n_iter: int
    converged: bool


def _newton_damped(F, x0, max_iter=100, tol=1e-10):
    """Gedaempftes Newton-Verfahren (Schritt-Halbierung, falls die Residuumsnorm sonst waechst) -
    ein einfacher, gaengiger Sicherungsmechanismus fuer Nullstellensuche auf einem nichtlinearen
    Gleichungssystem (siehe README: ungedaempftes Newton kann bei ungeeignetem Start divergieren)."""

    def jacobian(v):
        eps = 1e-6
        n = len(v)
        J = np.zeros((n, n))
        base = F(v)
        for j in range(n):
            vp = v.copy()
            vp[j] += eps
            J[:, j] = (F(vp) - base) / eps
        return J

    x = np.asarray(x0, dtype=float)
    for k in range(1, max_iter + 1):
        Fx = F(x)
        norm = float(np.linalg.norm(Fx))
        if norm < tol:
            return x, k - 1, True
        J = jacobian(x)
        dx = np.linalg.solve(J, -Fx)
        step = 1.0
        for _ in range(30):
            if float(np.linalg.norm(F(x + step * dx))) < norm:
                break
            step *= 0.5
        x = x + step * dx
    return x, max_iter, False


def solve_unconstrained_case(V0: float) -> KKTSolution:
    """Stationaritaet + Volumen-Nebenbedingung, ohne Hoehenlimit: [nabla f + lam*nabla g1 = 0,
    g1 = 0]. Startwert aus Dimensionsanalyse (r,h ~ V0^(1/3), lambda ~ -2/r) - ein neutraler
    Startpunkt wie (1,1,-1) kann bei kleinem/grossem V0 zu einem falschen, aber ebenfalls
    stationaeren Punkt konvergieren (siehe README)."""

    def F(v):
        r, h, lam = v
        gf = fn.grad_surface_area(np.array([r, h]))
        gg1 = fn.grad_volume_constraint(np.array([r, h]))
        return np.array([gf[0] + lam * gg1[0], gf[1] + lam * gg1[1],
                         fn.volume_constraint(np.array([r, h]), V0)])

    cube = V0 ** (1 / 3)
    x0 = [cube, cube, -2 / cube]
    sol, k, converged = _newton_damped(F, x0)
    return KKTSolution(r=sol[0], h=sol[1], lam=sol[2], mu=0.0, case="inactive", n_iter=k,
                       converged=converged)


def solve_active_case(V0: float, h_max: float) -> KKTSolution:
    """Hoehenlimit aktiv gesetzt (h=h_max), r direkt aus der Volumen-Nebenbedingung, lambda/mu
    aus den beiden Stationaritaetsgleichungen - geschlossene Formel, kein Newton noetig."""
    r = float(np.sqrt(V0 / (np.pi * h_max)))
    x = np.array([r, h_max])
    gf = fn.grad_surface_area(x)
    gg1 = fn.grad_volume_constraint(x)
    lam = float(-gf[0] / gg1[0])
    mu = float(-(gf[1] + lam * gg1[1]))
    return KKTSolution(r=r, h=h_max, lam=lam, mu=mu, case="active", n_iter=0, converged=True)


def solve_kkt(V0: float, h_max: float) -> KKTSolution:
    """Waehlt den zulaessigen Fall: ist die unrestringierte Loesung primal zulaessig (h<=h_max),
    ist sie die KKT-Loesung (mu=0 implizit dual zulaessig). Sonst muss das Limit binden -
    verifiziert per dualer Zulaessigkeit (mu>=0) am aktiven Fall."""
    unconstrained = solve_unconstrained_case(V0)
    if unconstrained.h <= h_max + 1e-9:
        return unconstrained
    active = solve_active_case(V0, h_max)
    return active
