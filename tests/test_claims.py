"""Jede Zahl aus README.md und App wird hier aus den echten Auswertungsfunktionen neu berechnet.
Modul-Fixtures berechnen jeden Sweep nur einmal."""
import pytest

import kkt_evaluation as ev


@pytest.fixture(scope="module")
def slackness_rows():
    return ev.complementary_slackness_sweep()


@pytest.fixture(scope="module")
def robustness_rows():
    return ev.unconstrained_case_robustness()


def test_claim_hand_derived_solution_is_exact():
    out = ev.hand_derived_check()
    assert out["max_abs_err"] < 1e-10
    assert out["n_iter"] <= 10


def test_claim_informed_start_converges_across_all_tested_volumes(robustness_rows):
    assert all(r["matches_analytic"] for r in robustness_rows)
    assert all(r["converged"] for r in robustness_rows)


def test_claim_mu_is_zero_when_height_limit_is_generous(slackness_rows):
    for row in slackness_rows:
        if row["h_max"] >= 2.34:
            assert row["mu"] == 0.0


def test_claim_mu_is_positive_when_height_limit_binds(slackness_rows):
    for row in slackness_rows:
        if row["h_max"] <= 2.0:
            assert row["mu"] > 0.0


def test_claim_height_exactly_matches_limit_when_binding(slackness_rows):
    for row in slackness_rows:
        if row["case"] == "active":
            assert row["h"] == row["h_max"]


def test_claim_scipy_cross_check_matches_own_solution():
    out = ev.scipy_cross_check()
    assert out["max_abs_diff"] < 1e-5
    assert out["scipy_success"]


def test_claim_scipy_naive_start_never_beats_the_informed_one():
    """Ehrlicher Nebenbefund (auf Windows/der lokal installierten SciPy-Version gemessen, siehe
    README): SLSQP kann von Startwert (1,1) zu einem schlechteren, nicht-globalen Optimum
    konvergieren und meldet trotzdem success=True. Die GROESSE der Luecke haengt am
    SciPy-/LAPACK-Build (auf der Linux-CI reproduziert sich der exakte Wert nicht - andere
    interne Rundung im SLSQP-Fortran-Code fuehrt dort zufaellig zum globalen Optimum). Die
    Testsuite prueft deshalb nur die robuste, plattformunabhaengige Richtung: der informierte
    Startwert ist NIE schlechter als der naive."""
    out = ev.scipy_local_optimum_pitfall()
    assert out["naive_success"]
    assert out["gap_percent"] >= -1e-6


def test_claim_gradient_check_below_1e_minus_6():
    out = ev.gradient_check()
    assert out["f_max_rel_err"] < 1e-6
    assert out["g1_max_rel_err"] < 1e-6
