"""Korrektheits-/Verhaltenstests mit billigen Parametern. Die offiziellen, teuren Sweep-Werte
stehen mit Toleranzband in test_claims.py (Modul-Fixtures, je Sweep nur einmal berechnet)."""
import kkt_evaluation as ev


def test_hand_derived_check_matches_within_tight_tolerance():
    out = ev.hand_derived_check()
    assert out["max_abs_err"] < 1e-8


def test_unconstrained_case_robustness_runs_with_small_set():
    rows = ev.unconstrained_case_robustness(V0_values=(1.0, 10.0))
    for row in rows:
        assert row["matches_analytic"]


def test_analyse_returns_valid_result():
    out = ev.analyse(ev.Settings(V0=10.0, h_max=5.0))
    assert out["f_val"] > 0
