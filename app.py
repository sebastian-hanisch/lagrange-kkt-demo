"""Lagrange/KKT — Nebenbedingungen zum ersten Mal

Sebastian Hanisch - Operations Research und Machine Learning

Stück 4 der "Nichtlineare Optimierung"-Reihe der "Konzepte"-Reihe:
Gradientenabstieg -> Newton-Verfahren -> Quasi-Newton -> Lagrange/KKT ->
{Straf-/Barriere-Verfahren, SQP -> Innere-Punkte-Verfahren} + Stochastische Gradientenverfahren.
Stück 1-3 kannten keine Nebenbedingungen. Die Karush-Kuhn-Tucker-Bedingungen (Karush 1939, Kuhn &
Tucker 1951) sind kein Abstiegsverfahren, sondern ein Gleichungssystem, das ein Optimum MIT
Gleichungs- und Ungleichungsnebenbedingungen erfüllen muss.

Lauffähig mit: streamlit run app.py
"""
import numpy as np
import streamlit as st

import kkt_constants as C
import kkt_evaluation as ev
import kkt_presets as pr
import kkt_visualization as viz

st.set_page_config(page_title="Lagrange/KKT", layout="wide")


@st.cache_data(show_spinner=False)
def _run(V0, h_max):
    out = ev.analyse(ev.Settings(V0=V0, h_max=h_max))
    s = out["solution"]
    return {"r": s.r, "h": s.h, "lam": s.lam, "mu": s.mu, "case": s.case, "n_iter": s.n_iter,
            "converged": s.converged, "f_val": out["f_val"]}


@st.cache_data(show_spinner=False)
def _hand_derived_check():
    return ev.hand_derived_check()


@st.cache_data(show_spinner=False)
def _robustness():
    return ev.unconstrained_case_robustness()


@st.cache_data(show_spinner=False)
def _slackness_sweep(V0):
    return ev.complementary_slackness_sweep(V0=V0)


@st.cache_data(show_spinner=False)
def _scipy_cross_check(V0, h_max):
    return ev.scipy_cross_check(V0=V0, h_max=h_max)


@st.cache_data(show_spinner=False)
def _scipy_pitfall(V0, h_max):
    return ev.scipy_local_optimum_pitfall(V0=V0, h_max=h_max)


@st.cache_data(show_spinner=False)
def _gradient_check():
    return ev.gradient_check()


st.title("📦 Lagrange/KKT — Nebenbedingungen zum ersten Mal")
st.markdown(
    "Bisher gab es keine Nebenbedingungen. Jetzt: minimiere die Materialkosten (Oberfläche) "
    "eines zylindrischen Transportbehälters bei **festem Volumen** — und optional einem "
    "**Höhenlimit**. Die **Karush-Kuhn-Tucker-Bedingungen** (KKT) sagen genau, wann ein Punkt "
    "optimal ist: kein Abstiegsverfahren, sondern ein Gleichungssystem, das die Lösung "
    "charakterisiert."
)
st.caption(
    "Stück 4 der 'Nichtlineare Optimierung'-Reihe. Folgestücke (alle gebaut): "
    "Straf-/Barriere-Verfahren, SQP, Innere-Punkte-Verfahren, Stochastische Gradientenverfahren."
)

with st.expander("So funktionieren Lagrange/KKT", expanded=True):
    st.markdown(
        "1. **Nur Gleichungsnebenbedingung** (Lagrange 1788): am Optimum zeigen $\\nabla f$ und "
        "$\\nabla g_1$ in dieselbe Richtung — $\\nabla f+\\lambda\\nabla g_1=0$.\n"
        "2. **Zusätzlich eine Ungleichung** $g_2\\le0$ (KKT): entweder ist sie inaktiv "
        "($g_2<0$, ihr Multiplikator $\\mu=0$) oder sie bindet ($g_2=0$, $\\mu\\ge0$) — nie "
        "beides zugleich (**komplementärer Schlupf**, $\\mu\\cdot g_2=0$).\n"
        "3. Bei nur einer Ungleichung gibt es genau zwei Fälle zu prüfen: inaktiv oder aktiv. "
        "Diese App löst beide und wählt den zulässigen."
    )

st.caption("🎯 Schnellstart – ein Klick lädt ein durchgerechnetes Beispiel:")
preset_cols = st.columns(len(C.PRESETS))
for col, (key, preset) in zip(preset_cols, C.PRESETS.items()):
    with col:
        st.button(preset["label"], help=preset["help"], on_click=pr.apply_preset, args=(key,),
                   use_container_width=True)

st.caption("🔗 Die Adresszeile speichert deine Einstellungen als Permalink.")

pr.load_permalink_settings()
pr.init_session_state_defaults()
ss = st.session_state

with st.sidebar:
    st.header("⚙️ Einstellungen")
    V0 = st.slider("Volumen V₀", C.V0_MIN, C.V0_MAX, ss["V0"], step=1.0, key="widget_V0",
                   on_change=pr.store_from_widget, args=("V0",))
    ss["V0"] = V0
    h_max = st.slider("Höhenlimit h_max", C.H_MAX_MIN, C.H_MAX_MAX, ss["h_max"], step=0.05,
                      key="widget_h_max", on_change=pr.store_from_widget, args=("h_max",))
    ss["h_max"] = h_max

pr.sync_query_params(dict(V0=V0, h_max=h_max))

out = _run(V0, h_max)

st.markdown("---")
st.subheader("🎯 Die Lösung")
r_star_now = (V0 / (2 * np.pi)) ** (1 / 3)
pad_r = max(2.5, r_star_now * 2.2)
pad_h = max(6.0, 2 * r_star_now * 2.2, h_max * 1.3)
col_left, col_right = st.columns([3, 2])
with col_left:
    fig = viz.build_problem_figure(V0, h_max, out["r"], out["h"], r_range=(0.05, pad_r),
                                   h_range=(0.05, pad_h),
                                   title=f"Volumen={V0:.0f}, Höhenlimit={h_max:.2f}")
    st.plotly_chart(fig, key=f"problem_{V0}_{h_max}", use_container_width=True)
with col_right:
    st.metric("Fall", "Höhenlimit inaktiv" if out["case"] == "inactive" else "Höhenlimit bindet")
    m1, m2 = st.columns(2)
    m1.metric("Radius r*", f"{out['r']:.4f}")
    m2.metric("Höhe h*", f"{out['h']:.4f}")
    m3, m4 = st.columns(2)
    m3.metric("λ (Volumen)", f"{out['lam']:.4f}")
    m4.metric("μ (Höhenlimit)", f"{out['mu']:.4f}")
    st.metric("Materialkosten f", f"{out['f_val']:.4f}")

st.markdown("---")
st.subheader("🎯 Die zentrale Messung: komplementärer Schlupf")
r_star = (V0 / (2 * np.pi)) ** (1 / 3)
h_star = 2 * r_star
sweep = _slackness_sweep(V0)
col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(viz.build_complementary_slackness_figure(sweep, h_star),
                    key=f"slackness_{V0}", use_container_width=True)
with col_b:
    st.plotly_chart(viz.build_f_vs_h_max_figure(sweep), key=f"fvs_{V0}",
                    use_container_width=True)
st.caption(
    f"Bei h_max={h_star:.3f} (der unrestringierten Lösung h*=2r*) springt der Multiplikator μ "
    "von 0 auf einen positiven Wert — genau dort, wo das Höhenlimit beginnt zu binden. Rechts "
    "steigen die Materialkosten erst ab diesem Punkt an."
)

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    "| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |\n"
    "|---|---|---|\n"
    "| Newton auf dem KKT-System startet nahe genug an der Lösung | Ein neutraler Startwert "
    "kann zu einem falschen, aber ebenfalls stationären Punkt konvergieren (siehe 📐) | "
    "Informierte Startwerte (hier bereits verwendet) |\n"
    "| Nur eine Ungleichung | Mehr Ungleichungen brauchen mehr als zwei Fälle in der "
    "Enumeration | SQP/Innere-Punkte (Stück 6 und 7) |\n"
    "| SciPy-Erfolgsmeldung bedeutet globales Optimum | Gilt hier NICHT: SLSQP kann bei einem "
    "ungünstigen Startwert bei einem schlechteren, lokalen Punkt landen (siehe 📐) | Eigene "
    "Korrektheits-Kette statt blindem Vertrauen in `success=True` |\n"
)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**KKT-Bedingungen:** $\nabla f+\lambda\nabla g_1+\mu\nabla g_2=0$, $g_1=0$, $g_2\le0$, $\mu\ge0$,
$\mu g_2=0$.

**Von Hand hergeleitete Lösung** (Höhenlimit inaktiv): $r^*=(V_0/(2\pi))^{1/3}$, $h^*=2r^*$
("Höhe = Durchmesser", der klassische Zylinder-Befund).
"""
    )
    hd = _hand_derived_check()
    h1, h2 = st.columns(2)
    h1.metric("r*: Formel vs. Newton", f"{hd['r_star']:.6f} / {hd['r_numeric']:.6f}")
    h2.metric("h*: Formel vs. Newton", f"{hd['h_star']:.6f} / {hd['h_numeric']:.6f}")
    st.caption(f"Newton-Iterationen: {hd['n_iter']}, max. Abweichung: {hd['max_abs_err']:.2e}")

    st.markdown("**Robustheit über verschiedene Volumina V₀** (informierter Startwert):")
    rob = _robustness()
    rob_ok = sum(1 for r in rob if r["matches_analytic"])
    st.metric("Trifft die analytische Lösung", f"{rob_ok}/{len(rob)} getestete V₀-Werte")

    st.markdown("**SciPy-Kreuzprüfung** (informierter Startwert):")
    cc = _scipy_cross_check(V0, h_max)
    c1, c2 = st.columns(2)
    c1.metric("SciPy (SLSQP)", f"r={cc['scipy_x'][0]:.4f}, h={cc['scipy_x'][1]:.4f}")
    c2.metric("Eigene Lösung", f"r={cc['own_x'][0]:.4f}, h={cc['own_x'][1]:.4f}")
    st.caption(f"Max. Abweichung: {cc['max_abs_diff']:.2e}")

    st.markdown(
        "**Ehrlicher Nebenbefund** (im README auf einem konkreten SciPy-/LAPACK-Build gemessen, "
        "~1,86 % Lücke): SciPy's SLSQP KANN `success=True` melden und trotzdem bei einem "
        "schlechteren, nicht-globalen Optimum landen, wenn der Startwert ungünstig ist — das "
        "Problem ist echt nichtkonvex (die Hesse-Matrix von f ist indefinit). Wie groß die Lücke "
        "ausfällt, hängt vom internen Rundungsverhalten des SLSQP-Codes ab und ist daher nicht "
        "auf jedem Rechner/jeder SciPy-Version identisch:"
    )
    pit = _scipy_pitfall(V0, h_max)
    p1, p2 = st.columns(2)
    p1.metric("Naiver Start (1,1): f", f"{pit['naive_f']:.4f}")
    p2.metric("Informierter Start: f", f"{pit['informed_f']:.4f}")
    st.caption(f"Lücke auf diesem Rechner: {pit['gap_percent']:.2f} % — trotz `success=True` bei "
              "beiden. Ist die Lücke hier 0 %, hat SciPy auf dieser Plattform zufällig auch vom "
              "naiven Start das globale Optimum gefunden (siehe Erklärung oben).")

    grad_err = _gradient_check()
    g1, g2 = st.columns(2)
    g1.metric("Gradienten-Check f", f"{grad_err['f_max_rel_err']:.1e}")
    g2.metric("Gradienten-Check g₁", f"{grad_err['g1_max_rel_err']:.1e}")

    st.markdown(
        "**Literatur:** Karush, W. (1939). Master-Arbeit, University of Chicago "
        "(unveröffentlicht). Kuhn, H. W. & Tucker, A. W. (1951). *Nonlinear programming.* "
        "Proc. 2nd Berkeley Symposium."
    )
    st.caption(
        "Implementiert in `kkt_functions.py` (Zielfunktion, Nebenbedingungen), "
        "`kkt_solver.py` (Newton auf dem KKT-System, aktive-Menge-Enumeration), "
        "`kkt_evaluation.py` (Korrektheits-Kette, SciPy-Kreuzprüfung), "
        "`kkt_visualization.py` (Plots)."
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Nichtlineare Optimierung: acht Stücke, zwei Äste](https://sebastianhanisch.net/konzepte-nichtlineare-optimierung.html)."
)
