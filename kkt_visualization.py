"""Reine Plotly-Figure-Builder, keine Streamlit-Aufrufe. Achsen fest uebergeben (siehe
feedback_plotly_fixedrange_convention/feedback_plotly_scaleanchor_explicit_range). Balken-/
Linien-x-Achsen nutzen echte Zahlenwerte (feedback_plotly_bar_numeric_labels_vline_mismatch).
Keine literalen "|" in Markdown-Tabellenzellen (feedback_markdown_table_literal_pipe_breaks_columns)."""
import numpy as np
import plotly.graph_objects as go

import kkt_functions as fn

COLOR_CONSTRAINT = "#d62728"
COLOR_LIMIT = "#9467bd"
COLOR_OPT = "#2ca02c"
COLOR_MU = "#1f77b4"


def build_problem_figure(V0, h_max, solution_r, solution_h, r_range=(0.1, 2.5),
                         h_range=(0.1, 6.0), title=""):
    rs = np.linspace(r_range[0], r_range[1], 150)
    hs = np.linspace(h_range[0], h_range[1], 150)
    Z = np.zeros((len(hs), len(rs)))
    for i, hv in enumerate(hs):
        for j, rv in enumerate(rs):
            Z[i, j] = fn.surface_area(np.array([rv, hv]))
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=rs, y=hs, z=Z, showscale=False, colorscale="Blues", contours=dict(coloring="fill"),
        opacity=0.6,
    ))
    # Volumen-Kurve: h = V0/(pi r^2)
    h_curve = V0 / (np.pi * rs ** 2)
    mask = (h_curve >= h_range[0]) & (h_curve <= h_range[1])
    fig.add_trace(go.Scatter(x=rs[mask], y=h_curve[mask], mode="lines", name="Volumen V=V₀",
                             line=dict(color=COLOR_CONSTRAINT, width=2)))
    # Hoehenlimit
    fig.add_trace(go.Scatter(x=list(r_range), y=[h_max, h_max], mode="lines",
                             name=f"Höhenlimit h≤{h_max:.2f}",
                             line=dict(color=COLOR_LIMIT, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[solution_r], y=[solution_h], mode="markers", name="Lösung",
                             marker=dict(color=COLOR_OPT, size=14, symbol="star")))
    fig.update_layout(
        title=title, xaxis=dict(range=list(r_range), fixedrange=True, title="Radius r"),
        yaxis=dict(range=list(h_range), fixedrange=True, title="Höhe h"),
        showlegend=True, height=420, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_complementary_slackness_figure(rows, h_star,
                                         title="Komplementärer Schlupf: μ vs. Höhenlimit"):
    h_max_vals = [r["h_max"] for r in rows]
    mu_vals = [r["mu"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h_max_vals, y=mu_vals, mode="lines+markers",
                             name="Multiplikator μ", line=dict(color=COLOR_MU, width=2)))
    fig.add_vline(x=h_star, line_dash="dash", line_color="#5B6B80",
                  annotation_text="h* (unrestringiert)", annotation_position="top")
    fig.update_layout(
        title=title, xaxis=dict(title="Höhenlimit h_max", fixedrange=True),
        yaxis=dict(title="μ (KKT-Multiplikator)", fixedrange=True),
        showlegend=False, height=340, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_f_vs_h_max_figure(rows, title="Zielfunktionswert vs. Höhenlimit"):
    h_max_vals = [r["h_max"] for r in rows]
    f_vals = [r["f"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h_max_vals, y=f_vals, mode="lines+markers", name="f am Optimum",
                             line=dict(color=COLOR_OPT, width=2)))
    fig.update_layout(
        title=title, xaxis=dict(title="Höhenlimit h_max", fixedrange=True),
        yaxis=dict(title="Materialkosten f(r,h)", fixedrange=True),
        showlegend=False, height=320, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
