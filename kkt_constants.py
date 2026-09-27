"""Regler-Grenzen, feste Annahmen, gemessene Werte und Presets."""

V0_MIN, V0_MAX, V0_DEFAULT = 1.0, 100.0, 10.0
H_MAX_MIN, H_MAX_MAX, H_MAX_DEFAULT = 0.5, 6.0, 5.0

PRESETS = {
    "hoehenlimit_nicht_bindend": dict(
        label="Höhenlimit nicht bindend",
        V0=10.0, h_max=5.0,
        help="Das Höhenlimit ist großzügig — die Lösung ist die klassische unrestringierte "
             "Formel h*=2r*.",
    ),
    "hoehenlimit_bindet": dict(
        label="Höhenlimit bindet",
        V0=10.0, h_max=1.5,
        help="Das Höhenlimit ist zu knapp für die unrestringierte Lösung — der Behälter wird "
             "breiter statt höher, mit einem echten Lagrange-Multiplikator μ>0.",
    ),
}
