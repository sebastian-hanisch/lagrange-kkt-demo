# Lagrange/KKT – Nebenbedingungen zum ersten Mal – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-lagrange-kkt-demo.streamlit.app/)**

Stück 4 der **Nichtlineare-Optimierung-Reihe** der "Konzepte"-Reihe im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Stück 1–3 kannten keine Nebenbedingungen. Die **Karush-Kuhn-Tucker-Bedingungen** (Karush 1939,
Kuhn & Tucker 1951) sind kein Abstiegsverfahren wie zuvor, sondern ein **Gleichungssystem**, das
ein Optimum mit Gleichungs- UND Ungleichungsnebenbedingungen erfüllen muss.

**Einordnung in die Reihe:**

```
Gradientenabstieg (WURZEL)                       [gebaut]
 └─ Newton-Verfahren                             [gebaut]
      └─ Quasi-Newton (BFGS/L-BFGS)               [gebaut]
           └─ Lagrange-Multiplikatoren/KKT        [DIESES STÜCK]
                ├─ Straf-/Barriere-Verfahren      [gebaut]
                └─ SQP                            [gebaut]
                     └─ Innere-Punkte-Verfahren   [gebaut]
 └─ Stochastische Gradientenverfahren             [gebaut, letztes Stück]
```

**Vehikel B (neu ab diesem Stück):** die Materialkosten (Oberfläche) eines zylindrischen
Transportbehälters minimieren, bei **festem Volumen** und einem **Höhenlimit** (z. B.
Stapel-/Transporthöhe) — der klassische "Höhe = Durchmesser"-Lehrbuchfall, jetzt mit einer echten
Ungleichung ergänzt.

**Ergebnis in Kürze:** Ist das Höhenlimit großzügig, reproduziert die numerische Lösung exakt die
von Hand hergeleitete Formel $r^*=(V_0/(2\pi))^{1/3}$, $h^*=2r^*$. Der KKT-Multiplikator $\mu$ für
das Höhenlimit springt **exakt** an der Stelle $h_{\max}=2r^*$ von 0 auf einen positiven Wert —
komplementärer Schlupf, gemessen statt nur zitiert. Die eigene Lösung stimmt mit
`scipy.optimize.minimize(method="SLSQP")` bis auf $10^{-7}$ überein. **Ehrlicher Nebenbefund:**
SciPy meldet `success=True` auch bei einem 1,86 % schlechteren Punkt, der gar kein Optimum ist
(vorzeitiger Abbruch von SLSQP mit Gradienten per Differenzen, Startwert (1, 1)) — auf der
Volumenkurve ist die Zielfunktion strikt konvex, das Optimum ist eindeutig.

## Warum dieses Problem

Stück 3 endete mit rein unrestringierter Minimierung. Reale Optimierungsprobleme haben fast immer
Nebenbedingungen — die KKT-Bedingungen sind der erste Rahmen, der Gleichungen UND Ungleichungen
gemeinsam erfasst und damit alle folgenden Stücke (Straf-/Barriere-Verfahren, SQP,
Innere-Punkte-Verfahren) erst motiviert.

## Vorab-Hypothesen (vor der Messung notiert, hier geprüft)

| Hypothese | Ergebnis |
|---|---|
| Bei großzügigem Höhenlimit reproduziert die numerische Lösung exakt $r^*=(V_0/(2\pi))^{1/3}$, $h^*=2r^*$ | ✅ max. Abweichung $4{,}4\cdot10^{-16}$ |
| Der Multiplikator μ springt exakt bei $h_{\max}=2r^*$ von 0 auf positiv | ✅ bei $h_{\max}=2{,}335$ (=$h^*$) springt μ von 0 auf $0{,}0002$, bei $h_{\max}=1{,}0$ bereits auf $14{,}4$ |
| Eigene Lösung stimmt mit SciPy (SLSQP) überein | ✅ max. Abweichung $1{,}2\cdot10^{-7}$ (informierter Startwert) |
| Gradienten-Check gegen finite Differenzen unter $10^{-6}$ | ✅ beide um $10^{-10}$ |
| ⚠️ **Nicht vorab vermutet, aber gefunden:** ein naiver Startwert für Newton auf dem KKT-System führt nicht zum Ziel | ⚠️ bestätigt — das gedämpfte Newton-Verfahren stagniert (kein falscher stationärer Punkt: für $r>0$ hat das KKT-System nur die Lösung $h=2r$); ein informierter, aus Dimensionsanalyse hergeleiteter Startwert löst das robust über alle getesteten $V_0$ |
| ⚠️ **Ebenfalls ungeplant gefunden:** SciPy's `success=True` bedeutet nicht automatisch ein Optimum | ⚠️ bei Startwert (1,1) bricht SLSQP 1,86 % über dem tatsächlichen Optimum ab (KKT-Residuum 1,08, also kein stationärer Punkt), meldet aber `success=True` |

## Befunde (gemessen, keine Behauptungen)

**Hand-Ableitung vs. Newton auf dem KKT-System** ($V_0=10$):

| | Formel | Numerisch |
|---|---|---|
| $r^*$ | 1,167544 | 1,167544 |
| $h^*$ | 2,335089 | 2,335089 |

Newton-Iterationen bis Konvergenz: 5. Max. Abweichung: $4{,}4\cdot10^{-16}$.

**Robustheit über verschiedene Volumina** (informierter Startwert
$x_0=(V_0^{1/3},\,V_0^{1/3},\,-2/V_0^{1/3})$): 6 von 6 getesteten $V_0\in\{0{,}5,1,5,10,50,100\}$
treffen die analytische Lösung exakt. Ein neutraler Startwert wie $(1,1,-1)$ konvergiert dagegen
nur bei $V_0=5$ (5 von 6 Werten scheitern): die gedämpfte Newton-Iteration stagniert nach 100
Iterationen (z. B. bei $V_0=1$ bei $r\approx0{,}079$, $h\approx46$ mit Residuum 1,7). Das ist **kein**
weiterer stationärer Punkt: aus den KKT-Gleichungen folgt $\lambda=-2/r$ und $h=2r$, es gibt für
$r>0$ genau eine Lösung (per `scipy.optimize.fsolve` aus 400 Zufallsstarts bestätigt).

**Komplementärer Schlupf** ($V_0=10$, $h^*=2{,}335$):

| $h_{\max}$ | Fall | $\mu$ | Materialkosten $f$ |
|---|---|---|---|
| 5,0 | inaktiv | 0,0000 | 25,6950 |
| 3,0 | inaktiv | 0,0000 | 25,6950 |
| 2,4 | inaktiv | 0,0000 | 25,6950 |
| 2,335 | aktiv (am Übergang) | 0,0002 | 25,6950 |
| 2,0 | aktiv | 1,0367 | 25,8533 |
| 1,5 | aktiv | 4,3124 | 27,0627 |
| 1,0 | aktiv | 14,3950 | 31,2100 |

**SciPy-Kreuzprüfung** (informierter Startwert $x_0=(1{,}2,\,2{,}3)$): max. Abweichung
$1{,}2\cdot10^{-7}$ bei $h_{\max}=5$ (inaktiv), $1{,}4\cdot10^{-11}$ bei $h_{\max}=1{,}5$ (aktiv; wie die Lücke unten SLSQP-/Plattform-abhängig).

**Ehrlicher Nebenbefund — SciPy-Fallstrick:** bei $h_{\max}=5$ (Höhenlimit inaktiv) konvergiert
SLSQP auf diesem Rechner (Windows, SciPy 1.18) von Startwert $(1,1)$ zu $f=26{,}173$
(`success=True`), während der Startwert $(1{,}2,\,2{,}3)$ das echte Optimum $f=25{,}695$ findet —
eine Lücke von **1,86 %**, unsichtbar allein am `success`-Flag. Der Endpunkt $(1{,}016;\,3{,}086)$ ist
**kein** weiteres lokales Optimum (auf der Volumenkurve ist $f(r)=2\pi r^2+2V_0/r$ strikt konvex, das
Optimum also eindeutig): $\nabla f$ und $\nabla g_1$ sind dort nicht parallel (KKT-Residuum 1,08 bei
$\|\nabla f\|\approx33$), SLSQP hat mit Gradienten per Differenzen vorzeitig abgebrochen; mit
analytischem Zielfunktionsgradienten findet auch der Startwert $(1,1)$ das Optimum. **Plattform-Hinweis:** die Größe
dieser Lücke hängt am internen Rundungsverhalten des SLSQP-Fortran-Codes (LAPACK/BLAS-Build) —
auf der Linux-CI dieses Repos reproduziert sich die exakte Zahl nicht (dort findet SLSQP von
$(1,1)$ zufällig ebenfalls das globale Optimum). Die *Existenz* des Fallstricks (SciPy kann bei
ungünstigem Start `success=True` UND einen Punkt liefern, der kein Optimum ist) ist der eigentliche,
robuste Befund; die konkrete Prozentzahl ist eine Momentaufnahme dieses Rechners, kein
Naturgesetz — dieselbe Lehre wie bei chaotischen Trainings-Trajektorien in der Neuronale-Netze-
Linie, hier aber an einem deterministischen Solver statt an stochastischem Training.

## Modell und Verfahren

- `kkt_functions.py` – Zielfunktion (Oberfläche), Volumen-Gleichungsnebenbedingung,
  Höhenlimit-Ungleichungsnebenbedingung, je mit Gradient von Hand.
- `kkt_solver.py` – aktive-Menge-Enumeration: `solve_unconstrained_case()` (Newton mit
  Schritt-Halbierung auf dem KKT-System), `solve_active_case()` (geschlossene Formel bei
  fixiertem Höhenlimit), `solve_kkt()` (wählt den zulässigen Fall).
- `kkt_evaluation.py` – Hand-Ableitungs-Check, Robustheits-Sweep, komplementärer-Schlupf-Sweep,
  SciPy-Kreuzprüfung (inkl. Fallstrick-Demonstration), Gradienten-Check.
- `kkt_visualization.py` – Plotly: Kontur mit Volumen-Kurve und Höhenlimit-Linie,
  μ-vs-Höhenlimit-Sweep, Materialkosten-vs-Höhenlimit.

## Was die App zeigt

Volumen und Höhenlimit in der Sidebar; Kontur mit Volumen-Kurve, Höhenlimit-Linie und Lösung für
die aktuelle Konfiguration; Fall (aktiv/inaktiv), $r^*$, $h^*$, $\lambda$, $\mu$ und
Materialkosten als Metriken; der komplementäre-Schlupf-Sweep als zentraler Befund; ein
"📐"-Abschnitt mit der vollständigen Korrektheits-Kette (Hand-Ableitung, Robustheit, SciPy-
Kreuzprüfung inkl. Fallstrick, Gradienten-Check).

## Was nicht funktioniert hat / Grenzen

**Zwei echte, ungeplante Funde (keine Hypothesen widerlegt, aber wichtige Ergänzungen):**
(1) Newton auf dem KKT-System braucht einen informierten Startwert — ein neutraler wie $(1,1,-1)$
stagniert bei fünf von sechs getesteten $V_0$ (kein falscher stationärer Punkt, sondern
Nicht-Konvergenz des gedämpften Newton-Verfahrens).
Behoben durch einen aus Dimensionsanalyse hergeleiteten Startwert, robust über 6 getestete
Größenordnungen. (2) SciPy's `success=True` ist kein Beleg für ein Optimum — ein
scheinbar harmloser Startwert wie $(1,1)$ liefert (mit Gradienten per Differenzen) einen 1,86 %
schlechteren Punkt, ebenfalls mit `success=True`; auf der Volumenkurve ist das Problem konvex (die
Hesse-Matrix von $f$ allein ist zwar indefinit), der Punkt ist also kein zweites lokales Optimum.

**Grenzen:** nur eine Gleichungs- und eine Ungleichungsnebenbedingung (mehr Ungleichungen würden
mehr als zwei Fälle in der Enumeration brauchen — SQP/Innere-Punkte-Verfahren, Stück 6 und 7,
lösen das systematischer).

## Tests

31 Tests, `python -m pytest tests/ -v` (Laufzeit lokal ~2,5 Sekunden):
- `test_functions.py` – Zielfunktion/Nebenbedingungen, Gradienten gegen finite Differenzen.
- `test_solver.py` – Hand-Ableitung, komplementärer Schlupf, Fallauswahl.
- `test_evaluation.py` – Sweep-Funktionen mit billigen Parametern.
- `test_claims.py` – jede Zahl oben nachgerechnet, mit Modul-Fixtures für die Sweeps.
- `test_presets.py`, `test_app.py` – Presets, Regler-Extremwerte, Footer.
- `test_oracle_lagrange_kkt.py` – unabhängige Orakel: Reduktion auf eine Variable (`minimize_scalar`),
  Multiplikatoren als Sensitivität des Optimalwerts, `fsolve` auf dem KKT-System, SLSQP-Endpunkt.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `kkt_constants.py` | Regler-Grenzen, Presets |
| `kkt_functions.py` | Zielfunktion, Nebenbedingungen |
| `kkt_solver.py` | Newton auf dem KKT-System, aktive-Menge-Enumeration |
| `kkt_evaluation.py` | Korrektheits-Kette, SciPy-Kreuzprüfung, Sweeps |
| `kkt_visualization.py` | Plotly-Plots |
| `kkt_presets.py` | Permalink-Sync, Presets |
| `tests/` | pytest-Suite |

## Bewusst nicht umgesetzt

Keine Mehrfach-Ungleichungen (bewusst bei genau einer gehalten, damit die
Enumerations-plus-Verifikations-Methode von Hand nachvollziehbar bleibt). Kein allgemeiner
KKT-Löser für beliebige Probleme — SQP und Innere-Punkte-Verfahren (Stück 6 und 7) sind die
systematische Antwort darauf.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements-dev.txt
streamlit run app.py
```

## Literatur

- Karush, W. (1939). *Minima of Functions of Several Variables with Inequalities as Side
  Constraints.* Master-Arbeit, University of Chicago (unveröffentlicht).
- Kuhn, H. W. & Tucker, A. W. (1951). *Nonlinear programming.* Proceedings of the 2nd Berkeley
  Symposium on Mathematical Statistics and Probability.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Nichtlineare Optimierung: acht Stücke, zwei Äste](https://sebastianhanisch.net/konzepte-nichtlineare-optimierung.html).
