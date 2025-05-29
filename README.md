# Monte-Carlo-Ising-model
# === Erklärungsteil ===
with st.expander("ℹ️ Theorie: Ising-Modell & Metropolis-Algorithmus", expanded=False):
    st.markdown(r'''
### 🧊 Ising-Modell

Das 2D-Ising-Modell beschreibt Spins $\sigma_i = \pm 1$ auf einem Gitter. Die Energie ist gegeben durch:

$$
E = -J \sum_{\langle i, j \\rangle} \sigma_i \sigma_j
$$

Dabei koppeln benachbarte Spins (hier: 4 Nachbarn pro Spin). Wir setzen $J = 1$.

---

### 🎲 Metropolis-Algorithmus

Für jeden Monte-Carlo-Schritt:

1. Wähle zufällig einen Spin $\sigma_{i,j}$
2. Berechne die Energieänderung $\Delta E$
3. Wenn $\Delta E < 0$, akzeptiere den Flip
4. Sonst akzeptiere mit Wahrscheinlichkeit $e^{-\Delta E / T}$

---

### 🐍 Beispiel-Code für einen MC-Schritt

```python
if dE < 0 or np.random.rand() < np.exp(-dE / T):
    lattice[i, j] *= -1
