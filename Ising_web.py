import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import imageio
from io import BytesIO
import os

# Importiere die Funktionen aus der separaten Datei
from ising_functions import (initialize_lattice, compute_energy, compute_magnetization,
                             metropolis_step, simulate, compute_correlation, simulate_thermo)

st.set_page_config(layout="wide")
st.title("🧲 Interaktive Ising-Modell Simulation")

# Theorie-Teil
with st.expander("ℹ️ Theorie zum Ising-Modell", expanded=True):
    with open("README.md", "r", encoding="utf-8") as f:
        st.markdown(f.read())

# Parameter-Bereich
st.subheader("⚙️ Parameter der Simulation")
col1, col2, col3, col4 = st.columns(4)
with col1:
    use_slider_N = st.checkbox("Slider für Gittergröße", value=True)
    if use_slider_N:
        N = st.slider("Gittergröße (N x N)", 10, 250, 50)
    else:
        N = st.number_input("Gittergröße (N x N)", min_value=10, max_value=250, value=50)
with col2:
    use_slider_T = st.checkbox("Slider für Temperatur", value=True)
    if use_slider_T:
        T = st.slider("Temperatur", 0.1, 5.0, 2.5, step=0.01)
    else:
        T = st.number_input("Temperatur", min_value=0.1, max_value=5.0, value=2.5, step=0.01)
with col3:
    use_slider_steps = st.checkbox("Slider für Zeitschritte", value=True)
    if use_slider_steps:
        steps = st.slider("Anzahl Zeitschritte", 10, 1000, 100)
    else:
        steps = st.number_input("Anzahl Zeitschritte", min_value=10, max_value=1000, value=100)
with col4:
    use_slider_m0 = st.checkbox("Slider für Magnetisierung", value=True)
    if use_slider_m0:
        m0 = st.slider("Anfangs-Magnetisierung", -1.0, 1.0, 0.0, step=0.01)
    else:
        m0 = st.number_input("Anfangs-Magnetisierung", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)

# Simulation starten
start_button = st.button("Simulation starten")

if start_button:
    st.info("Simulation läuft...")
    lattice = initialize_lattice(N, m0)
    snapshots, energies, magnetizations = simulate(lattice, T, steps)

    # Darstellung der zeitlichen Entwicklungen
    st.subheader("📊 Zeitliche Entwicklung der Parameter")
    col1, col2 = st.columns(2)
    with col1:
        fig_e, ax_e = plt.subplots(figsize=(8, 4))
        ax_e.plot(energies, 'r-', label='Energie')
        ax_e.set_xlabel('Monte-Carlo-Schritte')
        ax_e.set_ylabel('Energie pro Spin')
        ax_e.grid(True)
        ax_e.legend()
        st.pyplot(fig_e)
    with col2:
        fig_m, ax_m = plt.subplots(figsize=(8, 4))
        ax_m.plot(magnetizations, 'b-', label='Magnetisierung')
        ax_m.set_xlabel('Monte-Carlo-Schritte')
        ax_m.set_ylabel('Magnetisierung pro Spin')
        ax_m.grid(True)
        ax_m.legend()
        st.pyplot(fig_m)

    # Animation der Spins mit Pause-Button direkt daneben
    st.subheader("🎬 Animation der Spins")
    col_gif, col_control = st.columns([4, 1])
    with col_control:
        pause = st.checkbox("⏸️ Pause", value=False)
    with col_gif:
        gif_frames = []
        for idx, snap in enumerate(snapshots[::max(1, steps // 20)]):
            fig, ax = plt.subplots()
            ax.imshow(snap, cmap="binary", vmin=-1, vmax=1)
            ax.set_title(f'Zeitschritt: {idx * max(1, steps // 20)}')
            ax.axis("off")
            fig.tight_layout(pad=0)
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            gif_frames.append(imageio.imread(buf))
            plt.close(fig)
        gif_path = "ising_animation.gif"
        if not pause:
            imageio.mimsave(gif_path, gif_frames, fps=5, loop=0)
            st.image(gif_path, caption="Animation der Spins", use_column_width=True)
        else:
            st.image(gif_frames[-1], caption="Animation pausiert", use_column_width=True)
        with open(gif_path, "rb") as f:
            st.download_button("📥 GIF herunterladen", f, file_name="ising_animation.gif", mime="image/gif")

    # Zusätzliche Darstellungen:
    # 1. Spin-Konfigurationen
    st.subheader("🖼 Spin-Konfigurationen")
    n_snap = min(4, len(snapshots))
    cols_sc = st.columns(n_snap)
    for i in range(n_snap):
        idx = int(i * len(snapshots) / n_snap)
        with cols_sc[i]:
            fig, ax = plt.subplots()
            ax.imshow(snapshots[idx], cmap="binary", vmin=-1, vmax=1)
            ax.set_title(f"Snapshot {idx}")
            ax.axis("off")
            st.pyplot(fig)

    # 2. Histogramm der Magnetisierung
    st.subheader("📊 Histogramm der Magnetisierung")
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    ax_hist.hist(magnetizations, bins=20, color="c", edgecolor='k')
    ax_hist.set_xlabel("Magnetisierung pro Spin")
    ax_hist.set_ylabel("Häufigkeit")
    st.pyplot(fig_hist)

    # 3. Räumliche Korrelationsfunktion (am letzten Snapshot)
    st.subheader("📈 Räumliche Korrelationsfunktion")
    r_values, corr = compute_correlation(snapshots[-1])
    fig_corr, ax_corr = plt.subplots(figsize=(8, 4))
    ax_corr.plot(r_values, corr, 'b-o', label="Korrelation")
    ax_corr.set_xlabel("Abstand r")
    ax_corr.set_ylabel("Korrelationsfunktion")
    ax_corr.legend()
    st.pyplot(fig_corr)

    # 4. Thermodynamische Größen (T‑Scan)
    if st.checkbox("Thermodynamische Simulation (T‑Scan)", value=False):
        T_values = np.linspace(0.1, 5.0, 20)
        Cs, chis = simulate_thermo(N, m0, T_values, steps//2)
        col_thermo1, col_thermo2 = st.columns(2)
        with col_thermo1:
            fig_C, ax_C = plt.subplots(figsize=(8, 4))
            ax_C.plot(T_values, Cs, 'g-o', label="Wärmekapazität")
            ax_C.set_xlabel("Temperatur")
            ax_C.set_ylabel("C pro Spin")
            ax_C.legend()
            st.pyplot(fig_C)
        with col_thermo2:
            fig_chi, ax_chi = plt.subplots(figsize=(8, 4))
            ax_chi.plot(T_values, chis, 'm-o', label="Suszeptibilität")
            ax_chi.set_xlabel("Temperatur")
            ax_chi.set_ylabel("χ pro Spin")
            ax_chi.legend()
            st.pyplot(fig_chi)
