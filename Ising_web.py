import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from io import BytesIO

# === Ising-Funktionen ===
def initialize_lattice(N, m0):
    num_up = int((1 + m0) / 2 * N * N)
    lattice = np.array([-1] * (N*N - num_up) + [1] * num_up)
    np.random.shuffle(lattice)
    return lattice.reshape((N, N))

def compute_energy(lattice):
    energy = 0
    N = lattice.shape[0]
    for i in range(N):
        for j in range(N):
            S = lattice[i, j]
            neighbors = lattice[(i+1)%N, j] + lattice[i, (j+1)%N] + lattice[i-1, j] + lattice[i, j-1]
            energy += -S * neighbors
    return energy / 2

def compute_magnetization(lattice):
    return np.sum(lattice)

def metropolis_step(lattice, T):
    N = lattice.shape[0]
    for _ in range(N * N):
        i, j = np.random.randint(0, N, size=2)
        S = lattice[i, j]
        neighbors = lattice[(i+1)%N, j] + lattice[i, (j+1)%N] + lattice[i-1, j] + lattice[i, j-1]
        dE = 2 * S * neighbors
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            lattice[i, j] *= -1
    return lattice

def simulate(lattice, T, steps):
    snapshots = []
    energies = []
    magnetizations = []
    for _ in range(steps):
        lattice = metropolis_step(lattice, T)
        snapshots.append(lattice.copy())
        energies.append(compute_energy(lattice))
        magnetizations.append(compute_magnetization(lattice)/(N*N))  # Normierung pro Spin
    return snapshots, energies, magnetizations

# === Streamlit-App ===
st.set_page_config(layout="wide")
st.title("🧲 Interaktive Ising-Modell Simulation")

with st.sidebar:
    st.header("🔧 Parameter")
    N = st.slider("Gittergröße (N x N)", 10, 100, 50)
    T = st.slider("Temperatur", 0.1, 5.0, 2.5, step=0.01)
    steps = st.slider("Anzahl Zeitschritte", 10, 500, 100)
    m0 = st.slider("Anfangs-Magnetisierung (-1 bis +1)", -1.0, 1.0, 0.0, step=0.01)

if st.button("Simulation starten"):
    st.info("Simulation läuft...")
    lattice = initialize_lattice(N, m0)
    snapshots, energies, magnetizations = simulate(lattice, T, steps)

    # Layout für alle Visualisierungen
    st.subheader("📊 Zeitliche Entwicklung der Parameter")
    col1, col2 = st.columns(2)

    with col1:
        # Energie-Plot
        fig_e, ax_e = plt.subplots(figsize=(8, 4))
        ax_e.plot(energies, 'r-', label='Energie')
        ax_e.set_xlabel('Monte-Carlo-Schritte')
        ax_e.set_ylabel('Energie pro Spin')
        ax_e.grid(True)
        ax_e.legend()
        st.pyplot(fig_e)

    with col2:
        # Magnetisierungs-Plot
        fig_m, ax_m = plt.subplots(figsize=(8, 4))
        ax_m.plot(magnetizations, 'b-', label='Magnetisierung')
        ax_m.set_xlabel('Monte-Carlo-Schritte')
        ax_m.set_ylabel('Magnetisierung pro Spin')
        ax_m.grid(True)
        ax_m.legend()
        st.pyplot(fig_m)

    st.subheader("🎬 Animation der Spins")
    # === GIF erzeugen ===
    gif_frames = []
    for snap in snapshots[::max(1, steps // 20)]:
        fig, ax = plt.subplots()
        ax.imshow(snap, cmap="coolwarm", vmin=-1, vmax=1)
        ax.axis("off")
        fig.tight_layout(pad=0)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        gif_frames.append(imageio.imread(buf))
        plt.close(fig)

    gif_path = "ising_animation.gif"
    imageio.mimsave(gif_path, gif_frames, fps=5)

    st.image(gif_path, caption="Animation der Spins", use_column_width=True)

    with open(gif_path, "rb") as f:
        st.download_button("📥 GIF herunterladen", f, file_name="ising_animation.gif", mime="image/gif")
