import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import imageio
import imageio_ffmpeg
from io import BytesIO
import os
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Import functions from the separate file
from ising_functions import (initialize_lattice, compute_energy, compute_magnetization,
                             metropolis_step, simulate, compute_correlation, simulate_thermo)

t_c = 2 / np.log(1 + np.sqrt(2))  # Critical temperature for the Ising model

st.set_page_config(layout="wide")
st.title("Interactive Monte Carlo Ising Model")

# Funktion zum Überprüfen und Laden von Bildern
def load_image(image_path):
    """Lädt ein Bild wenn es existiert, sonst gibt None zurück"""
    if os.path.exists(image_path):
        return st.image(image_path, use_column_width=True)
    else:
        st.warning(f"Bild nicht gefunden: {image_path}")
        return Nonez

# Theory Section
with st.expander("Theory of the Ising Model", expanded=True):
    # Text aus README.md laden
    with open("README.md", "r", encoding="utf-8") as f:
        st.markdown(f.read())

# Parameter Section
st.subheader("Adjust Parameters")
col1, col2, col3, col4 = st.columns(4)
with col1:
    use_slider_N = st.checkbox("Use slider for grid size", value=True)
    if use_slider_N:
        N = st.slider("Grid size (N x N)", 10, 1000, 100)
    else:
        N = st.number_input("Grid size (N x N)", min_value=10, max_value=1000, value=50)
with col2:
    use_slider_T = st.checkbox("Use slider for temperature", value=True)
    t_c = 2 / np.log(1 + np.sqrt(2))  # Critical temperature
    if use_slider_T:
        T = st.slider("Temperature", 0.1, 5.0, t_c)  # Default value set to t_c
    else:
        T = st.number_input("Temperature", min_value=0.1, max_value=5.0, value=t_c, step=0.01)  # Default value set to t_c
with col3:
    use_slider_steps = st.checkbox("Use slider for time steps", value=True)
    if use_slider_steps:
        steps = st.slider("Number of time steps", 10, 1000, 100)
    else:
        steps = st.number_input("Number of time steps", min_value=10, max_value=1000, value=100)
with col4:
    use_slider_m0 = st.checkbox("Use slider for magnetization", value=True)
    if use_slider_m0:
        m0 = st.slider("Initial magnetization", -1.0, 1.0, 0.0, step=0.01)
    else:
        m0 = st.number_input("Initial magnetization", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)

# Start Simulation
start_button = st.button("Start Simulation")

if start_button:
    st.info("Simulation is running...")
    lattice = initialize_lattice(N, m0)
    snapshots, energies, magnetizations = simulate(lattice, T, steps)

    # Display time evolution
    st.subheader("Time Evolution of Parameters")
    col1, col2 = st.columns(2)
    with col1:
        fig_e, ax_e = plt.subplots(figsize=(8, 4))
        ax_e.plot(energies, 'r-', label='Energy')
        ax_e.set_xlabel('Monte Carlo Steps')
        ax_e.set_ylabel('Energy per Spin')
        ax_e.grid(True)
        ax_e.legend()
        st.pyplot(fig_e)
    with col2:
        fig_m, ax_m = plt.subplots(figsize=(8, 4))
        ax_m.plot(magnetizations, 'b-', label='Magnetization')
        ax_m.set_xlabel('Monte Carlo Steps')
        ax_m.set_ylabel('Magnetization per Spin')
        ax_m.grid(True)
        ax_m.legend()
        st.pyplot(fig_m)

    # Animation of spins
    st.subheader("Spin Animation")
    col_gif = st.columns(1)[0]
    with col_gif:
        fig, ax = plt.subplots()
        im = ax.imshow(snapshots[0], cmap="seismic", vmin=-1, vmax=1)
        ax.axis("off")

        def update(frame):
            im.set_array(snapshots[frame])
            return [im]

        ani = FuncAnimation(fig, update, frames=range(0, len(snapshots), 2), interval=200, blit=True)

        # Save as HTML5
        html_animation = ani.to_html5_video()

        # Display in Streamlit
        st.components.v1.html(html_animation, height=500)

    # Additional visualizations:
    # 1. Spin configurations
    st.subheader("Spin Configurations")
    n_snap = min(4, len(snapshots))
    cols_sc = st.columns(n_snap)
    for i in range(n_snap):
        idx = int(i * len(snapshots) / n_snap)
        with cols_sc[i]:
            fig, ax = plt.subplots()
            ax.imshow(snapshots[idx], cmap="seismic", vmin=-1, vmax=1)
            ax.set_title(f"Snapshot {idx}")
            ax.axis("off")
            st.pyplot(fig)

imageio_ffmpeg.get_ffmpeg_version()  # Ensure ffmpeg is available
