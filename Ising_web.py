import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import imageio
from io import BytesIO
import os

# Import functions from the separate file
from ising_functions import (initialize_lattice, compute_energy, compute_magnetization,
                             metropolis_step, simulate, compute_correlation, simulate_thermo)

t_c = 2 / np.log(1 + np.sqrt(2))  # Critical temperature for the Ising model

st.set_page_config(layout="wide")
st.title("Interactive Ising Model Simulation")

# Theory Section
with st.expander("Theory of the Ising Model", expanded=True):
    with open("README.md", "r", encoding="utf-8") as f:
        st.markdown(f.read())

# Parameter Section
st.subheader("Simulation Parameters")
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

    # Animation of spins with pause button
    st.subheader("Spin Animation")
    col_gif, col_control = st.columns([4, 1])
    with col_control:
        pause = st.checkbox("Pause", value=False)
    with col_gif:
        gif_frames = []
        for idx, snap in enumerate(snapshots[::max(1, steps // 20)]):
            fig, ax = plt.subplots()
            ax.imshow(snap, cmap="binary", vmin=-1, vmax=1)
            ax.set_title(f'Time Step: {idx * max(1, steps // 20)}')
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
            st.image(gif_path, caption="Spin Animation", use_column_width=True)
        else:
            st.image(gif_frames[-1], caption="Animation Paused", use_column_width=True)
        with open(gif_path, "rb") as f:
            st.download_button("Download GIF", f, file_name="ising_animation.gif", mime="image/gif")

    # Additional visualizations:
    # 1. Spin configurations
    st.subheader("Spin Configurations")
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

    # 2. Histogram of magnetization
    st.subheader("Histogram of Magnetization")
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    ax_hist.hist(magnetizations, bins=20, color="c", edgecolor='k')
    ax_hist.set_xlabel("Magnetization per Spin")
    ax_hist.set_ylabel("Frequency")
    st.pyplot(fig_hist)

    # 3. Spatial correlation function (last snapshot)
    st.subheader("Spatial Correlation Function")
    r_values, corr = compute_correlation(snapshots[-1])
    fig_corr, ax_corr = plt.subplots(figsize=(8, 4))
    ax_corr.plot(r_values, corr, 'b-o', label="Correlation")
    ax_corr.set_xlabel("Distance r")
    ax_corr.set_ylabel("Correlation Function")
    ax_corr.legend()
    st.pyplot(fig_corr)

    # 4. Thermodynamic quantities (T-scan)
    if st.checkbox("Thermodynamic Simulation (T-scan)", value=False):
        T_values = np.linspace(0.1, 5.0, 20)
        Cs, chis = simulate_thermo(N, m0, T_values, steps // 2)
        col_thermo1, col_thermo2 = st.columns(2)
        with col_thermo1:
            fig_C, ax_C = plt.subplots(figsize=(8, 4))
            ax_C.plot(T_values, Cs, 'g-o', label="Heat Capacity")
            ax_C.set_xlabel("Temperature")
            ax_C.set_ylabel("C per Spin")
            ax_C.legend()
            st.pyplot(fig_C)
        with col_thermo2:
            fig_chi, ax_chi = plt.subplots(figsize=(8, 4))
            ax_chi.plot(T_values, chis, 'm-o', label="Susceptibility")
            ax_chi.set_xlabel("Temperature")
            ax_chi.set_ylabel("χ per Spin")
            ax_chi.legend()
            st.pyplot(fig_chi)
