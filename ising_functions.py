from numba import jit, cuda
import numpy as np
@jit(nopython=True, parallel=True, cache=True)



def initialize_lattice(N, m0):
    num_up = int((1 + m0) / 2 * N * N)
    lattice = np.array([-1] * (N*N - num_up) + [1] * num_up)
    np.random.shuffle(lattice)
    return lattice.reshape((N, N))

@jit(nopython=True, parallel=True, cache=True)
def compute_energy(lattice):
    energy = 0
    N = lattice.shape[0]
    for i in range(N):
        for j in range(N):
            S = lattice[i, j]
            neighbors = (lattice[(i+1)%N, j] + lattice[i, (j+1)%N] +
                         lattice[i-1, j] + lattice[i, j-1])
            energy += -S * neighbors
    return energy / 2

def compute_magnetization(lattice):
    return np.sum(lattice)

@jit(nopython=True, parallel=True, cache=True)
def metropolis_step(lattice, T):
    N = lattice.shape[0]
    random_indices = np.random.randint(0, N, size=(N * N, 2))
    random_probs = np.random.rand(N * N)
    for idx in range(N * N):
        i, j = random_indices[idx]
        S = lattice[i, j]
        neighbors = (lattice[(i+1)%N, j] + lattice[i, (j+1)%N] +
                     lattice[i-1, j] + lattice[i, j-1])
        dE = 2 * S * neighbors
        if dE < 0 or random_probs[idx] < np.exp(-dE / T):
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
        magnetizations.append(compute_magnetization(lattice) / (lattice.shape[0]*lattice.shape[0]))
    return snapshots, energies, magnetizations

def compute_correlation(lattice):
    N = lattice.shape[0]
    max_r = N // 2
    r_values = np.arange(0, max_r + 1)
    corr = np.zeros_like(r_values, dtype=float)
    for idx, r in enumerate(r_values):
        sum_val = 0
        count = 0
        for i in range(N):
            for j in range(N):
                i2 = (i + r) % N
                sum_val += lattice[i, j] * lattice[i2, j]
                count += 1
        corr[idx] = sum_val / count
    return r_values, corr

def simulate_thermo(N, m0, T_values, steps):
    Cs = []
    chis = []
    for T in T_values:
        lattice = initialize_lattice(N, m0)
        energies = []
        magnetizations = []
        for _ in range(steps):
            lattice = metropolis_step(lattice, T)
            energies.append(compute_energy(lattice))
            magnetizations.append(compute_magnetization(lattice))
        E_arr = np.array(energies)
        M_arr = np.array(magnetizations)
        C = (np.mean(E_arr**2) - np.mean(E_arr)**2) / (T**2 * (N*N))
        chi = (np.mean(M_arr**2) - np.mean(M_arr)**2) / (T * (N*N))
        Cs.append(C)
        chis.append(chi)
    return Cs, chis

@cuda.jit
def metropolis_step_gpu(lattice, T, random_indices, random_probs):
    # GPU implementation of metropolis_step
    pass