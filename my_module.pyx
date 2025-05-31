# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport exp

ctypedef np.intc DTYPE_int  # Corrected: np.intc instead of np.intc_t
ctypedef np.float64_t DTYPE_double

# Optimized Metropolis function
def cython_metropolis_step(np.ndarray[DTYPE_int, ndim=2] lattice, double T):
    cdef int N = lattice.shape[0]
    cdef int i, j, k, N2 = N * N
    cdef int S
    cdef double dE, r
    # Loop over all spins in the lattice
    for k in range(N2):
        # Select random lattice position
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        S = lattice[i, j]
        dE = 2 * S * (lattice[(i+1) % N, j] + lattice[i, (j+1) % N] +
                      lattice[(i-1) % N, j] + lattice[i, (j-1) % N])  # Fixed indexing error
        if dE < 0:
            lattice[i, j] = -S
        else:
            r = np.random.rand()
            if r < exp(-dE / T):
                lattice[i, j] = -S
    return lattice

# Optimized simulation function; stores snapshots after each step
def cython_simulate(np.ndarray[DTYPE_int, ndim=2] lattice, double T, int steps):
    cdef int s
    cdef list snapshots = []
    for s in range(steps):
        # Update lattice with the Metropolis step
        lattice = cython_metropolis_step(lattice, T)
        snapshots.append(np.copy(lattice))  # Use np.copy() for a deep copy
    return snapshots