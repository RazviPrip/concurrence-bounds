import numpy as np

def bell_state(M, N):
    """Generalized Bell-like state for dimensions M,N"""
    state_vector = np.zeros((M * N, 1))
    state_vector[0] = 1
    min_dim = min(M, N)
    state_vector[::N+1][:min_dim] = 1  # properly spaced indices for |00⟩ + |11⟩ + ... style state
    state_vector /= np.sqrt(min_dim)
    return state_vector @ state_vector.T

def rho_b(M, N, b):
    """Generalized bound-entangled state."""
    dim = M * N
    rho = np.eye(dim) * b / dim
    rho[-1,-1] = (1 + b)/2
    rho[0,-1] = rho[-1,0] = np.sqrt(1 - b**2)/2
    return rho
