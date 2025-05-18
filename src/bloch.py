import numpy as np
from scipy.linalg import norm

def bloch_vectors11(rho, M, N):
    lambdas_A = [np.kron(gen, np.eye(N)) for gen in su_gen(M)]
    lambdas_B = [np.kron(np.eye(M), gen) for gen in su_gen(N)]

    r = np.array([M/2 * np.trace(rho @ la).real for la in lambdas_A])
    s = np.array([N/2 * np.trace(rho @ lb).real for lb in lambdas_B])
    T = np.array([[M*N/4 * np.trace(rho @ np.kron(la, lb)).real
                   for lb in su_gen(N)] for la in su_gen(M)])
    
    return r, s, T

def bloch_vectors(rho, M, N):
    lambdas_A = [np.kron(gen, np.eye(N)) for gen in su_gen_qubit()]
    lambdas_B = [np.kron(np.eye(M), gen) for gen in gellmann_matrices()]

    r = np.array([M/2*np.trace(rho @ la).real for la in lambdas_A])
    s = np.array([N/2*np.trace(rho @ lb).real for lb in lambdas_B])
    T = np.array([[M*N/4*np.trace(rho @ np.kron(la, lb)).real
                   for lb in gellmann_matrices()] for la in su_gen_qubit()])
    return r, s, T

def bloch_vectors(rho, M, N):
    """Compute Bloch vectors r, s and correlation matrix T for a bipartite state"""
    gens_A = su_generators(M)
    gens_B = su_generators(N)

    # Embed A's generators into full space: λ_A ⊗ I_N
    lambdas_A = [np.kron(gen, np.eye(N)) for gen in gens_A]
    # Embed B's generators into full space: I_M ⊗ λ_B
    lambdas_B = [np.kron(np.eye(M), gen) for gen in gens_B]

    # Compute Bloch vectors
    r = np.array([(M / 2) * np.trace(rho @ la).real for la in lambdas_A])
    s = np.array([(N / 2) * np.trace(rho @ lb).real for lb in lambdas_B])

    # Compute correlation matrix
    T = np.array([[(M * N / 4) * np.trace(rho @ np.kron(la, lb)).real
                   for lb in gens_B] for la in gens_A])

    return r, s, T

def su_gen_qubit():
    return [
        np.array([[0, 1], [1, 0]]), 
        np.array([[0, -1j], [1j, 0]]), 
        np.array([[1, 0], [0, -1]])
    ]

def gellmann_matrices():
    gellmann = []
    # Symmetric off-diagonal
    for i in range(4):
        for j in range(i+1, 4):
            mat = np.zeros((4,4))
            mat[i,j] = mat[j,i] = 1
            gellmann.append(mat)
    # Anti-symmetric off-diagonal
    for i in range(4):
        for j in range(i+1, 4):
            mat = np.zeros((4,4), dtype=complex)
            mat[i,j], mat[j,i] = -1j, 1j
            gellmann.append(mat)
    # Diagonal
    gellmann.append(np.diag([1,-1,0,0])/np.sqrt(2))
    gellmann.append(np.diag([1,1,-2,0])/np.sqrt(6))
    gellmann.append(np.diag([1,1,1,-3])/np.sqrt(12))
    return gellmann


def su_gen(d):
    gens = []
    for i in range(d):
        for j in range(i+1,d):
            mat = np.zeros((d,d))
            mat[i,j] = mat[j,i] = 1
            gens.append(mat)

            mat_im = np.zeros((d,d),dtype=complex)
            mat_im[i,j],mat_im[j,i]=-1j,1j
            gens.append(mat_im)

    for k in range(1,d):
        diag = np.diag([1]*k+[-k]+[0]*(d-k-1))
        mat_diag = np.sqrt(2/(k*(k+1))) * diag
        gens.append(mat_diag)
    return gens


def su_generators(d):
    gens = []
    for i in range(d):
        for j in range(i + 1, d):
            mat_sym = np.zeros((d, d))
            mat_sym[i, j] = mat_sym[j, i] = 1
            gens.append(mat_sym)

            mat_antisym = np.zeros((d, d), dtype=complex)
            mat_antisym[i, j] = -1j
            mat_antisym[j, i] = 1j
            gens.append(mat_antisym)

    for k in range(1, d):
        diag = np.diag([1] * k + [-k] + [0] * (d - k - 1))
        mat_diag = np.sqrt(2 / (k * (k + 1))) * diag
        gens.append(mat_diag)

    return gens

def omega_m(vec, m):
    return np.tile(vec, (m, 1))
