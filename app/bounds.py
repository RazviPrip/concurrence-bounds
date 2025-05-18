import numpy as np

def k_prime_mn(m, M, N, alpha, beta):
    return 0.5 * np.sqrt((2 * m * beta**2 + M**2 - M) * (2 * m * alpha**2 + N**2 - N))

def theorem1_bound(trace_norm, m, M, N, alpha, beta):
    numerator = 8
    denominator = M**3 * N**2 * (M - 1)
    k_prime = k_prime_mn(m, M, N, alpha, beta)
    return np.sqrt(numerator / denominator) * (trace_norm - k_prime)

def cm_bound(trace_norm_cm, M, N):
    kmn = np.sqrt(M * N * (M - 1) * (N - 1)) / 2
    return np.sqrt(8 / (M**3 * N**2 * (M - 1))) * (trace_norm_cm - kmn)