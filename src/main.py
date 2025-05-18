import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from states import bell_state, rho_b
from bloch import bloch_vectors, omega_m


# Load parameters
with open('config.json') as f:
    config = json.load(f)

M, N, m = config["M"], config["N"], config["m"]
alpha = config["alpha"] if config["alpha"] else np.sqrt(2/(M*(M-1)))
beta = config["beta"] if config["beta"] else np.sqrt(2/(N*(N-1)))

x_points = config.get("x_points",100)
b_points = config.get("b_points", 50)

x_vals = np.linspace(0, 1, x_points)
b_vals = np.linspace(0.01, 0.99, b_points)

K_MN_prime = 0.5 * np.sqrt((2*m*beta**2 + M**2 - M)*(2*m*alpha**2 + N**2 - N))
K_MN = np.sqrt(M*N*(M-1)*(N-1)/2)

# Initialize heatmaps
heatmap_bloch = np.zeros((b_points, x_points))
heatmap_CM = np.zeros((b_points, x_points))

# Compute heatmaps
for i, b in enumerate(b_vals):
    for j, x in enumerate(x_vals):
        rho_x = x * bell_state(M,N) + (1 - x) * rho_b(M,N,b)
        r, s, T = bloch_vectors(rho_x, M, N)

        S_matrix = np.block([
            [alpha * beta * np.ones((m, m)), beta * omega_m(s, m)],
            [alpha * omega_m(r, m).T, T]
        ])

        S_norm = norm(S_matrix, ord='nuc')
        T_norm = norm(T, ord='nuc')

        C_bloch = np.sqrt(8/(M**3*N**2*(M-1))) * (S_norm - K_MN_prime)
        C_CM = np.sqrt(8/(M**3*N**2*(M-1))) * (T_norm - K_MN)

        heatmap_bloch[i, j] = C_bloch
        heatmap_CM[i, j] =  C_CM

heatmap_diff = heatmap_bloch - heatmap_CM

# Function to plot heatmap
def plot_heatmap(data, title, filename):
    plt.figure(figsize=(8,6))
    plt.imshow(data, cmap='Greys', extent=[0,1,0.01,0.99], origin='lower', aspect='auto')
    plt.colorbar()
    plt.xlabel("x parameter")
    plt.ylabel("b parameter")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{filename}', dpi=300)
    plt.close()

# Plot all three heatmaps
plot_heatmap(heatmap_bloch, "Bloch representation bound", "plots/heatmap_bloch.png")
plot_heatmap(heatmap_CM, "Correlation Matrix (CM) bound", "plots/heatmap_CM.png")
plot_heatmap(heatmap_diff, "Difference (Bloch - CM)", "plots/heatmap_difference.png")

# Also plot the line graph from before for reference
plt.figure(figsize=(10,6))
plt.plot(x_vals, heatmap_bloch[b_points//2], label='Bloch representation bound', linewidth=2)
plt.plot(x_vals, heatmap_CM[b_points//2], '--', label='Correlation matrix bound', linewidth=2)
plt.title(f"Lower bounds of concurrence (α={alpha:.2f}, β={beta:.2f}, m={m}, b={b_vals[b_points//2]:.2f})")
plt.xlabel("x parameter")
plt.ylabel("Concurrence lower bound")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('plots/concurrence_bounds.png', dpi=300)
plt.close()
