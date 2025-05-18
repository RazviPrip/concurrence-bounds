import numpy as np
import matplotlib.pyplot as plt
from bounds import theorem1_bound, cm_bound

def construct_rho_x(x, b):
    ket = np.zeros((8, 1))
    ket[0, 0] = 1 / np.sqrt(2)
    ket[3, 0] = 1 / np.sqrt(2)
    rho_pure = ket @ ket.T

    rho_b = np.array([
        [b, 0, 0, 0, 0, b, 0, 0],
        [0, b, 0, 0, 0, 0, b, 0],
        [0, 0, b, 0, 0, 0, 0, b],
        [0, 0, 0, b, 0, 0, 0, 0],
        [0, 0, 0, 0, (1+b)/2, 0, 0, np.sqrt(1 - b**2)/2],
        [b, 0, 0, 0, 0, b, 0, 0],
        [0, b, 0, 0, 0, 0, b, 0],
        [0, 0, b, 0, np.sqrt(1 - b**2)/2, 0, 0, (1+b)/2]
    ])

    return x * rho_pure + (1 - x) * rho_b

def trace_norm(matrix):
    return np.linalg.norm(matrix, ord='nuc')

def plot_comparison(args):
    xs = np.linspace(0, 1, 100)
    bound1 = []
    bound_cm = []
    for x in xs:
        rho = construct_rho_x(x, b=0.5)
        tr_norm = trace_norm(rho)
        tr_norm_cm = trace_norm(rho)
        bound1.append(theorem1_bound(tr_norm, args.m, args.M, args.N, args.alpha, args.beta))
        bound_cm.append(cm_bound(tr_norm_cm, args.M, args.N))

    plt.plot(xs, bound1, label="Theorem 1 Bound")
    plt.plot(xs, bound_cm, '--', label="CM Bound")
    plt.xlabel('x')
    plt.ylabel('Lower Bound of Concurrence')
    plt.legend()
    plt.title('Comparison of Bounds for ρₓ')
    plt.savefig("output/comparison_plot.png")
    plt.close()

def plot_heatmaps(args):
    bs = np.linspace(0.01, 0.99, 50)
    xs = np.linspace(0, 1, 50)
    heat_theorem1 = np.zeros((len(bs), len(xs)))
    heat_cm = np.zeros((len(bs), len(xs)))

    for i, b in enumerate(bs):
        for j, x in enumerate(xs):
            rho = construct_rho_x(x, b)
            tr_norm = trace_norm(rho)
            tr_norm_cm = trace_norm(rho)
            heat_theorem1[i, j] = theorem1_bound(tr_norm, args.m, args.M, args.N, args.alpha, args.beta)
            heat_cm[i, j] = cm_bound(tr_norm_cm, args.M, args.N)

    diff = heat_theorem1 - heat_cm
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, title in zip(axs, [heat_theorem1, heat_cm, diff],
                               ["Theorem 1", "CM Bound", "Difference"]):
        c = ax.imshow(data, extent=[0, 1, 0.01, 0.99], origin='lower', cmap='gray', aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('b')
        fig.colorbar(c, ax=ax)
    plt.tight_layout()
    plt.savefig("output/heatmaps.png")
    plt.close()