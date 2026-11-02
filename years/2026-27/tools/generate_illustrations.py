"""Rebuild the static teaching illustrations: python tools/generate_illustrations.py.

Run from any directory. Numerical experiments remain in the course notebooks.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREEN, RED = "#6c8ebf", "#d79b00", "#82b366", "#b85450"
INK = "#303030"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.prop_cycle": plt.cycler(color=[BLUE, RED, GREEN, ORANGE]),
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888888", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "grid.color": "#dddddd", "grid.alpha": 0.6,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def save(fig, name, folder="NN"):
    path = ROOT / folder / "fig" / (name + ".png")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(path.relative_to(ROOT))


def axes_labels(ax, title, x="x", y="y"):
    ax.set(title=title, xlabel=x, ylabel=y)
    ax.grid(True)
    ax.set_axisbelow(True)


def network(sizes, labels, captions, name, weights=False):
    fig, ax = plt.subplots(figsize=(8 if len(sizes) == 3 else 6, 4))
    positions = [[(2.7 * l, y) for y in np.linspace((n-1)/2, -(n-1)/2, n)]
                 for l, n in enumerate(sizes)]
    radius = 0.30
    for left, right in zip(positions[:-1], positions[1:]):
        for i, p in enumerate(left):
            for q in right:
                ax.add_patch(FancyArrowPatch(p, q, arrowstyle="->", mutation_scale=12,
                                            shrinkA=23, shrinkB=23, color="#777777", lw=1))
                if weights:
                    x, y = np.array(p) * 0.62 + np.array(q) * 0.38
                    ax.text(x, y + 0.12, rf"$w_{i+1}$", ha="center", fontsize=12,
                            bbox=dict(facecolor="white", edgecolor="none", pad=1))
    colors = ["#dae8fc", "#ffe6cc", "#d5e8d4"] if len(sizes) == 3 else ["#dae8fc", "#d5e8d4"]
    for l, layer in enumerate(positions):
        for i, (x, y) in enumerate(layer):
            ax.add_patch(Circle((x, y), radius, fc=colors[l], ec=INK, lw=1, zorder=3))
            ax.text(x, y, labels[l][i], ha="center", va="center", fontsize=13, zorder=4)
        ax.text(2.7*l, -1.65, captions[l], ha="center", va="top")
    if weights:
        ax.text(2.7, 0.65, "$+\\,b$", ha="center", fontsize=12)
        ax.text(3.55, 0, r"$\hat{y}$", ha="center", va="center", fontsize=14)
        ax.add_patch(FancyArrowPatch((3.02, 0), (3.4, 0), arrowstyle="->", mutation_scale=12, color=INK))
    ax.set(xlim=(-0.6, 2.7*(len(sizes)-1)+1.1), ylim=(-2.25, 1.6), aspect="equal")
    ax.axis("off")
    save(fig, name)


def main():
    # Nested fields with an overlapping Data Science domain.
    fig, ax = plt.subplots(figsize=(9, 5))
    for xy, w, h, color in [((0, 0), 7, 5, "#dae8fc"),
                            ((0.1, -0.45), 4.8, 3.4, "#b4cceb"),
                            ((0.1, -0.9), 2.7, 1.65, "#8fafd6"),
                            ((3.6, -0.4), 4, 3.4, "#ffe6cc")]:
        ax.add_patch(Ellipse(xy, w, h, fc=color, ec=BLUE if color != "#ffe6cc" else ORANGE,
                             alpha=0.8, lw=1.2))
    for x, y, label in [(-0.8, 1.65, "Artificial Intelligence"),
                         (-0.3, 0.55, "Machine Learning"),
                         (0.1, -0.9, "Deep Learning"), (3.9, -0.1, "Data Science")]:
        ax.text(x, y, label, ha="center", va="center", fontsize=12)
    ax.set(xlim=(-3.8, 5.9), ylim=(-2.8, 2.8), aspect="equal")
    ax.axis("off")
    save(fig, "ai_ml_dl_data_science")

    xx = np.linspace(-5, 5, 10)
    rng = np.random.RandomState(42)
    x = np.linspace(-5, 5, 20)
    data = 1 + 0.2*x + rng.normal(0, 0.5, x.shape)
    for with_data, name in [(False, "linear_model_parameters"), (True, "linear_model_fit")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xx, 1+0.2*xx, "o--", label=r"$\Theta=(1,\,0.2)$", ms=4)
        ax.plot(xx, -xx, "o--", label=r"$\Theta=(0,\,-1)$", ms=4)
        if with_data:
            ax.scatter(x, data, color=GREEN, label="Data", zorder=3)
        axes_labels(ax, r"Model predictions" + (" and observed data" if with_data else ""))
        ax.legend()
        save(fig, name)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xx, 0.1*xx, "o--", label="Linear model", ms=4)
    ax.scatter(x, np.sin(x), color=GREEN, label="Data")
    axes_labels(ax, r"A linear model approximating $\sin(x)$")
    ax.legend()
    save(fig, "linear_model_limitations")

    rng = np.random.RandomState(0)
    x = rng.uniform(-3, 3, 100)
    y = np.sin(x)+0.2*rng.normal(size=x.shape)
    X = rng.normal(size=(200, 2))
    cls = (np.sum(X**2, axis=1) < 1).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    axes[0].scatter(x, y, s=20, color=BLUE)
    axes_labels(axes[0], r"Regression: $y\in\mathbb{R}$")
    for k, color in enumerate([BLUE, RED]):
        axes[1].scatter(*X[cls == k].T, s=20, color=color, label=f"Class {k}")
    axes_labels(axes[1], r"Classification: $y\in\{0,1\}$", "$x_1$", "$x_2$")
    axes[1].set_aspect("equal")
    axes[1].legend()
    save(fig, "regression_classification_examples")

    network([3, 1], [["$x_1$", "$x_2$", "$x_3$"], [r"$\Sigma, H$"]],
            ["Input", "Neuron"], "artificial_neuron", weights=True)
    network([2, 3, 1], [["$x_1$", "$x_2$"], ["$h_1$", "$h_2$", "$h_3$"], [r"$\hat{y}$"]],
            ["Input", "Hidden layer\n(one neuron per edge)", "Output\n(AND)"], "triangle_network")

    rng = np.random.RandomState(0)
    X = rng.uniform(-1.5, 1.5, (1000, 2))
    def hidden(X):
        return np.stack([X[:, 1]+0.7 >= 0, 1.7*X[:, 0]-X[:, 1]+1 >= 0,
                         -1.7*X[:, 0]-X[:, 1]+1 >= 0], axis=1)
    y = hidden(X).all(axis=1)
    fig, ax = plt.subplots(figsize=(5, 5))
    for k, color in enumerate([BLUE, RED]):
        ax.scatter(*X[y == k].T, s=10, color=color, label=f"Class {k}")
    axes_labels(ax, "A non linearly separable dataset", "$x_1$", "$x_2$")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    save(fig, "triangle_dataset")

    a, b = np.meshgrid(np.linspace(-1.5, 1.5, 300), np.linspace(-1.5, 1.5, 300))
    h = hidden(np.column_stack([a.ravel(), b.ravel()]))
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), layout="constrained")
    for j, ax in enumerate(axes):
        z = h[:, j] if j < 3 else h.all(axis=1)
        ax.contourf(a, b, z.reshape(a.shape), levels=[-0.5, 0.5, 1.5],
                    colors=["white", "#ffe6cc" if j < 3 else "#d5e8d4"])
        ax.contour(a, b, z.reshape(a.shape), levels=[0.5], colors=[ORANGE if j < 3 else GREEN], linewidths=1)
        axes_labels(ax, rf"Neuron $h_{j+1}$" if j < 3 else r"Output $\hat{y}$ (AND)", "$x_1$", "$x_2$")
        ax.set_aspect("equal")
    axes[3].scatter(*X[y].T, s=3, color=GREEN)
    save(fig, "triangle_activation_regions")

    t = np.linspace(-5, 5, 500)
    sigmoid = 1/(1+np.exp(-t))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    values = [(t >= 0, sigmoid, np.tanh(t), np.maximum(0, t)),
              (np.where(t == 0, np.nan, 0), sigmoid*(1-sigmoid), 1-np.tanh(t)**2,
               np.where(t == 0, np.nan, (t > 0).astype(float)))]
    for j, ax in enumerate(axes):
        for z, label, color in zip(values[j], ["Heaviside", "Sigmoid", "tanh", "ReLU"], [BLUE, ORANGE, GREEN, RED]):
            if label == "Heaviside" or (j == 1 and label == "ReLU"):
                for mask in [t < 0, t > 0]:
                    ax.plot(t[mask], z[mask], color=color, label=label if mask[0] else None)
                for level in ([0, 1] if j == 0 or label == "ReLU" else [0]):
                    ax.plot(0, level, "o", mfc="white", mec=color, ms=4)
                if j == 0:
                    ax.plot(0, 1, "o", mfc=color, mec=color, ms=4)
            else:
                ax.plot(t, z, color=color, label=label)
        axes_labels(ax, "Activation functions" if j == 0 else "Their derivatives (away from zero)", "t", "")
        ax.set_ylim((-1.5, 2.5) if j == 0 else (-0.5, 1.5))
        ax.legend()
    save(fig, "activation_functions_derivatives")

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), layout="constrained")
    xx = np.linspace(-4.1, 4.1, 400)
    for ax, alpha, regime in zip(axes, [0.05, 0.5, 0.9, 1.05],
                                  ["Slow convergence", "One-step convergence", "Oscillatory convergence", "Divergence"]):
        xs = 2*(1-2*alpha)**np.arange(11)
        ax.plot(xx, xx**2, color=BLUE, lw=1.4)
        ax.plot(xs, xs**2, "o-", color=RED, ms=3)
        ax.plot(0, 0, "*", color=GREEN, ms=10, zorder=4)
        axes_labels(ax, regime+"\n"+rf"$\alpha={alpha}$, $\alpha\lambda={2*alpha:.1f}$", "x", "f(x)")
        ax.set(xlim=(-4.1, 4.1), ylim=(-0.8, 18))
    save(fig, "step_size_regimes", "Optimization")


if __name__ == "__main__":
    main()
