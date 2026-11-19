"""
Task 09 - Eigenvalues and Eigenvectors
BIT 2118 Scientific Computing

Part 0: Concept check (A v = lambda v).
Part 1: Original problem - which node of a small network is most influential?
Part 2: Changing the inputs (add/remove links, star network, weighted link).
Part 3: Related real-world case - natural vibration frequencies of a 3-storey building.

Just run this one file. It will:
  1. print the results in the terminal,
  2. save them to output/task_09_output.txt,
  3. draw ONE summary diagram,
  4. save it to output/task_09_diagram.png and open it automatically.

Run from the repo root:  python code/Task_09_eigenvalues.py
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

np.set_printoptions(precision=4, suppress=True)

lines = []


def show(text=""):
    print(text)
    lines.append(str(text))


def header(title):
    show("=" * 64)
    show(title)
    show("=" * 64)


def dominant_pair(A):
    """Dominant eigenvalue, its eigenvector, and the importance scores."""
    eigenvalues, eigenvectors = np.linalg.eig(A)
    index = np.argmax(np.real(eigenvalues))
    value = np.real(eigenvalues[index])
    v = np.real(eigenvectors[:, index])
    v = v if v.sum() >= 0 else -v                  # eigenvectors can have either sign
    centrality = np.abs(v) / np.abs(v).sum()
    return value, v, centrality, np.sort(np.real(eigenvalues))[::-1]


def top_nodes(cent, names=("A", "B", "C", "D")):
    """Names of the highest-scoring node(s); ties are reported together."""
    best = np.max(cent)
    winners = [n for n, c in zip(names, cent) if abs(c - best) < 1e-9]
    if len(winners) == len(names):
        return "all nodes equal"
    return " and ".join(winners) + (" (tied)" if len(winners) > 1 else "")


# ---------------------------------------------------------------------------
# PART 0: Concept check
# ---------------------------------------------------------------------------
header("PART 0: CONCEPT CHECK  (A v = lambda v)")
A0 = np.array([[2.0, 1.0], [1.0, 2.0]])
vals0, vecs0 = np.linalg.eig(A0)
vals0, vecs0 = np.real(vals0), np.real(vecs0)
order = np.argsort(vals0)[::-1]
vals0, vecs0 = vals0[order], vecs0[:, order]
show(f"A =\n{A0}")
show(f"Eigenvalues: {vals0}")
for k in range(2):
    v = vecs0[:, k]
    show(f"lambda = {vals0[k]:.0f}, v = {v}:  A @ v = {A0 @ v}  =  lambda * v = {vals0[k] * v}")
show("A v = 3 v: applying A triples the length of v without changing its direction.")
show()

# ---------------------------------------------------------------------------
# PART 1: Original problem - network importance
# ---------------------------------------------------------------------------
nodes = ["A", "B", "C", "D"]
A = np.array([
    [0., 1., 1., 0.],
    [1., 0., 1., 1.],
    [1., 1., 0., 1.],
    [0., 1., 1., 0.]
])
value, v, centrality, spectrum = dominant_pair(A)
degree = A.sum(axis=1)

header("PART 1: NETWORK WITH NODES A, B, C, D")
show(f"Adjacency matrix A =\n{A}")
show(f"All eigenvalues: {spectrum}")
show(f"Dominant eigenvalue: {value:.4f}")
show("Eigenvector-based importance:")
for n, s, d in zip(nodes, centrality, degree):
    show(f"   {n}: {s:.4f}   (number of links = {d:.0f})")
show(f"Verification error ||A v - lambda v||: {np.linalg.norm(A @ v - value * v):.2e}")

# Power iteration: how the dominant eigenvector can be found without eig()
x = np.ones(4) / 2
power_err = []
for _ in range(25):
    x = A @ x
    x = x / np.linalg.norm(x)
    power_err.append(np.linalg.norm(np.abs(x) - np.abs(v) / np.linalg.norm(v)))
show(f"Power iteration error after 5 / 10 / 25 steps: "
     f"{power_err[4]:.1e} / {power_err[9]:.1e} / {power_err[24]:.1e}")
show()

# ---------------------------------------------------------------------------
# PART 2: Changing the inputs
# ---------------------------------------------------------------------------
def with_edge(M, i, j, w=1.0):
    M = M.copy(); M[i, j] = M[j, i] = w
    return M


scenarios = [
    ("1. Original",             A),
    ("2. Add link A-D",         with_edge(A, 0, 3)),
    ("3. Remove link B-C",      with_edge(A, 1, 2, 0.0)),
    ("4. Star centred on B",    np.array([[0, 1, 0, 0], [1, 0, 1, 1],
                                          [0, 1, 0, 0], [0, 1, 0, 0]], float)),
    ("5. Link B-C weight 3",    with_edge(A, 1, 2, 3.0)),
]

header("PART 2: CHANGING THE INPUTS")
scen_vals, scen_cent = [], []
for name, M in scenarios:
    val, vec, cent, _ = dominant_pair(M)
    scen_vals.append(val); scen_cent.append(cent)
    show(f"--- {name} ---")
    show(f"Dominant eigenvalue: {val:.4f}")
    show("Importance A, B, C, D: " + ", ".join(f"{c:.4f}" for c in cent))
    show(f"Most influential node: {top_nodes(cent)}")
    show()

two_pairs = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], float)
show("--- 6. Disconnected network (A-B and C-D only) ---")
show(f"Eigenvalues: {np.sort(np.linalg.eigvals(two_pairs).real)[::-1]}")
show("The dominant eigenvalue 1 is repeated, so the eigenvector is not unique "
     "and the scores are not meaningful.")
show()

# ---------------------------------------------------------------------------
# PART 3: Related real-world case - building vibration
# ---------------------------------------------------------------------------
header("PART 3: NATURAL FREQUENCIES OF A 3-STOREY BUILDING")
m_floor = 1.0e4          # mass of each floor (kg)
k_story = 2.0e6          # stiffness between floors (N/m)
K = k_story * np.array([[ 2.0, -1.0,  0.0],
                        [-1.0,  2.0, -1.0],
                        [ 0.0, -1.0,  1.0]])
M_inv_K = K / m_floor                          # equal masses, so M^-1 K = K / m
w2, modes = np.linalg.eigh(M_inv_K)            # w2 = omega^2 (symmetric matrix)
omega = np.sqrt(w2)
freq = omega / (2 * np.pi)
modes = modes * np.sign(modes[-1, :])          # make the top floor move "up" in every mode
show(f"Assumptions: 3 floors, each {m_floor:.0f} kg, storey stiffness {k_story:.0e} N/m, "
     "linear springs, no damping.")
show("Eigenvalue problem: (K / m) v = omega^2 v")
for i in range(3):
    show(f"Mode {i + 1}: omega^2 = {w2[i]:10.2f},  frequency = {freq[i]:.3f} Hz,  "
         f"shape (floors 1-3) = {modes[:, i]}")

freq_stiff = np.sqrt(np.linalg.eigvalsh(2 * M_inv_K)) / (2 * np.pi)
show(f"\nDoubling the stiffness: frequencies = {freq_stiff} Hz  "
     f"(ratio {freq_stiff[0] / freq[0]:.3f} = sqrt(2))")
freq_heavy = np.sqrt(np.linalg.eigvalsh(K / (2 * m_floor))) / (2 * np.pi)
show(f"Doubling the floor mass: frequencies = {freq_heavy} Hz  "
     f"(ratio {freq_heavy[0] / freq[0]:.3f} = 1/sqrt(2))")
show()

# ---------------------------------------------------------------------------
# Save text output
# ---------------------------------------------------------------------------
output_dir = Path(__file__).resolve().parent.parent / "output"
output_dir.mkdir(exist_ok=True)
(output_dir / "task_09_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Output saved to: {output_dir / 'task_09_output.txt'}")

# ---------------------------------------------------------------------------
# ONE summary diagram
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 11.5))
gs = fig.add_gridspec(3, 3, height_ratios=[0.7, 1, 1], hspace=0.5, wspace=0.3)
fig.suptitle("Task 09: Eigenvalues and Eigenvectors - Summary Diagram",
             fontsize=16, fontweight="bold")

# ---- Row 1: flowchart ----
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 100); ax.set_ylim(0, 20); ax.axis("off")
ax.set_title("Method: network -> adjacency matrix -> dominant eigenvector -> importance scores",
             fontsize=12)
boxes = [
    (1,  "Network\nnodes A, B, C, D\nand their links",        "#dbe9f6"),
    (22, "Adjacency matrix\nA[i][j] = 1 if\nnodes i, j linked", "#fde2c4"),
    (43, "Eigen-decomposition\nnp.linalg.eig(A)",              "#fff3c4"),
    (64, "Dominant pair\nlargest lambda,\nits eigenvector v",  "#e3f2d9"),
    (84, "Importance\nv / sum(v)\ncheck A v = lambda v",       "#f6d9e8"),
]
for xpos, text, col in boxes:
    ax.add_patch(FancyBboxPatch((xpos, 4.5), 15, 11, boxstyle="round,pad=0.4",
                                fc=col, ec="black", lw=1.2))
    ax.text(xpos + 7.5, 10, text, ha="center", va="center", fontsize=8.5)
for x0 in (16.5, 37.5, 58.5, 79.5):
    ax.add_patch(FancyArrowPatch((x0 + 0.5, 10), (x0 + 5, 10), arrowstyle="-|>",
                                 mutation_scale=18, lw=1.5, color="black"))

# ---- Row 2, col 1: network drawing ----
ax = fig.add_subplot(gs[1, 0])
pos = {0: (0, 1), 1: (1, 1.7), 2: (1, 0.3), 3: (2, 1)}
for i in range(4):
    for j in range(i + 1, 4):
        if A[i, j]:
            ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], "grey", lw=2, zorder=1)
sc = ax.scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()],
                s=centrality * 9000, c=centrality, cmap="viridis", zorder=2,
                edgecolors="black")
for i, n in enumerate(nodes):
    ax.text(pos[i][0], pos[i][1], f"{n}\n{centrality[i]:.3f}", ha="center", va="center",
            fontsize=9, color="white", fontweight="bold", zorder=3)
ax.set_xlim(-0.6, 2.6); ax.set_ylim(-0.4, 2.4); ax.axis("off")
ax.set_title(f"Part 1: network (node size = importance)\ndominant eigenvalue = {value:.4f}")

# ---- Row 2, col 2: centrality vs degree ----
ax = fig.add_subplot(gs[1, 1])
xt = np.arange(4); w = 0.38
b1 = ax.bar(xt - w / 2, centrality, w, label="Eigenvector importance")
b2 = ax.bar(xt + w / 2, degree / degree.sum(), w, label="Share of links (degree)")
ax.bar_label(b1, fmt="%.3f", fontsize=8); ax.bar_label(b2, fmt="%.3f", fontsize=8)
ax.set_xticks(xt); ax.set_xticklabels(nodes)
ax.set_ylabel("Score (sums to 1)"); ax.set_ylim(0, 0.40)
ax.set_title("Part 1: B and C are the most influential")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

# ---- Row 2, col 3: eigenvector geometry ----
ax = fig.add_subplot(gs[1, 2])
th = np.linspace(0, 2 * np.pi, 200)
circle = np.vstack([np.cos(th), np.sin(th)])
ellipse = A0 @ circle
ax.plot(*circle, "b-", label="unit circle (before)")
ax.plot(*ellipse, "r-", label="after A")
for k, col in zip(range(2), ["green", "purple"]):
    vv = vecs0[:, k]
    ax.annotate("", xy=vals0[k] * vv, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
    ax.annotate("", xy=vv, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5, ls="--"))
    ax.text(*(vals0[k] * vv * 1.1), f"lambda={vals0[k]:.0f}", color=col, fontsize=9)
ax.set_xlim(-3.4, 3.4); ax.set_ylim(-3.4, 3.4); ax.set_aspect("equal")
ax.axhline(0, color="grey", lw=0.8); ax.axvline(0, color="grey", lw=0.8)
ax.set_title("Part 0: eigenvectors keep their direction\n(A = [[2,1],[1,2]])", fontsize=10)
ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)

# ---- Row 3, col 1: scenario centralities ----
ax = fig.add_subplot(gs[2, 0])
w = 0.16
for k, (name, _) in enumerate(scenarios):
    ax.bar(np.arange(4) + (k - 2) * w, scen_cent[k], w, label=name)
ax.set_xticks(range(4)); ax.set_xticklabels(nodes)
ax.set_ylabel("Importance"); ax.set_ylim(0, 0.52)
ax.set_title("Part 2: importance after changing the network")
ax.legend(fontsize=6.5, ncol=2, loc="upper center"); ax.grid(axis="y", alpha=0.3)

# ---- Row 3, col 2: power iteration ----
ax = fig.add_subplot(gs[2, 1])
ax.semilogy(range(1, 26), np.maximum(power_err, 1e-17), "o-")
ax.set_xlabel("Iteration"); ax.set_ylabel("Error vs eig() eigenvector")
abs_sorted = np.sort(np.abs(spectrum))[::-1]
ax.set_title(f"Part 1: power iteration converges\n"
             f"(speed set by |lambda2| / lambda1 = {abs_sorted[1] / abs_sorted[0]:.2f})", fontsize=10)
ax.grid(alpha=0.3, which="both")

# ---- Row 3, col 3: building mode shapes ----
ax = fig.add_subplot(gs[2, 2])
floors = np.arange(0, 4)
for i, col in enumerate(["#1f77b4", "#ff7f0e", "#2ca02c"]):
    shape = np.concatenate(([0.0], modes[:, i]))
    ax.plot(shape, floors, "o-", color=col, lw=2, label=f"Mode {i + 1}: {freq[i]:.2f} Hz")
ax.axvline(0, color="grey", lw=0.8)
ax.set_yticks(floors); ax.set_yticklabels(["Ground", "Floor 1", "Floor 2", "Floor 3"])
ax.set_xlabel("Sideways displacement (relative)")
ax.set_title("Part 3: building vibration modes")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plot_file = output_dir / "task_09_diagram.png"
fig.savefig(plot_file, dpi=150, bbox_inches="tight")
print(f"Diagram saved to: {plot_file}")

try:
    if sys.platform.startswith("win"):
        os.startfile(str(plot_file))
    elif sys.platform == "darwin":
        subprocess.run(["open", str(plot_file)], check=False)
    else:
        subprocess.run(["xdg-open", str(plot_file)], check=False, stderr=subprocess.DEVNULL)
except Exception:
    print("Could not open the image automatically - open it from the output/ folder.")