"""
Task 08 - Iterative Methods for Linear Systems (Jacobi and Gauss-Seidel)
BIT 2118 Scientific Computing

Part 1: Original problem - steady-state temperatures at 5 interior grid points.
Part 2: Changing the inputs (tolerance, starting guess, boundary values,
        number of points, and a matrix that is NOT diagonally dominant).
Part 3: Related real-world case - temperature of a 2-D heated metal plate.

Just run this one file. It will:
  1. print the results in the terminal,
  2. save them to output/task_08_output.txt,
  3. draw ONE summary diagram,
  4. save it to output/task_08_diagram.png and open it automatically.

Run from the repo root:  python code/Task_08_iterative_linear.py
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


# ---------------------------------------------------------------------------
# The two iterative methods
# ---------------------------------------------------------------------------
def jacobi(A, b, tol=1e-8, max_iter=500, x0=None):
    """Jacobi: every new value uses only the OLD estimates."""
    D = np.diag(A)
    R = A - np.diagflat(D)
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)
    errors = []
    for _ in range(max_iter):
        x_new = (b - R @ x) / D
        error = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(error)
        x = x_new
        if error < tol:
            break
    return x, errors


def gauss_seidel(A, b, tol=1e-8, max_iter=500, x0=None):
    """Gauss-Seidel: each new value is used immediately in the same sweep."""
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)
    errors = []
    for _ in range(max_iter):
        old = x.copy()
        for i in range(len(b)):
            x[i] = (b[i] - A[i, :i] @ x[:i] - A[i, i + 1:] @ old[i + 1:]) / A[i, i]
        error = np.linalg.norm(x - old, ord=np.inf)
        errors.append(error)
        if error < tol:
            break
    return x, errors


def line_system(n, left, right):
    """Steady-state 1-D temperatures at n interior points between two ends."""
    A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    b = np.zeros(n)
    b[0], b[-1] = left, right
    return A, b


def is_diagonally_dominant(A):
    d = np.abs(np.diag(A))
    off = np.sum(np.abs(A), axis=1) - d
    return bool(np.all(d >= off) and np.any(d > off))


# ---------------------------------------------------------------------------
# PART 1: Original problem
# ---------------------------------------------------------------------------
header("PART 1: 5 INTERIOR POINTS, LEFT = 100 C, RIGHT = 20 C")
A, b = line_system(5, 100.0, 20.0)
x_j, err_j = jacobi(A, b)
x_gs, err_gs = gauss_seidel(A, b)
x_exact = np.linalg.solve(A, b)

show(f"A =\n{A}")
show(f"b = {b}")
show(f"Diagonally dominant? {is_diagonally_dominant(A)}\n")
show(f"Jacobi temperatures       : {x_j}   ({len(err_j)} iterations)")
show(f"Gauss-Seidel temperatures : {x_gs}   ({len(err_gs)} iterations)")
show(f"Direct solution           : {x_exact}")
show(f"Max difference Jacobi vs direct       : {np.max(np.abs(x_j - x_exact)):.2e}")
show(f"Max difference Gauss-Seidel vs direct : {np.max(np.abs(x_gs - x_exact)):.2e}")
show(f"Gauss-Seidel needed about {len(err_j) / len(err_gs):.1f}x fewer iterations.")
show()

# ---------------------------------------------------------------------------
# PART 2: Changing the inputs
# ---------------------------------------------------------------------------
header("PART 2: CHANGING THE INPUTS")

show("a) Tolerance (iterations needed):")
tols = [1e-2, 1e-4, 1e-8, 1e-12]
for t in tols:
    _, ej = jacobi(A, b, tol=t, max_iter=5000)
    _, eg = gauss_seidel(A, b, tol=t, max_iter=5000)
    show(f"   tol = {t:7.0e}:  Jacobi {len(ej):4d},  Gauss-Seidel {len(eg):4d}")

show("\nb) Starting guess (tol = 1e-8):")
_, ej0 = jacobi(A, b)
_, ej60 = jacobi(A, b, x0=np.full(5, 60.0))
_, ejx = jacobi(A, b, x0=x_exact)
show(f"   all zeros: {len(ej0)} iterations,  all 60: {len(ej60)} iterations,  "
     f"exact answer: {len(ejx)} iteration(s)")

show("\nc) Different boundary temperatures (left = 200 C, right = 50 C):")
A2, b2 = line_system(5, 200.0, 50.0)
x2, e2 = gauss_seidel(A2, b2)
show(f"   Gauss-Seidel: {x2}   ({len(e2)} iterations)")

show("\nd) More grid points (tol = 1e-8):")
sizes = [5, 10, 20, 40]
it_j, it_gs = [], []
for n in sizes:
    An, bn = line_system(n, 100.0, 20.0)
    _, ej = jacobi(An, bn, max_iter=20000)
    _, eg = gauss_seidel(An, bn, max_iter=20000)
    it_j.append(len(ej)); it_gs.append(len(eg))
    show(f"   n = {n:3d}:  Jacobi {len(ej):5d},  Gauss-Seidel {len(eg):5d}")

show("\ne) A matrix that is NOT diagonally dominant: [[1, 2], [3, 1]], b = [3, 4]")
A_bad = np.array([[1.0, 2.0], [3.0, 1.0]])
b_bad = np.array([3.0, 4.0])
xb, eb_j = jacobi(A_bad, b_bad, max_iter=15)
_, eb_gs = gauss_seidel(A_bad, b_bad, max_iter=15)
show(f"   Diagonally dominant? {is_diagonally_dominant(A_bad)}")
show(f"   Jacobi change after 5, 10, 15 iterations: "
     f"{eb_j[4]:.1f}, {eb_j[9]:.1f}, {eb_j[14]:.1f}  -> DIVERGES")
show(f"   Direct solution works fine: {np.linalg.solve(A_bad, b_bad)}")
show()

# ---------------------------------------------------------------------------
# PART 3: Related real-world case - 2-D heated metal plate
# ---------------------------------------------------------------------------
header("PART 3: 2-D METAL PLATE (20 x 20 interior grid = 400 equations)")
m = 20
N = m * m
T1 = 2 * np.eye(m) - np.eye(m, k=1) - np.eye(m, k=-1)
I_m = np.eye(m)
A_plate = np.kron(I_m, T1) + np.kron(T1, I_m)          # 5-point Laplace stencil
top, left, right, bottom = 100.0, 20.0, 20.0, 20.0
grid_b = np.zeros((m, m))
grid_b[0, :] += top
grid_b[-1, :] += bottom
grid_b[:, 0] += left
grid_b[:, -1] += right
b_plate = grid_b.ravel()

zeros_share = 100 * np.mean(A_plate == 0)
show("Assumptions: steady state, square plate, edges held at "
     f"top = {top:.0f} C, other three edges = {left:.0f} C.")
show(f"Matrix size {N} x {N}; {zeros_share:.1f}% of the coefficients are zero.")
xp_j, ep_j = jacobi(A_plate, b_plate, tol=1e-6, max_iter=20000)
xp_gs, ep_gs = gauss_seidel(A_plate, b_plate, tol=1e-6, max_iter=20000)
xp_ex = np.linalg.solve(A_plate, b_plate)
show(f"Jacobi        : {len(ep_j):5d} iterations, max error vs direct = "
     f"{np.max(np.abs(xp_j - xp_ex)):.2e}")
show(f"Gauss-Seidel  : {len(ep_gs):5d} iterations, max error vs direct = "
     f"{np.max(np.abs(xp_gs - xp_ex)):.2e}")
plate = xp_gs.reshape(m, m)
show(f"Hottest interior point: {plate.max():.2f} C (next to the 100 C edge); "
     f"centre of plate: {plate[m // 2, m // 2]:.2f} C")
show()

# ---------------------------------------------------------------------------
# Save text output
# ---------------------------------------------------------------------------
output_dir = Path(__file__).resolve().parent.parent / "output"
output_dir.mkdir(exist_ok=True)
(output_dir / "task_08_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Output saved to: {output_dir / 'task_08_output.txt'}")

# ---------------------------------------------------------------------------
# ONE summary diagram
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 11.5))
gs = fig.add_gridspec(3, 3, height_ratios=[0.7, 1, 1], hspace=0.5, wspace=0.28)
fig.suptitle("Task 08: Iterative Methods (Jacobi and Gauss-Seidel) - Summary Diagram",
             fontsize=16, fontweight="bold")

# ---- Row 1: flowchart ----
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 100); ax.set_ylim(0, 20); ax.axis("off")
ax.set_title("Method: guess, improve, measure the change, repeat until it is below the tolerance",
             fontsize=12)
boxes = [
    (1,  "Start\ninitial guess x0\n(e.g. all zeros)",       "#dbe9f6"),
    (22, "Improve\nsolve each equation\nfor its own unknown", "#fde2c4"),
    (43, "Measure\nerror = max |x_new - x_old|",              "#fff3c4"),
    (64, "Test\nerror < tolerance ?\nno: repeat",             "#e3f2d9"),
    (84, "Stop\nreturn x and the\nerror history",             "#f6d9e8"),
]
for xpos, text, col in boxes:
    ax.add_patch(FancyBboxPatch((xpos, 4.5), 15, 11, boxstyle="round,pad=0.4",
                                fc=col, ec="black", lw=1.2))
    ax.text(xpos + 7.5, 10, text, ha="center", va="center", fontsize=8.5)
for x0 in (16.5, 37.5, 58.5, 79.5):
    ax.add_patch(FancyArrowPatch((x0 + 0.5, 10), (x0 + 5, 10), arrowstyle="-|>",
                                 mutation_scale=18, lw=1.5, color="black"))

# ---- Row 2, col 1: convergence ----
ax = fig.add_subplot(gs[1, 0])
ax.semilogy(range(1, len(err_j) + 1), err_j, label=f"Jacobi ({len(err_j)} iter.)")
ax.semilogy(range(1, len(err_gs) + 1), err_gs, label=f"Gauss-Seidel ({len(err_gs)} iter.)")
ax.axhline(1e-8, color="grey", ls="--", lw=1, label="tolerance 1e-8")
ax.set_xlabel("Iteration"); ax.set_ylabel("Maximum change")
ax.set_title("Part 1: convergence of the 5-point model")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

# ---- Row 2, col 2: temperatures ----
ax = fig.add_subplot(gs[1, 1])
pos = np.arange(1, 6)
ax.plot([0] + list(pos) + [6], [100] + list(x_exact) + [20], "k-", lw=2, label="Direct solution")
ax.plot(pos, x_j, "o", ms=9, label="Jacobi")
ax.plot(pos, x_gs, "x", ms=9, mew=2, label="Gauss-Seidel")
for p, v in zip(pos, x_exact):
    ax.annotate(f"{v:.1f}", (p, v), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8)
ax.set_xlabel("Grid point (0 and 6 are the boundaries)"); ax.set_ylabel("Temperature (C)")
ax.set_title("Part 1: temperatures agree with the direct solution")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ---- Row 2, col 3: iterations vs tolerance ----
ax = fig.add_subplot(gs[1, 2])
ij = [len(jacobi(A, b, tol=t, max_iter=5000)[1]) for t in tols]
ig = [len(gauss_seidel(A, b, tol=t, max_iter=5000)[1]) for t in tols]
xt = np.arange(len(tols)); w = 0.38
b1 = ax.bar(xt - w / 2, ij, w, label="Jacobi")
b2 = ax.bar(xt + w / 2, ig, w, label="Gauss-Seidel")
ax.bar_label(b1, fontsize=8); ax.bar_label(b2, fontsize=8)
ax.set_xticks(xt); ax.set_xticklabels([f"{t:.0e}" for t in tols])
ax.set_xlabel("Tolerance"); ax.set_ylabel("Iterations needed")
ax.set_title("Part 2: stricter tolerance needs more iterations")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

# ---- Row 3, col 1: iterations vs size ----
ax = fig.add_subplot(gs[2, 0])
ax.plot(sizes, it_j, "o-", label="Jacobi")
ax.plot(sizes, it_gs, "s-", label="Gauss-Seidel")
for s, v in zip(sizes, it_j):
    ax.annotate(str(v), (s, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
ax.set_xlabel("Number of grid points"); ax.set_ylabel("Iterations to reach 1e-8")
ax.set_title("Part 2: bigger grids converge more slowly")
ax.set_ylim(0, 6800)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ---- Row 3, col 2: divergence ----
ax = fig.add_subplot(gs[2, 1])
ax.semilogy(range(1, len(eb_j) + 1), eb_j, "o-", label="Jacobi")
ax.semilogy(range(1, len(eb_gs) + 1), eb_gs, "s-", label="Gauss-Seidel")
ax.set_xlabel("Iteration"); ax.set_ylabel("Maximum change")
ax.set_title("Part 2: NOT diagonally dominant -> diverges")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

# ---- Row 3, col 3: plate heatmap ----
ax = fig.add_subplot(gs[2, 2])
full = np.pad(plate, 1)
full[0, :], full[-1, :], full[:, 0], full[:, -1] = top, bottom, left, right
im = ax.imshow(full, cmap="inferno", origin="upper")
fig.colorbar(im, ax=ax, label="Temperature (C)", shrink=0.85)
ax.set_title(f"Part 3: 2-D plate, {N} equations\n"
             f"Gauss-Seidel {len(ep_gs)} iter. vs Jacobi {len(ep_j)} iter.", fontsize=10)
ax.set_xticks([]); ax.set_yticks([])

plot_file = output_dir / "task_08_diagram.png"
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