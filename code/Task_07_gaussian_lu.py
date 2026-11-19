"""
BIT 2118 Scientific Computing - Task 07: Gaussian Elimination and LU Decomposition

Just run this one file. It will:
  1. print the numerical results in the terminal,
  2. draw ONE summary diagram,
  3. save it to output/task_07_diagram.png,
  4. open the diagram automatically.

Run from the repo root:  python code/Task_07_gaussian_lu.py
"""
import os
import sys
import time
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.linalg import lu, lu_factor, lu_solve

os.makedirs("output", exist_ok=True)

# ------------------------- Original problem -------------------------
A = np.array([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 3.0]])
bs = [np.array([15.0, 10.0, 10.0]), np.array([16.0, 11.0, 9.0]), np.array([14.0, 12.0, 11.0])]
P, L, U = lu(A)
lu_piv, piv = lu_factor(A)                       # factor ONCE
sols = [lu_solve((lu_piv, piv), b) for b in bs]  # reuse for each b

# ------------------------- Changing an input ------------------------
x_b = lu_solve((lu_piv, piv), np.array([20.0, 10.0, 10.0]))
A2 = A.copy(); A2[2, 2] = 5.0
x_A = np.linalg.solve(A2, bs[0])

# ------------------------- Heated rod case --------------------------
n, k, T_end, dx = 5, 50.0, 25.0, 1.0 / 6
A_rod = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
q_cases = {"08:00 uniform": [5e4] * 5,
           "12:00 centre":  [1e4, 5e4, 1.5e5, 5e4, 1e4],
           "16:00 left":    [1.5e5, 8e4, 2e4, 0, 0]}
lu_r, piv_r = lu_factor(A_rod)
x_pos = np.linspace(0, 1, n + 2)

# ------------------------- Timing -----------------------------------
rng = np.random.default_rng(0)
N = 400
M = rng.random((N, N)) + N * np.eye(N)
counts = [1, 10, 50, 100, 200]
t_dir, t_lu = [], []
for m in counts:
    B = rng.random((m, N))
    t0 = time.perf_counter()
    for b in B:
        np.linalg.solve(M, b)
    t_dir.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    f, p = lu_factor(M)
    for b in B:
        lu_solve((f, p), b)
    t_lu.append(time.perf_counter() - t0)


# =========================== TERMINAL RESULTS =======================
print("=" * 60)
print("TASK 07 - GAUSSIAN ELIMINATION AND LU DECOMPOSITION")
print("=" * 60)
print("A =\n", A)
print("L =\n", np.round(L, 4))
print("U =\n", np.round(U, 4))
for i, (b, x) in enumerate(zip(bs, sols), start=1):
    print(f"Set {i}: b = {b}  ->  x = {np.round(x, 4)}  "
          f"(residual {np.linalg.norm(b - A @ x):.1e})")
print(f"Change b[1] to 20      -> x = {np.round(x_b, 4)}")
print(f"Change A[3,3] to 5     -> x = {np.round(x_A, 4)}")
print(f"Timing, {counts[-1]} vectors: direct {t_dir[-1]:.3f} s vs LU {t_lu[-1]:.3f} s")

# =========================== DIAGRAM ================================
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 3, height_ratios=[0.9, 1.1, 1.1], hspace=0.5, wspace=0.3)
fig.suptitle("Task 07: Gaussian Elimination and LU Decomposition - Summary Diagram",
             fontsize=16, fontweight="bold")

# ---- Row 1: flow diagram of the method ----
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 100); ax.set_ylim(0, 20); ax.axis("off")
ax.set_title("Method: factor A once, reuse for every new b", fontsize=12)
boxes = [
    (1,  "Inputs\nA (3x3), b1, b2, b3",       "#dbe9f6"),
    (22, "Factor once\nPA = LU\n(lu_factor)",  "#fde2c4"),
    (43, "Forward substitution\nLy = Pb",       "#e3f2d9"),
    (64, "Back substitution\nUx = y",           "#e3f2d9"),
    (84, "Output\nx for each b\nresidual ~ 1e-15", "#f6d9e8"),
]
for xpos, text, col in boxes:
    ax.add_patch(FancyBboxPatch((xpos, 5), 15, 10, boxstyle="round,pad=0.4",
                                fc=col, ec="black", lw=1.2))
    ax.text(xpos + 7.5, 10, text, ha="center", va="center", fontsize=9)
for x0 in (16.5, 37.5, 58.5, 79.5):
    ax.add_patch(FancyArrowPatch((x0 + 0.5, 10), (x0 + 5, 10), arrowstyle="-|>",
                                 mutation_scale=18, lw=1.5, color="black"))
ax.annotate("new b every second: only the two cheap substitution steps repeat",
            xy=(57, 4.7), xytext=(30, 0.8), fontsize=9, ha="left",
            arrowprops=dict(arrowstyle="->", color="gray"))

# ---- Row 2, col 1: L and U ----
ax = fig.add_subplot(gs[1, 0])
LU_show = np.hstack([L, np.full((3, 1), np.nan), U])
im = ax.imshow(LU_show, cmap="coolwarm", vmin=-4, vmax=4)
for r in range(3):
    for c in range(7):
        if c == 3:
            continue
        v = LU_show[r, c]
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8)
ax.set_xticks([1, 5]); ax.set_xticklabels(["L (lower)", "U (upper)"])
ax.set_yticks([])
ax.set_title("Factors of A (P = identity)")

# ---- Row 2, col 2: solutions per set ----
ax = fig.add_subplot(gs[1, 1])
w = 0.25
for j, lab in enumerate(["x1", "x2", "x3"]):
    bars = ax.bar(np.arange(3) + (j - 1) * w, [s[j] for s in sols], w, label=lab)
    ax.bar_label(bars, fmt="%.2f", fontsize=7)
ax.set_xticks(range(3)); ax.set_xticklabels(["Set 1", "Set 2", "Set 3"])
ax.set_ylim(0, 6.8); ax.set_ylabel("Solution value")
ax.set_title("Solutions (same L, U reused)")
ax.legend(fontsize=8, ncol=3, loc="upper center"); ax.grid(axis="y", alpha=0.3)

# ---- Row 2, col 3: changing an input ----
ax = fig.add_subplot(gs[1, 2])
for j, (lab, x) in enumerate([("Original", sols[0]), ("b[1] = 20", x_b), ("A[3,3] = 5", x_A)]):
    bars = ax.bar(np.arange(3) + (j - 1) * w, x, w, label=lab)
    ax.bar_label(bars, fmt="%.2f", fontsize=7)
ax.set_xticks(range(3)); ax.set_xticklabels(["x1", "x2", "x3"])
ax.set_ylim(0, 8); ax.set_ylabel("Solution value")
ax.set_title("Effect of changing an input")
ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="y", alpha=0.3)

# ---- Row 3, cols 1-2: heated rod ----
ax = fig.add_subplot(gs[2, 0:2])
for name, q in q_cases.items():
    b = dx ** 2 * np.array(q) / k
    b[0] += T_end; b[-1] += T_end
    T = lu_solve((lu_r, piv_r), b)
    ax.plot(x_pos, np.concatenate(([T_end], T, [T_end])), marker="o", label=name)
    ax.annotate(f"{T.max():.0f} C", (x_pos[T.argmax() + 1], T.max()),
                textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax.set_ylim(0, 240)
ax.set_xlabel("Position along rod (m)"); ax.set_ylabel("Temperature (C)")
ax.set_title("Modified case: heated steel rod (ends held at 25 C)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ---- Row 3, col 3: timing ----
ax = fig.add_subplot(gs[2, 2])
ax.plot(counts, t_dir, marker="o", label="Re-solve each time")
ax.plot(counts, t_lu, marker="s", label="Factor once (LU)")
ax.set_xlabel("Number of vectors b"); ax.set_ylabel("Total time (s)")
ax.set_title(f"Why LU helps ({N}x{N} system)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

path = os.path.join("output", "task_07_diagram.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
print(f"\nDiagram saved to {path}")

# ---- open the diagram automatically ----
try:
    if sys.platform.startswith("win"):
        os.startfile(os.path.abspath(path))
    elif sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)
except Exception:
    print("Could not open the image automatically - open it from the output/ folder.")