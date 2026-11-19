"""
Task 06 - Solution of Systems of Linear Equations
BIT 2118 Scientific Computing

Real-world problem: determine how many servers of each type are active,
given total CPU and memory usage.

    x = number of Type A servers      (4 CPU units,  8 GB memory each)
    y = number of Type B servers      (2 CPU units, 12 GB memory each)
    Observed totals: 20 CPU units and 72 GB memory

    4x +  2y = 20   (CPU)
    8x + 12y = 72   (Memory)

Just run this one file. It will:
  1. print the results in the terminal,
  2. save them to output/task_06_output.txt,
  3. draw ONE summary diagram (method flowchart, the lines for every variation,
     and a server-count comparison),
  4. save it to output/task_06_diagram.png and open it automatically.

Run from the repo root:  python code/Task_06_linear_systems.py
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

TOL = 1e-9   # tolerance for "is this a whole / non-negative number"

# ---------------------------------------------------------------------------
# Output handling: print every line and remember it for the output file
# ---------------------------------------------------------------------------
lines = []


def show(text=""):
    print(text)
    lines.append(str(text))


# ---------------------------------------------------------------------------
# Solver with checks
# ---------------------------------------------------------------------------
def solve_system(A, b):
    """Solve Ax = b safely and decide whether it makes sense as server counts."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    result = {"A": A, "b": b, "x": None, "residual_norm": None}

    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:          # singular matrix, det(A) = 0
        result["tag"] = "no unique solution"
        result["status"] = "NO UNIQUE SOLUTION (singular matrix, det(A) = 0)"
        return result

    result["x"] = x
    result["residual_norm"] = np.linalg.norm(b - A @ x)

    issues = []
    if np.any(x < -TOL):
        issues.append("negative")
    if np.any(np.abs(x - np.round(x)) > TOL):
        issues.append("not whole")

    if issues:
        result["tag"] = ", ".join(issues)
        result["status"] = ("SOLVED, BUT NOT REALISTIC "
                            f"({' and '.join(issues)} server count)")
    else:
        result["tag"] = "valid"
        result["status"] = "VALID (whole, non-negative server counts)"
    return result


# ---------------------------------------------------------------------------
# 1. Textbook example: 2x + 3y = 8 and 4x + y = 6
# ---------------------------------------------------------------------------
A_example = np.array([[2.0, 3.0], [4.0, 1.0]])
b_example = np.array([8.0, 6.0])
x_example = np.linalg.solve(A_example, b_example)

show("=" * 60)
show("EXAMPLE: 2x + 3y = 8  and  4x + y = 6")
show("=" * 60)
show(f"x = {x_example[0]:.0f}, y = {x_example[1]:.0f}")
show(f"Verification A @ x: {A_example @ x_example}")
show()

# ---------------------------------------------------------------------------
# 2. Real-world problem: server types
# ---------------------------------------------------------------------------
A = np.array([[4.0, 2.0], [8.0, 12.0]])   # rows: CPU, Memory; cols: Type A, Type B
b = np.array([20.0, 72.0])
solution = np.linalg.solve(A, b)
residual = b - A @ solution

show("=" * 60)
show("REAL-WORLD PROBLEM: ACTIVE SERVERS")
show("=" * 60)
show("Coefficient matrix A:")
show(A)
show(f"Right-hand side b: {b}")
show()
show(f"Type A servers: {solution[0]:.0f}")
show(f"Type B servers: {solution[1]:.0f}")
show(f"Verification A @ solution: {A @ solution}")
show(f"Residual: {residual}")
show(f"Residual norm: {np.linalg.norm(residual)}")
show()

# ---------------------------------------------------------------------------
# 3. Variations of the input values
# ---------------------------------------------------------------------------
scenarios = [
    ("1. Original",               [[4, 2], [8, 12]], [20, 72]),
    ("2. Different coefficients", [[3, 1], [2, 5]],  [14, 24]),
    ("3. Fractional answer",      [[4, 2], [8, 12]], [21, 70]),
    ("4. Negative answer",        [[4, 2], [8, 12]], [2, -8]),
    ("5. Singular (parallel)",    [[4, 2], [8, 4]],  [20, 50]),
]

show("=" * 60)
show("VARIATIONS OF THE INPUT VALUES")
show("=" * 60)

results = []
for name, A_s, b_s in scenarios:
    res = solve_system(A_s, b_s)
    res["name"] = name
    results.append(res)

    show(f"--- {name} ---")
    show(f"A = {res['A'].tolist()}   b = {res['b'].tolist()}")
    if res["x"] is not None:
        show(f"Type A servers: {res['x'][0]:.4f}")
        show(f"Type B servers: {res['x'][1]:.4f}")
        show(f"Residual norm: {res['residual_norm']:.2e}")
    show(f"Status: {res['status']}")
    show()

# ---------------------------------------------------------------------------
# 4. Save the text output (output folder is next to the code folder)
# ---------------------------------------------------------------------------
output_dir = Path(__file__).resolve().parent.parent / "output"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "task_06_output.txt"
output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Output saved to: {output_file}")

# ---------------------------------------------------------------------------
# 5. ONE summary diagram
# ---------------------------------------------------------------------------
STATUS_COLOUR = {"valid": "#2e8b57", "not whole": "#e6a817",
                 "negative": "#c0392b", "negative, not whole": "#c0392b",
                 "no unique solution": "#7f8c8d"}

fig = plt.figure(figsize=(16, 11.5))
gs = fig.add_gridspec(3, 3, height_ratios=[0.7, 1, 1], hspace=0.55, wspace=0.3)
fig.suptitle("Task 06: Solving Systems of Linear Equations (Ax = b) - Summary Diagram",
             fontsize=16, fontweight="bold")

# ---- Row 1: method flowchart ----
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 100); ax.set_ylim(0, 20); ax.axis("off")
ax.set_title("Method: write the problem as Ax = b, solve, then check the answer", fontsize=12)
boxes = [
    (1,  "Inputs\nA = [[4, 2], [8, 12]]\nb = [20, 72]",        "#dbe9f6"),
    (22, "Solve\nnp.linalg.solve(A, b)",                         "#fde2c4"),
    (43, "Check\ndet(A) = 0 ?\nwhole & non-negative ?",          "#fff3c4"),
    (64, "Verify\nresidual = b - A @ x\n(should be ~ 0)",         "#e3f2d9"),
    (84, "Output\nType A = 3\nType B = 4",                       "#f6d9e8"),
]
for xpos, text, col in boxes:
    ax.add_patch(FancyBboxPatch((xpos, 4.5), 15, 11, boxstyle="round,pad=0.4",
                                fc=col, ec="black", lw=1.2))
    ax.text(xpos + 7.5, 10, text, ha="center", va="center", fontsize=8.5)
for x0 in (16.5, 37.5, 58.5, 79.5):
    ax.add_patch(FancyArrowPatch((x0 + 0.5, 10), (x0 + 5, 10), arrowstyle="-|>",
                                 mutation_scale=18, lw=1.5, color="black"))

# ---- Rows 2-3: one panel per variation (lines cross at the solution) ----
xs = np.linspace(-1, 10, 200)
styles = ["-", "--"]
row_names = ["CPU", "Memory"]
slots = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]

for (r, c), res in zip(slots, results):
    ax = fig.add_subplot(gs[r, c])
    for i in range(2):
        a1, a2 = res["A"][i]
        cst = res["b"][i]
        label = f"{row_names[i]}: {a1:g}x + {a2:g}y = {cst:g}"
        if abs(a2) > 1e-12:
            ax.plot(xs, (cst - a1 * xs) / a2, styles[i], linewidth=2, label=label)
        else:
            ax.axvline(cst / a1, linestyle=styles[i], linewidth=2, label=label)
    if res["x"] is not None:
        px, py = res["x"]
        ax.plot(px, py, "ko", markersize=8)
        ax.annotate(f"({px:.2f}, {py:.2f})", (px, py),
                    textcoords="offset points", xytext=(8, 8), fontsize=9)
    else:
        ax.text(0.5, 0.08, "lines never meet: no solution",
                transform=ax.transAxes, ha="center", fontsize=9)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlim(-1, 10); ax.set_ylim(-5, 12)
    ax.set_xlabel("Type A servers (x)"); ax.set_ylabel("Type B servers (y)")
    ax.set_title(f"{res['name']}\n[{res['tag']}]", fontsize=10,
                 color=STATUS_COLOUR.get(res["tag"], "black"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")

# ---- Last panel: server counts per variation ----
ax = fig.add_subplot(gs[2, 2])
labels = [f"{i + 1}" for i in range(len(results))]
a_vals = [res["x"][0] if res["x"] is not None else 0 for res in results]
b_vals = [res["x"][1] if res["x"] is not None else 0 for res in results]
w = 0.38
bars_a = ax.bar(np.arange(len(results)) - w / 2, a_vals, w, label="Type A servers")
bars_b = ax.bar(np.arange(len(results)) + w / 2, b_vals, w, label="Type B servers")
ax.bar_label(bars_a, fmt="%.2f", fontsize=7)
ax.bar_label(bars_b, fmt="%.2f", fontsize=7)
for i, res in enumerate(results):
    if res["x"] is None:
        ax.text(i, 0.3, "none", ha="center", fontsize=8, color="#7f8c8d")
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_xticks(range(len(results))); ax.set_xticklabels(labels)
ax.set_xlabel("Variation number")
ax.set_ylabel("Number of servers")
ax.set_title("Server counts for each variation")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

plot_file = output_dir / "task_06_diagram.png"
fig.savefig(plot_file, dpi=150, bbox_inches="tight")
print(f"Diagram saved to: {plot_file}")

# ---- open the diagram automatically ----
try:
    if sys.platform.startswith("win"):
        os.startfile(str(plot_file))
    elif sys.platform == "darwin":
        subprocess.run(["open", str(plot_file)], check=False)
    else:
        subprocess.run(["xdg-open", str(plot_file)], check=False,
                       stderr=subprocess.DEVNULL)
except Exception:
    print("Could not open the image automatically - open it from the output/ folder.")