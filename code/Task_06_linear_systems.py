"""
Task 06 - Solution of Systems of Linear Equations
BIT 2118 Scientific Computing

Real-world problem: determine how many servers of each type are active,
given total CPU and memory usage.

    x = number of Type A servers
    y = number of Type B servers

    Type A uses 4 CPU units and  8 GB memory
    Type B uses 2 CPU units and 12 GB memory
    Total observed usage: 20 CPU units and 72 GB memory

    4x +  2y = 20   (CPU)
    8x + 12y = 72   (Memory)

Written as Ax = b and solved with np.linalg.solve().

The program:
  1. Solves the textbook example and the main server problem.
  2. Tests variations of the input values, handling singular systems and
     flagging answers that are not realistic (fractional or negative servers).
  3. Displays everything on screen and saves it to output/task_06_output.txt
  4. Plots every variation and saves it to output/linear_system_plot.png
"""

from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:          # the maths still runs without matplotlib
    HAVE_MPL = False

SHOW_PLOT = True             # set to False to only save the figure
TOL = 1e-9                   # tolerance for "is this a whole / non-negative number"

# ---------------------------------------------------------------------------
# Output handling: collect every line, print it, and save it at the end
# ---------------------------------------------------------------------------
lines = []


def show(text=""):
    """Print a line on screen and remember it for the output file."""
    print(text)
    lines.append(str(text))


# ---------------------------------------------------------------------------
# Solver with checks
# ---------------------------------------------------------------------------
def solve_system(A, b):
    """
    Solve Ax = b safely and decide whether the answer makes sense as a
    number of servers.

    Returns a dict with: A, b, x, residual_norm, tag, status
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    result = {"A": A, "b": b, "x": None, "residual_norm": None}

    # Singular matrix (det = 0): no unique solution, solve() would raise an error
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        result["tag"] = "no unique solution"
        result["status"] = "NO UNIQUE SOLUTION (singular matrix, det(A) = 0)"
        return result

    result["x"] = x
    result["residual_norm"] = np.linalg.norm(b - A @ x)

    # Servers must be whole, non-negative numbers
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
# 1. Small textbook example: 2x + 3y = 8 and 4x + y = 6
# ---------------------------------------------------------------------------
A_example = np.array([
    [2.0, 3.0],
    [4.0, 1.0]
])
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
# Rows:    [CPU equation, Memory equation]
# Columns: [Type A servers, Type B servers]
A = np.array([
    [4.0,  2.0],
    [8.0, 12.0]
])

# Observed totals: 20 CPU units and 72 GB memory
b = np.array([20.0, 72.0])

# Solve Ax = b
solution = np.linalg.solve(A, b)

# Residual: how far A @ solution is from b (should be ~0)
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
# 3. Variations: how does the code behave for different input values?
# ---------------------------------------------------------------------------
scenarios = [
    ("1. Original",              [[4, 2], [8, 12]], [20, 72]),
    ("2. Different coefficients", [[3, 1], [2, 5]],  [14, 24]),
    ("3. Fractional answer",     [[4, 2], [8, 12]], [21, 70]),
    ("4. Negative answer",       [[4, 2], [8, 12]], [2, -8]),
    ("5. Singular (parallel)",   [[4, 2], [8, 4]],  [20, 50]),
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
# 4. Save the text output to the output folder (created if it does not exist)
# ---------------------------------------------------------------------------
# The script lives in code/, so the output folder is one level up.
output_dir = Path(__file__).resolve().parent.parent / "output"
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "task_06_output.txt"
output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Output saved to: {output_file}")

# ---------------------------------------------------------------------------
# 5. Visualise the variations
# ---------------------------------------------------------------------------
# Each equation is a straight line. The solution is where the lines cross.
if HAVE_MPL:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    xs = np.linspace(-1, 10, 200)
    styles = ["-", "--"]
    row_names = ["CPU", "Memory"]

    for ax, res in zip(axes, results):
        for i in range(2):
            a1, a2 = res["A"][i]
            c = res["b"][i]
            label = f"{row_names[i]}: {a1:g}x + {a2:g}y = {c:g}"
            if abs(a2) > 1e-12:
                ax.plot(xs, (c - a1 * xs) / a2, styles[i], linewidth=2, label=label)
            else:
                ax.axvline(c / a1, linestyle=styles[i], linewidth=2, label=label)

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
        ax.set_xlim(-1, 10)
        ax.set_ylim(-5, 12)
        ax.set_xlabel("Type A servers (x)")
        ax.set_ylabel("Type B servers (y)")
        ax.set_title(f"{res['name']}\n[{res['tag']}]", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    for ax in axes[len(results):]:       # hide unused panel(s)
        ax.axis("off")

    fig.suptitle("Solving Ax = b: where the CPU and Memory lines cross",
                 fontsize=13)
    fig.tight_layout()

    plot_file = output_dir / "linear_system_plot.png"
    fig.savefig(plot_file, dpi=150)
    print(f"Plot saved to: {plot_file}")

    if SHOW_PLOT:
        plt.show()
else:
        print("matplotlib not installed - skipping the plot "
          "(install it with: pip install matplotlib)")