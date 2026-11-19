"""
Task 10 - Root Finding: Bisection and Fixed-Point Methods
BIT 2118 Scientific Computing

Part 0: Concept check - f(x) = x^3 - x - 2 has a root between 1 and 2.
Part 1: Original problem - equilibrium temperature of a nonlinear heat balance.
Part 2: Changing the inputs (tolerance, bracket, starting guess, a bad
        rearrangement, a different heat source, an invalid bracket).
Part 3: Related real-world case - break-even points of a business.

Just run this one file. It will:
  1. print the results in the terminal,
  2. save them to output/task_10_output.txt,
  3. draw ONE summary diagram,
  4. save it to output/task_10_diagram.png and open it automatically.

Run from the repo root:  python code/Task_10_bisection_fixed_point.py
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
from scipy.optimize import root_scalar

lines = []


def show(text=""):
    print(text)
    lines.append(str(text))


def header(title):
    show("=" * 64)
    show(title)
    show("=" * 64)


# ---------------------------------------------------------------------------
# The two methods (written out so every step is visible)
# ---------------------------------------------------------------------------
def bisection(f, a, b, tol=1e-10, max_iter=200):
    """Halve [a, b] until the half-width is below tol. Returns root and history."""
    if f(a) * f(b) > 0:
        raise ValueError("f(a) and f(b) have the same sign: no root is guaranteed "
                         "in this interval.")
    mids, widths = [], []
    for _ in range(max_iter):
        mid = (a + b) / 2
        mids.append(mid)
        widths.append((b - a) / 2)
        if f(mid) == 0 or (b - a) / 2 < tol:
            break
        if f(a) * f(mid) < 0:
            b = mid
        else:
            a = mid
    return mids[-1], mids, widths


def fixed_point(g, x0, tol=1e-10, max_iter=100):
    """Repeat x_new = g(x) until the change is below tol (or give up)."""
    x = x0
    history = [x]
    for _ in range(max_iter):
        x_new = g(x)
        history.append(x_new)
        if not np.isfinite(x_new) or abs(x_new) > 1e12:
            break
        if abs(x_new - x) < tol:
            break
        x = x_new
    return history[-1], history


# ---------------------------------------------------------------------------
# PART 0: Concept check
# ---------------------------------------------------------------------------
header("PART 0: CONCEPT CHECK  f(x) = x^3 - x - 2")
f0 = lambda x: x ** 3 - x - 2
root0, mids0, _ = bisection(f0, 1, 2, tol=1e-6)
show(f"f(1) = {f0(1)}, f(2) = {f0(2)}  -> sign change, so a root lies in [1, 2]")
show(f"Bisection root: {root0:.6f}  after {len(mids0)} halvings   "
     f"(f(root) = {f0(root0):.1e})")
show("First midpoints: " + ", ".join(f"{m:.4f}" for m in mids0[:5]))
g_good = lambda x: np.cbrt(x + 2)          # x = (x + 2)^(1/3)
g_bad = lambda x: x ** 3 - 2               # x = x^3 - 2
r_good, h_good = fixed_point(g_good, 1.5)
r_bad, h_bad = fixed_point(g_bad, 1.5, max_iter=12)
show(f"Fixed point with g(x) = (x+2)^(1/3): converges to {r_good:.6f} in {len(h_good) - 1} steps")
show(f"Fixed point with g(x) = x^3 - 2    : diverges ({h_bad[-1]:.3g} after {len(h_bad) - 1} steps)")
show("Same equation, two rearrangements - only one converges.")
show()

# ---------------------------------------------------------------------------
# PART 1: Original problem - equilibrium temperature
# ---------------------------------------------------------------------------
def heat_balance(T, source=10.0):
    """f(T) = 0 means heat gain equals heat loss."""
    return T - 25 - source * np.exp(-T / 20)


def g(T, source=10.0):
    """Fixed-point form: T = 25 + source * exp(-T / 20)."""
    return 25 + source * np.exp(-T / 20)


header("PART 1: EQUILIBRIUM TEMPERATURE  f(T) = T - 25 - 10 exp(-T/20)")
show(f"f(20) = {heat_balance(20):.4f}, f(40) = {heat_balance(40):.4f}  -> sign change in [20, 40]")
T_bis, mids, widths = bisection(heat_balance, 20, 40, tol=1e-10)
sol = root_scalar(heat_balance, bracket=[20, 40], method="bisect", xtol=1e-10)
T_fp, hist = fixed_point(g, 30.0, tol=1e-10)
T_true = root_scalar(heat_balance, bracket=[20, 40], method="brentq", xtol=1e-14).root

show(f"Bisection (own code)     : {T_bis:.6f} C   ({len(mids)} iterations)")
show(f"Bisection (scipy)        : {sol.root:.6f} C   ({sol.iterations} iterations)")
show(f"Fixed-point iteration    : {T_fp:.6f} C   ({len(hist) - 1} iterations)")
show(f"Check f(root) = {heat_balance(T_bis):.2e}")
show(f"|g'(T)| at the root = {abs(-0.5 * np.exp(-T_true / 20)):.3f}  "
     "(< 1, so fixed-point iteration converges)")
show(f"Fixed point reaches the answer in {len(hist) - 1} steps vs {len(mids)} for bisection.")
show()

# ---------------------------------------------------------------------------
# PART 2: Changing the inputs
# ---------------------------------------------------------------------------
header("PART 2: CHANGING THE INPUTS")

show("a) Tolerance (iterations needed, start T = 30, bracket [20, 40]):")
tols = [1e-2, 1e-4, 1e-6, 1e-10, 1e-14]
it_bis, it_fp = [], []
for t in tols:
    _, mb, _ = bisection(heat_balance, 20, 40, tol=t)
    _, hf = fixed_point(g, 30.0, tol=t)
    it_bis.append(len(mb)); it_fp.append(len(hf) - 1)
    show(f"   tol = {t:7.0e}:  bisection {len(mb):3d},  fixed point {len(hf) - 1:3d}")

show("\nb) Wider bracket (tol = 1e-10):")
_, mb_wide, _ = bisection(heat_balance, 0, 100, tol=1e-10)
show(f"   [20, 40]: {len(mids)} iterations,  [0, 100]: {len(mb_wide)} iterations  "
     "(each doubling of the width costs about 1 more halving)")

show("\nc) Different starting guess for fixed point (tol = 1e-10):")
for x0 in (25.0, 30.0, 100.0):
    r, h = fixed_point(g, x0)
    show(f"   T0 = {x0:5.1f}: {r:.6f} C in {len(h) - 1} iterations")

show("\nd) Invalid bracket [30, 40] (no sign change):")
try:
    bisection(heat_balance, 30, 40)
except ValueError as err:
    show(f"   ValueError: {err}")

show("\ne) A different rearrangement: T = -20 ln((T - 25) / 10)")
g_bad2 = lambda T: -20 * np.log(np.maximum(T - 25, 1e-12) / 10)
r2, h2 = fixed_point(g_bad2, 30.0, max_iter=15)
show(f"   Values: " + ", ".join(f"{v:.2f}" for v in h2[:8]) + " ...")
show(f"   |g'(T)| at the root = {20 / (T_true - 25):.2f} > 1, so it does NOT converge, "
     "even though the equation is the same.")

show("\nf) Stronger heat source (10 -> 40):")
f40 = lambda T: heat_balance(T, source=40.0)
T40, m40, _ = bisection(f40, 20, 60, tol=1e-10)
T40fp, h40 = fixed_point(lambda T: g(T, 40.0), 30.0)
show(f"   Bisection: {T40:.4f} C ({len(m40)} it.),  fixed point: {T40fp:.4f} C "
     f"({len(h40) - 1} it.)")
show()

# ---------------------------------------------------------------------------
# PART 3: Related real-world case - break-even points
# ---------------------------------------------------------------------------
header("PART 3: BREAK-EVEN POINTS OF A BUSINESS")
show("Revenue R(q) = 80 q - 0.03 q^2   (price falls as volume grows)")
show("Cost    C(q) = 5000 + 20 q + 0.02 q^2")
show("Profit f(q) = R(q) - C(q) = 60 q - 0.05 q^2 - 5000;  break-even when f(q) = 0.")
profit = lambda q: 60 * q - 0.05 * q ** 2 - 5000
q1, _, _ = bisection(profit, 0, 500, tol=1e-8)
q2, _, _ = bisection(profit, 500, 1500, tol=1e-8)
disc = np.sqrt(60 ** 2 - 4 * 0.05 * 5000)
q1_exact, q2_exact = (60 - disc) / 0.1, (60 + disc) / 0.1
show(f"Bisection on [0, 500]    : q1 = {q1:.4f} units   (exact {q1_exact:.4f})")
show(f"Bisection on [500, 1500] : q2 = {q2:.4f} units   (exact {q2_exact:.4f})")
q_peak = 600.0
show(f"Profit is positive between {q1:.0f} and {q2:.0f} units; the best volume is "
     f"{q_peak:.0f} units with profit {profit(q_peak):.0f}.")
g_q = lambda q: (5000 + 0.05 * q ** 2) / 60
rq, hq = fixed_point(g_q, 200.0)
show(f"Fixed point q = (5000 + 0.05 q^2)/60 from q0 = 200: {rq:.4f} units "
     f"({len(hq) - 1} iterations) - finds only the LOWER root.")
rq2, hq2 = fixed_point(g_q, 1200.0, max_iter=15)
show(f"Starting at q0 = 1200 it runs away ({hq2[-1]:.3g}): the upper root is repelling "
     "(|g'(q2)| > 1), but bisection finds it easily.")
show()

# ---------------------------------------------------------------------------
# Save text output
# ---------------------------------------------------------------------------
output_dir = Path(__file__).resolve().parent.parent / "output"
output_dir.mkdir(exist_ok=True)
(output_dir / "task_10_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Output saved to: {output_dir / 'task_10_output.txt'}")

# ---------------------------------------------------------------------------
# ONE summary diagram
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 11.5))
gs = fig.add_gridspec(3, 3, height_ratios=[0.7, 1, 1], hspace=0.5, wspace=0.3)
fig.suptitle("Task 10: Root Finding (Bisection and Fixed-Point) - Summary Diagram",
             fontsize=16, fontweight="bold")

# ---- Row 1: flowchart ----
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 100); ax.set_ylim(0, 20); ax.axis("off")
ax.set_title("Method: bracket the root and halve it (bisection), or iterate x = g(x) (fixed point)",
             fontsize=12)
boxes = [
    (1,  "Define f(T)\nheat gain - heat loss\nroot: f(T) = 0",      "#dbe9f6"),
    (22, "Bracket\nfind a, b with\nf(a) f(b) < 0",                  "#fde2c4"),
    (43, "Bisection\nmid = (a+b)/2, keep the\nhalf with the sign change", "#fff3c4"),
    (64, "Stop\nhalf-width < tolerance\n(or |f| small)",            "#e3f2d9"),
    (84, "Compare\nfixed point T = g(T)\nsame answer?",             "#f6d9e8"),
]
for xpos, text, col in boxes:
    ax.add_patch(FancyBboxPatch((xpos, 4.5), 15, 11, boxstyle="round,pad=0.4",
                                fc=col, ec="black", lw=1.2))
    ax.text(xpos + 7.5, 10, text, ha="center", va="center", fontsize=8.2)
for x0 in (16.5, 37.5, 58.5, 79.5):
    ax.add_patch(FancyArrowPatch((x0 + 0.5, 10), (x0 + 5, 10), arrowstyle="-|>",
                                 mutation_scale=18, lw=1.5, color="black"))

# ---- Row 2, col 1: function with root ----
ax = fig.add_subplot(gs[1, 0])
temps = np.linspace(20, 40, 300)
ax.plot(temps, heat_balance(temps), lw=2, label="f(T)")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(T_bis, color="red", ls="--", label=f"root = {T_bis:.4f} C")
ax.plot(mids[:6], [heat_balance(m) for m in mids[:6]], "o", color="orange",
        label="first 6 midpoints")
for k, m in enumerate(mids[:4], start=1):
    ax.annotate(str(k), (m, heat_balance(m)), textcoords="offset points",
                xytext=(5, 6), fontsize=8)
ax.set_xlabel("Temperature (C)"); ax.set_ylabel("Heat-balance function f(T)")
ax.set_title("Part 1: the root and the bisection midpoints")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ---- Row 2, col 2: error vs iteration ----
ax = fig.add_subplot(gs[1, 1])
err_b = np.maximum(np.abs(np.array(mids) - T_true), 1e-16)
err_f = np.maximum(np.abs(np.array(hist) - T_true), 1e-16)
ax.semilogy(range(1, len(err_b) + 1), err_b, "o-", ms=3, label="Bisection")
ax.semilogy(range(len(err_f)), err_f, "s-", ms=3, label="Fixed point")
ax.axhline(1e-10, color="grey", ls="--", lw=1, label="tolerance 1e-10")
ax.set_xlabel("Iteration"); ax.set_ylabel("Error |T - root|")
ax.set_title("Part 1: error versus iteration")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

# ---- Row 2, col 3: cobweb ----
ax = fig.add_subplot(gs[1, 2])
tt = np.linspace(26, 36, 200)
ax.plot(tt, g(tt), lw=2, label="y = g(T)")
ax.plot(tt, tt, "k--", lw=1, label="y = T")
Tc = 35.0
for _ in range(7):
    Tn = g(Tc)
    ax.plot([Tc, Tc], [Tc, Tn], color="red", lw=1)
    ax.plot([Tc, Tn], [Tn, Tn], color="red", lw=1)
    Tc = Tn
ax.set_xlim(26, 36); ax.set_ylim(26, 36)
ax.set_xlabel("T"); ax.set_ylabel("g(T)")
ax.set_title("Part 1: fixed-point iteration spirals in (cobweb)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ---- Row 3, col 1: iterations vs tolerance ----
ax = fig.add_subplot(gs[2, 0])
xt = np.arange(len(tols)); w = 0.38
b1 = ax.bar(xt - w / 2, it_bis, w, label="Bisection")
b2 = ax.bar(xt + w / 2, it_fp, w, label="Fixed point")
ax.bar_label(b1, fontsize=8); ax.bar_label(b2, fontsize=8)
ax.set_xticks(xt); ax.set_xticklabels([f"{t:.0e}" for t in tols], fontsize=8)
ax.set_xlabel("Tolerance"); ax.set_ylabel("Iterations needed")
ax.set_title("Part 2: stricter tolerance needs more iterations")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

# ---- Row 3, col 2: bad rearrangement ----
ax = fig.add_subplot(gs[2, 1])
ax.plot(range(len(h2)), h2, "o-", color="#c0392b", label="T = -20 ln((T-25)/10)")
ax.plot(range(len(hist[:10])), hist[:10], "s-", color="#2e8b57", label="T = 25 + 10 exp(-T/20)")
ax.axhline(T_true, color="grey", ls="--", lw=1, label=f"true root {T_true:.2f}")
ax.set_xlabel("Iteration"); ax.set_ylabel("Estimate of T (C)")
ax.set_title("Part 2: same equation, different rearrangement")
ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

# ---- Row 3, col 3: break-even ----
ax = fig.add_subplot(gs[2, 2])
qq = np.linspace(0, 1250, 400)
ax.plot(qq, profit(qq), lw=2, label="profit f(q)")
ax.axhline(0, color="black", lw=0.8)
ax.fill_between(qq, 0, profit(qq), where=profit(qq) > 0, alpha=0.25, color="green",
                label="profit")
ax.fill_between(qq, 0, profit(qq), where=profit(qq) < 0, alpha=0.25, color="red",
                label="loss")
for q, lab in ((q1, "q1"), (q2, "q2")):
    ax.plot(q, 0, "ko")
    ax.annotate(f"{lab} = {q:.0f}", (q, 0), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9)
ax.set_xlabel("Units produced and sold (q)"); ax.set_ylabel("Profit")
ax.set_title("Part 3: two break-even points")
ax.legend(fontsize=8, loc="lower center"); ax.grid(alpha=0.3)

plot_file = output_dir / "task_10_diagram.png"
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