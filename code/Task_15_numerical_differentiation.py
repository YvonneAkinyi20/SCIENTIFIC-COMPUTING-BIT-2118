"""
Task 15 - Numerical Differentiation
BIT 2118 Scientific Computing

Part 0: Concept check - forward and central differences on known functions.
Part 1: Original problem - vehicle speed from GPS position data.
Part 2: Changing the inputs (step size and round-off, noisy data, wider spacing,
        better formulas at the ends of the data).
Part 3: Related real-world case - speed and deceleration of a braking car.

Just run this one file. It will:
  1. print the results in the terminal,
  2. save them to output/task_15_output.txt,
  3. draw ONE summary diagram,
  4. save it to output/task_15_diagram.png and open it automatically.

Run from the repo root:  python code/Task_15_numerical_differentiation.py
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


def forward_diff(f, x, h):
    return (f(x + h) - f(x)) / h


def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def gps_speed(time, position):
    """Forward difference at the start, central inside, backward at the end."""
    speed = np.zeros_like(position)
    speed[0] = (position[1] - position[0]) / (time[1] - time[0])
    for i in range(1, len(time) - 1):
        speed[i] = (position[i + 1] - position[i - 1]) / (time[i + 1] - time[i - 1])
    speed[-1] = (position[-1] - position[-2]) / (time[-1] - time[-2])
    return speed


# ---------------------------------------------------------------------------
# PART 0: Concept check
# ---------------------------------------------------------------------------
header("PART 0: CONCEPT CHECK")
show(f"Average rate of change: (112 - 100) / 2 = {(112 - 100) / 2:.0f} m/s")
f0 = lambda x: x ** 2
show("\nf(x) = x^2 at x = 1 (exact derivative 2):")
show(f"   {'h':>8s} {'forward':>10s} {'error':>10s} {'central':>10s} {'error':>10s}")
for h in (1.0, 0.1, 0.01, 0.001):
    fw, ce = forward_diff(f0, 1.0, h), central_diff(f0, 1.0, h)
    show(f"   {h:8.3f} {fw:10.5f} {abs(fw - 2):10.2e} {ce:10.5f} {abs(ce - 2):10.2e}")
show("   The central difference is exact for x^2, while the forward error is h.")
show()

# ---------------------------------------------------------------------------
# PART 1: Original problem - GPS speed
# ---------------------------------------------------------------------------
time = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
position = np.array([0.0, 4.8, 10.2, 16.1, 22.7, 30.0, 38.0])
speed = gps_speed(time, position)
speed_np = np.gradient(position, time)                    # same formulas as above
speed_np2 = np.gradient(position, time, edge_order=2)     # second-order at the ends

# Smooth reference: derivative of a quadratic fitted to the positions
quad = np.polyfit(time, position, 2)
ref_speed = np.polyval(np.polyder(quad), time)
ref_acc = 2 * quad[0]

header("PART 1: VEHICLE SPEED FROM GPS POSITIONS")
show(f"Quadratic trend of the data: x(t) = {quad[0]:.3f} t^2 {quad[1]:+.3f} t {quad[2]:+.3f}")
show(f"{'t (s)':>6s} {'position (m)':>13s} {'speed (m/s)':>12s} {'method':>10s} {'smooth ref':>11s}")
labels = ["forward"] + ["central"] * 5 + ["backward"]
for t, x, v, lab, r in zip(time, position, speed, labels, ref_speed):
    show(f"{t:6.0f} {x:13.1f} {v:12.2f} {lab:>10s} {r:11.2f}")
show(f"Same result as np.gradient: {np.allclose(speed, speed_np)}")
show(f"Speed rises from {speed[0]:.1f} to {speed[-1]:.1f} m/s; the acceleration is about "
     f"{2 * quad[0]:.2f} m/s^2.")
show(f"Speed at 3 s: {speed[3]:.2f} m/s = {speed[3] * 3.6:.1f} km/h")
show()

# ---------------------------------------------------------------------------
# PART 2: Changing the inputs
# ---------------------------------------------------------------------------
header("PART 2: CHANGING THE INPUTS")

show("a) Step size h for f(x) = sin(x) at x = 1 (exact derivative cos(1) = %.10f):" % np.cos(1.0))
exact = np.cos(1.0)
hs = np.logspace(-1, -15, 57)
err_f = np.abs(forward_diff(np.sin, 1.0, hs) - exact)
err_c = np.abs(central_diff(np.sin, 1.0, hs) - exact)
show(f"   {'h':>8s} {'forward error':>15s} {'central error':>15s}")
for h in (1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-15):
    show(f"   {h:8.0e} {abs(forward_diff(np.sin, 1.0, h) - exact):15.2e} "
         f"{abs(central_diff(np.sin, 1.0, h) - exact):15.2e}")
show(f"   Best forward error {err_f.min():.1e} at h = {hs[err_f.argmin()]:.0e};  "
     f"best central error {err_c.min():.1e} at h = {hs[err_c.argmin()]:.0e}.")
show("   Making h smaller helps at first (truncation error falls), then hurts "
     "(round-off error grows).")

show("\nb) Noisy GPS positions (noise of about +/-0.3 m added):")
noise = np.array([0.30, 0.20, -0.30, -0.20, 0.30, 0.25, -0.30])
pos_noisy = position + noise
sp1 = np.array([(pos_noisy[i + 1] - pos_noisy[i - 1]) / 2.0 for i in range(1, 6)])   # i = 1..5
sp2 = np.array([(pos_noisy[i + 2] - pos_noisy[i - 2]) / 4.0 for i in range(2, 5)])   # i = 2..4
clean1 = np.array([(position[i + 1] - position[i - 1]) / 2.0 for i in range(1, 6)])
err1 = sp1[1:4] - ref_speed[2:5]
err2 = sp2 - ref_speed[2:5]
show(f"   Clean data, central difference over 2 s : speeds {np.round(clean1, 2)}")
show(f"   Noisy data, central difference over 2 s : speeds {np.round(sp1, 2)}")
show(f"   Noisy data, central difference over 4 s : speeds {np.round(sp2, 2)} (t = 2, 3, 4 s)")
show(f"   RMS error vs smooth reference at t = 2, 3, 4 s:  2 s span {np.sqrt(np.mean(err1 ** 2)):.2f} m/s,"
     f"  4 s span {np.sqrt(np.mean(err2 ** 2)):.2f} m/s")
show("   Differentiation amplifies noise; a wider span averages it down (the data is "
     "nearly a parabola, so the central formula stays accurate).")

show("\nc) Fewer samples (use every 2nd GPS point, so the interval is 2 s):")
t2, x2 = time[::2], position[::2]
sp_sparse = gps_speed(t2, x2)
for t, v in zip(t2, sp_sparse):
    show(f"   t = {t:.0f} s: {v:.2f} m/s   (smooth reference {np.polyval(np.polyder(quad), t):.2f})")

show("\nd) Better formulas at the ends of the data (second-order one-sided):")
first_order_0 = (position[1] - position[0]) / 1.0
second_order_0 = (-3 * position[0] + 4 * position[1] - position[2]) / 2.0
first_order_n = (position[-1] - position[-2]) / 1.0
second_order_n = (3 * position[-1] - 4 * position[-2] + position[-3]) / 2.0
show(f"   t = 0 s: forward {first_order_0:.2f}, second-order {second_order_0:.2f}, reference {ref_speed[0]:.2f}")
show(f"   t = 6 s: backward {first_order_n:.2f}, second-order {second_order_n:.2f}, reference {ref_speed[-1]:.2f}")
show("   The one-sided first-order formula gives the speed at the midpoint of the interval, "
     "so it lags at the ends; the second-order formula is closer.")
show()

# ---------------------------------------------------------------------------
# PART 3: Related real-world case - a braking car
# ---------------------------------------------------------------------------
header("PART 3: SPEED AND DECELERATION OF A BRAKING CAR")
tb = np.arange(0.0, 4.01, 0.5)
exact_pos = 20 * tb - 2 * tb ** 2          # starts at 20 m/s, decelerates at 4 m/s^2
noise_b = np.array([0.02, -0.03, 0.04, -0.02, 0.03, -0.04, 0.02, -0.03, 0.01])
pos_b = exact_pos + noise_b
hb = 0.5
speed_b = np.array([(pos_b[i + 1] - pos_b[i - 1]) / (2 * hb) for i in range(1, len(tb) - 1)])
acc_b = np.array([(pos_b[i + 1] - 2 * pos_b[i] + pos_b[i - 1]) / hb ** 2 for i in range(1, len(tb) - 1)])
true_speed_b = 20 - 4 * tb[1:-1]
show("Assumptions: GPS every 0.5 s, small errors (+/-0.04 m); true motion starts at 20 m/s and "
     "slows down at 4 m/s^2.")
show(f"{'t (s)':>6s} {'position':>9s} {'speed est.':>11s} {'true speed':>11s} {'accel. est.':>12s}")
for t, x, v, tv, a_ in zip(tb[1:-1], pos_b[1:-1], speed_b, true_speed_b, acc_b):
    show(f"{t:6.1f} {x:9.2f} {v:11.2f} {tv:11.2f} {a_:12.2f}")
show(f"Speed error: RMS {np.sqrt(np.mean((speed_b - true_speed_b) ** 2)):.3f} m/s "
     f"(first derivative is accurate).")
show(f"Acceleration estimates: mean {acc_b.mean():.2f} m/s^2, range {acc_b.min():.2f} to "
     f"{acc_b.max():.2f}  (true value -4.00)")
show("The second derivative is much more sensitive to noise (it divides errors by h^2 = 0.25), "
     "so the average is reliable but single values are not.")
slope = np.polyfit(tb[1:-1], speed_b, 1)
t_stop = -slope[1] / slope[0]
show(f"Straight line through the speed estimates: v = {slope[0]:.2f} t {slope[1]:+.2f}  ->  "
     f"deceleration {abs(slope[0]):.2f} m/s^2, stops at about {t_stop:.1f} s")
show(f"Stopping distance from 20 m/s at {abs(slope[0]):.2f} m/s^2: "
     f"{20 ** 2 / (2 * abs(slope[0])):.1f} m (true value 50 m)")
show()

# ---------------------------------------------------------------------------
# Save text output
# ---------------------------------------------------------------------------
output_dir = Path(__file__).resolve().parent.parent / "output"
output_dir.mkdir(exist_ok=True)
(output_dir / "task_15_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Output saved to: {output_dir / 'task_15_output.txt'}")

# ---------------------------------------------------------------------------
# ONE summary diagram
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 11.5))
gs = fig.add_gridspec(3, 3, height_ratios=[0.7, 1, 1], hspace=0.5, wspace=0.3)
fig.suptitle("Task 15: Numerical Differentiation - Summary Diagram",
             fontsize=16, fontweight="bold")

# ---- Row 1: flowchart ----
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 100); ax.set_ylim(0, 20); ax.axis("off")
ax.set_title("Method: speed = change in position / change in time, using neighbouring GPS points",
             fontsize=12)
boxes = [
    (1,  "GPS data\ntime (s) and\nposition (m)",                  "#dbe9f6"),
    (22, "Start: forward\n(x1 - x0) / dt",                          "#fde2c4"),
    (43, "Inside: central\n(x[i+1] - x[i-1]) / (2 dt)",             "#fff3c4"),
    (64, "End: backward\n(x[n] - x[n-1]) / dt",                     "#e3f2d9"),
    (84, "Speed (m/s)\ncompare step\nsizes and errors",             "#f6d9e8"),
]
for xpos, text, col in boxes:
    ax.add_patch(FancyBboxPatch((xpos, 4.5), 15, 11, boxstyle="round,pad=0.4",
                                fc=col, ec="black", lw=1.2))
    ax.text(xpos + 7.5, 10, text, ha="center", va="center", fontsize=8.2)
for x0 in (16.5, 37.5, 58.5, 79.5):
    ax.add_patch(FancyArrowPatch((x0 + 0.5, 10), (x0 + 5, 10), arrowstyle="-|>",
                                 mutation_scale=18, lw=1.5, color="black"))

# ---- Row 2, col 1: position and speed ----
ax = fig.add_subplot(gs[1, 0])
ax.plot(time, position, "ko-", label="Position (m)")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Position (m)")
ax2 = ax.twinx()
colors = {"forward": "#ff7f0e", "central": "#1f77b4", "backward": "#2ca02c"}
for t, v, lab in zip(time, speed, labels):
    ax2.plot(t, v, "s", color=colors[lab], ms=8)
ax2.plot(time, ref_speed, "--", color="grey", lw=1.5, label="smooth reference")
ax2.plot([], [], "s", color=colors["forward"], label="forward")
ax2.plot([], [], "s", color=colors["central"], label="central")
ax2.plot([], [], "s", color=colors["backward"], label="backward")
ax2.set_ylabel("Estimated speed (m/s)"); ax2.set_ylim(0, 12)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left")
ax.set_title("Part 1: GPS position and estimated speed", fontsize=10)
ax.grid(alpha=0.3)

# ---- Row 2, col 2: geometry of the three formulas ----
ax = fig.add_subplot(gs[1, 1])
xg = np.linspace(0, 2.2, 200)
fg = lambda x: x ** 2
x0g, hg = 1.0, 0.7
ax.plot(xg, fg(xg), "k", lw=2, label="f(x) = x^2")
ax.plot(xg, 2 * x0g * (xg - x0g) + fg(x0g), ":", color="black", lw=1.5, label="true tangent (slope 2)")
ax.plot([x0g, x0g + hg], [fg(x0g), fg(x0g + hg)], "-", color="#ff7f0e", lw=2,
        label=f"forward: slope {forward_diff(fg, x0g, hg):.2f}")
ax.plot([x0g - hg, x0g], [fg(x0g - hg), fg(x0g)], "-", color="#2ca02c", lw=2,
        label=f"backward: slope {(fg(x0g) - fg(x0g - hg)) / hg:.2f}")
ax.plot([x0g - hg, x0g + hg], [fg(x0g - hg), fg(x0g + hg)], "-", color="#1f77b4", lw=2,
        label=f"central: slope {central_diff(fg, x0g, hg):.2f}")
ax.plot([x0g - hg, x0g, x0g + hg], [fg(x0g - hg), fg(x0g), fg(x0g + hg)], "ko")
ax.set_xlabel("x"); ax.set_ylabel("f(x)")
ax.set_title("Part 0: the three formulas (central is closest to the tangent)", fontsize=10)
ax.legend(fontsize=7.5, loc="upper left"); ax.grid(alpha=0.3)

# ---- Row 2, col 3: error vs h ----
ax = fig.add_subplot(gs[1, 2])
ax.loglog(hs, err_f, "o-", ms=3, color="#ff7f0e", label="forward difference")
ax.loglog(hs, np.maximum(err_c, 1e-17), "s-", ms=3, color="#1f77b4", label="central difference")
ax.loglog(hs, hs * 0.4, "--", color="#ff7f0e", lw=1, alpha=0.6, label="slope 1 (error ~ h)")
ax.loglog(hs, hs ** 2 * 0.1, "--", color="#1f77b4", lw=1, alpha=0.6, label="slope 2 (error ~ h^2)")
ax.set_xlabel("Step size h"); ax.set_ylabel("Absolute error")
ax.invert_xaxis()
ax.set_ylim(1e-12, 1)
ax.set_title("Part 2: error vs step size (round-off beats truncation at tiny h)", fontsize=9)
ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

# ---- Row 3, col 1: noise ----
ax = fig.add_subplot(gs[2, 0])
ax.plot(time[1:6], clean1, "ko-", label="clean data, 2 s span")
ax.plot(time[1:6], sp1, "s--", color="#d62728", label="noisy data, 2 s span")
ax.plot(time[2:5], sp2, "^-", color="#2ca02c", label="noisy data, 4 s span")
ax.plot(time, ref_speed, ":", color="grey", label="smooth reference")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Estimated speed (m/s)")
ax.set_title("Part 2: noise is amplified; a wider span smooths it", fontsize=10)
ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

# ---- Row 3, col 2: edge formulas ----
ax = fig.add_subplot(gs[2, 1])
labels_e = ["t = 0 s", "t = 6 s"]
ref_e = np.array([ref_speed[0], ref_speed[-1]])
fo = np.array([first_order_0, first_order_n]); so = np.array([second_order_0, second_order_n])
xe = np.arange(2); w = 0.25
b0 = ax.bar(xe - w, ref_e, w, color="grey", label="smooth reference")
b1 = ax.bar(xe, fo, w, color="#ff7f0e", label="first-order (forward/backward)")
b2 = ax.bar(xe + w, so, w, color="#1f77b4", label="second-order one-sided")
for bars in (b0, b1, b2):
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
ax.set_xticks(xe); ax.set_xticklabels(labels_e)
ax.set_ylabel("Speed (m/s)"); ax.set_ylim(0, 10.5)
ax.set_title("Part 2: better formulas at the ends of the data", fontsize=10)
ax.legend(fontsize=7.5, loc="upper left"); ax.grid(axis="y", alpha=0.3)

# ---- Row 3, col 3: braking car ----
ax = fig.add_subplot(gs[2, 2])
ax.plot(tb[1:-1], true_speed_b, "k-", lw=1.5, label="true speed")
ax.plot(tb[1:-1], speed_b, "o", color="#1f77b4", label="estimated speed (central)")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed (m/s)")
ax3 = ax.twinx()
ax3.plot(tb[1:-1], acc_b, "s--", color="#d62728", label="estimated acceleration")
ax3.axhline(-4, color="#d62728", ls=":", lw=1)
ax3.set_ylabel("Acceleration (m/s$^2$)", color="#d62728"); ax3.set_ylim(-8, 0)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="lower left")
ax.set_title("Part 3: braking car (speed accurate, acceleration noisy)", fontsize=10)
ax.grid(alpha=0.3)

plot_file = output_dir / "task_15_diagram.png"
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