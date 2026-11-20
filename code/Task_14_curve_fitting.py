"""
Task 14 - Curve Fitting and Approximation (least squares)
BIT 2118 Scientific Computing

Part 0: Concept check - fit a straight line y = a + b x by least squares.
Part 1: Original problem - fit a quadratic to server response time vs users.
Part 2: Changing the inputs (model choice, outlier, fewer points, extrapolation).
Part 3: Related real-world case - cooling of a cup of coffee (exponential fit).

Just run this one file. It will:
  1. print the results in the terminal,
  2. save them to output/task_14_output.txt,
  3. draw ONE summary diagram,
  4. save it to output/task_14_diagram.png and open it automatically.

Run from the repo root:  python code/Task_14_curve_fitting.py
"""

import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.optimize import curve_fit

warnings.simplefilter("ignore")        # keep the terminal output clean

lines = []


def show(text=""):
    print(text)
    lines.append(str(text))


def header(title):
    show("=" * 64)
    show(title)
    show("=" * 64)


def rmse(y, y_hat):
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def r_squared(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# PART 0: Concept check - least squares line
# ---------------------------------------------------------------------------
header("PART 0: CONCEPT CHECK  y = a + b x")
x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y0 = np.array([2.2, 4.1, 5.9, 8.2, 9.8])
X0 = np.column_stack([np.ones_like(x0), x0])
coef0, res0, _, _ = np.linalg.lstsq(X0, y0, rcond=None)
a0, b0 = coef0
show(f"Data: x = {x0},  y = {y0}")
show(f"Least squares line: y = {a0:.3f} + {b0:.3f} x")
show(f"Sum of squared residuals: {np.sum((y0 - (a0 + b0 * x0)) ** 2):.4f}")
worse = np.sum((y0 - (0.0 + 2.1 * x0)) ** 2)
show(f"Any other line is worse, e.g. y = 0 + 2.1 x gives {worse:.4f}.")
show()

# ---------------------------------------------------------------------------
# PART 1: Original problem - server response time
# ---------------------------------------------------------------------------
users = np.array([50, 100, 150, 200, 250, 300, 350], dtype=float)
response_ms = np.array([82, 95, 115, 145, 185, 238, 305], dtype=float)


def model(x, a, b, c):
    """A simple quadratic model for increasing delay."""
    return a * x ** 2 + b * x + c


parameters, covariance = curve_fit(model, users, response_ms)
a, b, c = parameters
predicted = model(users, a, b, c)
residuals = response_ms - predicted
errors = np.sqrt(np.diag(covariance))
fit_rmse = rmse(response_ms, predicted)
fit_r2 = r_squared(response_ms, predicted)

header("PART 1: SERVER RESPONSE TIME VS CONCURRENT USERS")
show(f"Users      : {users}")
show(f"Response ms: {response_ms}")
show(f"Fitted model: y = {a:.6f}x^2 {b:+.4f}x {c:+.2f}")
show(f"Parameter standard errors: a +/- {errors[0]:.6f}, b +/- {errors[1]:.4f}, c +/- {errors[2]:.2f}")
show(f"RMSE: {fit_rmse:.2f} ms")
show(f"R^2 : {fit_r2:.5f}")
show(f"Residuals (ms): {np.round(residuals, 2)}   (sum = {residuals.sum():.2e})")
show(f"Predicted response at 400 users: {model(400, a, b, c):.2f} ms")
show(f"Predicted response at 500 users: {model(500, a, b, c):.2f} ms")
show()

# ---------------------------------------------------------------------------
# PART 2: Changing the inputs
# ---------------------------------------------------------------------------
header("PART 2: CHANGING THE INPUTS")


def exp_model(x, p, q, r):
    return p * np.exp(q * x) + r


def fit_model(name, xs, ys):
    """Return a function that predicts for any x, fitted to (xs, ys)."""
    if name == "exponential":
        par, _ = curve_fit(exp_model, xs, ys, p0=[10, 0.01, 70], maxfev=20000)
        return lambda xx: exp_model(xx, *par)
    deg = {"linear": 1, "quadratic": 2, "cubic": 3, "degree 5": 5}[name]
    co = np.polyfit(xs, ys, deg)
    return lambda xx: np.polyval(co, xx)


names = ["linear", "quadratic", "cubic", "exponential", "degree 5"]
fits = {n: fit_model(n, users, response_ms) for n in names}

show("a) Different models (7 data points):")
show(f"   {'model':12s} {'RMSE':>8s} {'R^2':>9s} {'leave-one-out RMSE':>20s} {'at 400 users':>14s} {'at 500 users':>14s}")
model_rmse, model_loo = {}, {}
for n in names:
    pred = fits[n](users)
    loo = []
    for i in range(len(users)):
        keep = np.arange(len(users)) != i
        f_i = fit_model(n, users[keep], response_ms[keep])
        loo.append(response_ms[i] - f_i(users[i]))
    model_rmse[n] = rmse(response_ms, pred)
    model_loo[n] = float(np.sqrt(np.mean(np.square(loo))))
    show(f"   {n:12s} {model_rmse[n]:8.2f} {r_squared(response_ms, pred):9.5f} "
         f"{model_loo[n]:20.2f} {fits[n](400.0):14.1f} {fits[n](500.0):14.1f}")
show("   Cubic and degree-5 curves fit the 7 points almost exactly (RMSE 0.14), but the "
     "leave-one-out error shows degree 5 is overfitting (it cannot predict a point it has not seen).")
show("   The predictions at 500 users range from about 386 to 699 ms: the model you choose "
     "matters most when you extrapolate.")

show("\nb) One outlier (the 250-user measurement 185 ms is recorded as 300 ms):")
y_out = response_ms.copy(); y_out[4] = 300.0
par_out, _ = curve_fit(model, users, y_out)
show(f"   Original fit: y = {a:.6f}x^2 {b:+.4f}x {c:+.2f},  RMSE {fit_rmse:.2f} ms,  "
     f"at 400 users: {model(400, a, b, c):.1f} ms")
show(f"   With outlier: y = {par_out[0]:.6f}x^2 {par_out[1]:+.4f}x {par_out[2]:+.2f},  "
     f"RMSE {rmse(y_out, model(users, *par_out)):.2f} ms,  at 400 users: "
     f"{model(400, *par_out):.1f} ms")
show("   One bad point shifts every parameter - least squares is sensitive to outliers.")

show("\nc) Fewer data points (fit only users 50-200, then predict the rest):")
sub = users <= 200
par_sub, _ = curve_fit(model, users[sub], response_ms[sub])
for u, true in zip(users[~sub], response_ms[~sub]):
    show(f"   {u:.0f} users: predicted {model(u, *par_sub):6.1f} ms,  measured {true:.0f} ms,  "
         f"error {model(u, *par_sub) - true:+6.1f} ms")
show("   Predicting beyond the fitted range (extrapolation) gets less reliable.")

show("\nd) More noise (add the same +/-15 ms pattern to the measurements):")
noise = np.array([10, -15, 12, -8, 15, -12, 9], dtype=float)
par_noisy, cov_noisy = curve_fit(model, users, response_ms + noise)
err_noisy = np.sqrt(np.diag(cov_noisy))
show(f"   RMSE rises to {rmse(response_ms + noise, model(users, *par_noisy)):.2f} ms "
     f"(from {fit_rmse:.2f});  standard error of a rises from {errors[0]:.6f} to {err_noisy[0]:.6f}.")
show()

# ---------------------------------------------------------------------------
# PART 3: Related real-world case - cooling coffee
# ---------------------------------------------------------------------------
header("PART 3: COOLING OF A CUP OF COFFEE  T(t) = T_room + (T0 - T_room) exp(-k t)")
t_min = np.array([0, 2, 4, 6, 8, 10, 15, 20, 30], dtype=float)
temp_c = np.array([90.2, 80.7, 73.9, 66.2, 61.3, 55.4, 46.3, 38.5, 30.6])


def cooling(t, T_room, dT, k):
    return T_room + dT * np.exp(-k * t)


par_c, cov_c = curve_fit(cooling, t_min, temp_c, p0=[20, 70, 0.05])
T_room, dT, k = par_c
err_c = np.sqrt(np.diag(cov_c))
pred_c = cooling(t_min, *par_c)
lin_c = np.polyfit(t_min, temp_c, 1)
show(f"Measured temperatures (C) at t = {t_min} min:\n   {temp_c}")
show(f"Fitted: T(t) = {T_room:.2f} + {dT:.2f} exp(-{k:.4f} t)")
show(f"Room temperature estimate: {T_room:.2f} +/- {err_c[0]:.2f} C;  "
     f"cooling constant k = {k:.4f} +/- {err_c[2]:.4f} per minute")
show(f"Time constant 1/k = {1 / k:.1f} min;  half-life of the temperature "
     f"difference = {np.log(2) / k:.1f} min")
show(f"RMSE exponential: {rmse(temp_c, pred_c):.2f} C   "
     f"(a straight line gives {rmse(temp_c, np.polyval(lin_c, t_min)):.2f} C)")
target = 45.0
t_target = -np.log((target - T_room) / dT) / k
show(f"Time for the coffee to cool to {target:.0f} C: {t_target:.1f} minutes")
show(f"Predicted temperature after 60 minutes: {cooling(60.0, *par_c):.1f} C "
     f"(approaches room temperature {T_room:.1f} C)")
show()

# ---------------------------------------------------------------------------
# Save text output
# ---------------------------------------------------------------------------
output_dir = Path(__file__).resolve().parent.parent / "output"
output_dir.mkdir(exist_ok=True)
(output_dir / "task_14_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Output saved to: {output_dir / 'task_14_output.txt'}")

# ---------------------------------------------------------------------------
# ONE summary diagram
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 11.5))
gs = fig.add_gridspec(3, 3, height_ratios=[0.7, 1, 1], hspace=0.5, wspace=0.3)
fig.suptitle("Task 14: Curve Fitting and Approximation - Summary Diagram",
             fontsize=16, fontweight="bold")

# ---- Row 1: flowchart ----
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 100); ax.set_ylim(0, 20); ax.axis("off")
ax.set_title("Method: choose a model, minimise the squared residuals, then inspect the fit",
             fontsize=12)
boxes = [
    (1,  "Data\nusers and response\ntime (ms)",                  "#dbe9f6"),
    (22, "Choose a model\ny = a x^2 + b x + c",                  "#fde2c4"),
    (43, "Fit\ncurve_fit minimises\nsum of (y - y_hat)^2",        "#fff3c4"),
    (64, "Inspect\nRMSE, R^2 and\nresidual plot",                 "#e3f2d9"),
    (84, "Predict\nresponse at\n400 users",                       "#f6d9e8"),
]
for xpos, text, col in boxes:
    ax.add_patch(FancyBboxPatch((xpos, 4.5), 15, 11, boxstyle="round,pad=0.4",
                                fc=col, ec="black", lw=1.2))
    ax.text(xpos + 7.5, 10, text, ha="center", va="center", fontsize=8.5)
for x0 in (16.5, 37.5, 58.5, 79.5):
    ax.add_patch(FancyArrowPatch((x0 + 0.5, 10), (x0 + 5, 10), arrowstyle="-|>",
                                 mutation_scale=18, lw=1.5, color="black"))

# ---- Row 2, col 1: fit with uncertainty band ----
ax = fig.add_subplot(gs[1, 0])
x_plot = np.linspace(users.min(), 400, 300)
rng = np.random.default_rng(0)
draws = rng.multivariate_normal(parameters, covariance, size=400)
curves = np.array([model(x_plot, *d) for d in draws])
lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
ax.fill_between(x_plot, lo, hi, alpha=0.25, label="95% parameter band")
ax.scatter(users, response_ms, color="black", zorder=3, label="Measurements")
ax.plot(x_plot, model(x_plot, a, b, c), lw=2, label="Fitted curve")
ax.plot(400, model(400, a, b, c), "r*", ms=14, label=f"400 users: {model(400, a, b, c):.0f} ms")
ax.set_xlabel("Concurrent users"); ax.set_ylabel("Response time (ms)")
ax.set_title(f"Part 1: quadratic fit (RMSE {fit_rmse:.2f} ms, R^2 {fit_r2:.4f})", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ---- Row 2, col 2: residuals ----
ax = fig.add_subplot(gs[1, 1])
colors = ["#2e8b57" if r >= 0 else "#c0392b" for r in residuals]
bars = ax.bar(users, residuals, width=30, color=colors)
ax.bar_label(bars, fmt="%.1f", fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Concurrent users"); ax.set_ylabel("Residual (measured - fitted, ms)")
ax.set_ylim(-4, 4)
ax.set_title("Part 1: residuals (small and mixed in sign = good fit)", fontsize=10)
ax.grid(axis="y", alpha=0.3)

# ---- Row 2, col 3: model comparison curves ----
ax = fig.add_subplot(gs[1, 2])
x_wide = np.linspace(50, 500, 300)
ax.scatter(users, response_ms, color="black", zorder=3, label="Measurements")
for n, col in zip(names, ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]):
    ax.plot(x_wide, fits[n](x_wide), color=col, lw=1.8, label=n)
ax.axvline(350, color="grey", ls=":", lw=1)
ax.text(355, 30, "extrapolation", fontsize=8, color="grey")
ax.set_ylim(0, 900)
ax.set_xlabel("Concurrent users"); ax.set_ylabel("Response time (ms)")
ax.set_title("Part 2: different models, different predictions", fontsize=10)
ax.legend(fontsize=7.5, ncol=2, loc="upper left"); ax.grid(alpha=0.3)

# ---- Row 3, col 1: RMSE vs leave-one-out ----
ax = fig.add_subplot(gs[2, 0])
xt = np.arange(len(names)); w = 0.38
b1 = ax.bar(xt - w / 2, [model_rmse[n] for n in names], w, label="Fit RMSE (training)")
b2 = ax.bar(xt + w / 2, [model_loo[n] for n in names], w, label="Leave-one-out RMSE")
ax.bar_label(b1, fmt="%.1f", fontsize=7); ax.bar_label(b2, fmt="%.1f", fontsize=7)
ax.set_xticks(xt); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("RMSE (ms)")
ax.set_title("Part 2: a closer fit is not always a better model", fontsize=10)
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

# ---- Row 3, col 2: outlier ----
ax = fig.add_subplot(gs[2, 1])
ax.scatter(users, y_out, color="black", zorder=3, label="Data with outlier")
ax.scatter([users[4]], [y_out[4]], s=140, facecolors="none", edgecolors="red", lw=2, zorder=4)
ax.plot(x_plot, model(x_plot, a, b, c), lw=2, label="Fit without outlier")
ax.plot(x_plot, model(x_plot, *par_out), lw=2, ls="--", label="Fit with outlier")
ax.set_xlabel("Concurrent users"); ax.set_ylabel("Response time (ms)")
ax.set_title(f"Part 2: one outlier moves the curve (400 users: "
             f"{model(400, a, b, c):.0f} -> {model(400, *par_out):.0f} ms)", fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ---- Row 3, col 3: cooling coffee ----
ax = fig.add_subplot(gs[2, 2])
tt = np.linspace(0, 60, 300)
ax.scatter(t_min, temp_c, color="black", zorder=3, label="Measurements")
ax.plot(tt, cooling(tt, *par_c), lw=2, label=f"Exponential fit, k = {k:.3f}/min")
ax.plot(tt, np.polyval(lin_c, tt), "--", color="grey", lw=1.5, label="Straight line (poor)")
ax.axhline(T_room, color="blue", ls=":", lw=1, label=f"Room {T_room:.1f} C")
ax.axhline(target, color="red", ls=":", lw=1)
ax.plot(t_target, target, "r*", ms=13, label=f"{target:.0f} C at {t_target:.1f} min")
ax.set_ylim(0, 100)
ax.set_xlabel("Time (minutes)"); ax.set_ylabel("Temperature (C)")
ax.set_title("Part 3: cooling coffee (exponential fit)", fontsize=10)
ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

plot_file = output_dir / "task_14_diagram.png"
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