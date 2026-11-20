import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson, quad

# ---------- Save everything printed to output/task_16_output.txt ----------
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(OUT_DIR, exist_ok=True)

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for s in self.streams:
            s.write(text)
    def flush(self):
        for s in self.streams:
            s.flush()

out_file = open(os.path.join(OUT_DIR, "task_16_output.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, out_file)

def trapezoid(y, x):
    """Trapezoidal rule (works on every NumPy version)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    return float(np.sum((y[:-1] + y[1:]) / 2 * np.diff(x)))

# ---------- A. Simple example: constant power ----------
print("=== A. SIMPLE EXAMPLE: 2 kW for 3 hours ===")
t_const = np.array([0.0, 1.5, 3.0])
p_const = np.array([2.0, 2.0, 2.0])
print(f"Energy = {trapezoid(p_const, t_const):.3f} kWh (expected 2 x 3 = 6)")

# ---------- B. Original problem: smart-meter readings every two hours ----------
time_hours = np.array([0, 2, 4, 6, 8, 10, 12], dtype=float)
power_kw = np.array([0.8, 1.0, 1.4, 1.8, 1.6, 1.2, 0.9])

energy_trapezoid = trapezoid(power_kw, time_hours)
energy_simpson = float(simpson(power_kw, x=time_hours))

print("\n=== B. ORIGINAL PROBLEM: smart-meter energy over 12 hours ===")
print("Inputs: time_hours =", time_hours)
print("        power_kw   =", power_kw)
print(f"Energy using trapezoidal rule: {energy_trapezoid:.3f} kWh")
print(f"Energy using Simpson's rule:   {energy_simpson:.3f} kWh")
print(f"Difference between estimates:  {abs(energy_simpson - energy_trapezoid):.3f} kWh")

# ---------- C. Changing the inputs ----------
print("\n=== C. CHANGING THE INPUTS ===")

# C1: change one reading (peak at 06:00 raised from 1.8 to 2.4)
p2 = power_kw.copy()
p2[3] = 2.4
print("C1. Reading at 06:00 changed from 1.8 kW to 2.4 kW:")
print(f"    Trapezoid: {energy_trapezoid:.3f} -> {trapezoid(p2, time_hours):.3f} kWh")
print(f"    Simpson:   {energy_simpson:.3f} -> {float(simpson(p2, x=time_hours)):.3f} kWh")

# C2: how the answer depends on the reading interval, using a known smooth function
# P(t) = 1.2 + 0.7*sin(pi*t/12) kW over 0..12 h; the exact energy is 1.2*12 + 0.7*24/pi
print("\nC2. Reading interval (smooth test function, exact answer known):")
f = lambda t: 1.2 + 0.7 * np.sin(np.pi * t / 12)
exact = 1.2 * 12 + 0.7 * 24 / np.pi
print(f"    Exact energy: {exact:.6f} kWh")
print("    Interval  Trapezoid error   Simpson error")
for n in (2, 4, 6, 12):               # number of sub-intervals
    t = np.linspace(0, 12, n + 1)
    y = f(t)
    print(f"    {12 / n:5.1f} h   {abs(trapezoid(y, t) - exact):13.6f}   "
          f"{abs(float(simpson(y, x=t)) - exact):13.6f}")

# C3: adaptive quadrature (quad) on the same function
val, err = quad(f, 0, 12)
print(f"\nC3. Adaptive quad on the smooth function: {val:.6f} kWh (error estimate {err:.1e})")

# ---------- E. Related case: distance travelled from a bus's speed readings ----------
# Assumed speed readings every 5 minutes for 1 hour
t_min = np.arange(0, 65, 5, dtype=float)
speed_kmh = np.array([0, 18, 32, 40, 45, 30, 0, 12, 35, 48, 50, 36, 0], dtype=float)
t_h = t_min / 60.0
dist_trap = trapezoid(speed_kmh, t_h)
dist_simp = float(simpson(speed_kmh, x=t_h))
avg_speed = dist_trap / t_h[-1]

print("\n=== E. REAL-WORLD CASE: distance of a city bus from speed readings ===")
print("Time (min):", t_min.astype(int))
print("Speed (km/h):", speed_kmh.astype(int))
print(f"Distance using trapezoidal rule: {dist_trap:.3f} km")
print(f"Distance using Simpson's rule:   {dist_simp:.3f} km")
print(f"Difference between estimates:    {abs(dist_simp - dist_trap):.3f} km")
print(f"Average speed over the hour:     {avg_speed:.2f} km/h")

sys.stdout = sys.__stdout__
out_file.close()

# ---------- Graphs ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(time_hours, power_kw, "o-", color="tab:blue", label="Power readings")
ax.fill_between(time_hours, power_kw, alpha=0.25, label=f"Area = energy ({energy_trapezoid:.2f} kWh)")
ax.set_xlabel("Time (hours)"); ax.set_ylabel("Power (kW)")
ax.set_title("Original problem: smart-meter energy")
ax.legend(); ax.grid(True)

ax = axes[1]
ax.plot(t_min, speed_kmh, "o-", color="tab:green", label="Speed readings")
ax.fill_between(t_min, speed_kmh, alpha=0.25, color="tab:green",
                label=f"Area = distance ({dist_trap:.2f} km)")
ax.set_xlabel("Time (minutes)"); ax.set_ylabel("Speed (km/h)")
ax.set_title("Real-world case: bus distance")
ax.legend(); ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "task_16_diagram.png"), dpi=150)
plt.show()