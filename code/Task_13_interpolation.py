import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline

# ---------- Save everything printed to output/task_13_output.txt ----------
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

out_file = open(os.path.join(OUT_DIR, "task_13_output.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, out_file)

def clock(h):
    """Turn 10.5 into '10:30'."""
    return f"{int(h):02d}:{int(round((h - int(h)) * 60)):02d}"

# ---------- A. Simple example ----------
print("=== A. SIMPLE EXAMPLE: 20 C at 10:00 and 24 C at 12:00 ===")
simple = interp1d([10, 12], [20, 24])
print(f"Linear estimate at 11:00: {float(simple(11)):.2f} C")

# ---------- B. Original problem: hourly sensor, estimates every 10 minutes ----------
hours = np.array([8, 9, 10, 11, 12, 13, 14], dtype=float)
temperature = np.array([18.0, 20.0, 22.5, 24.0, 25.0, 25.5, 25.0])

linear = interp1d(hours, temperature, kind="linear")
cubic = CubicSpline(hours, temperature)

# Estimate every 10 minutes = 1/6 hour (linspace avoids floating-point overshoot past 14)
ten_minute_times = np.linspace(8, 14, 37)
linear_estimates = linear(ten_minute_times)
cubic_estimates = cubic(ten_minute_times)

target_time = 10.5  # 10:30
print("\n=== B. ORIGINAL PROBLEM: hourly temperatures, estimate every 10 minutes ===")
print("Inputs: hours       =", hours)
print("        temperature =", temperature)
print(f"Linear estimate at 10:30: {float(linear(target_time)):.2f} C")
print(f"Cubic estimate at 10:30:  {float(cubic(target_time)):.2f} C")
print(f"Number of 10-minute estimates: {len(ten_minute_times)}")
print("\nTime   Linear   Cubic")
for t in ten_minute_times[6:13]:          # 09:00 to 10:00 as a sample
    print(f"{clock(t)}  {float(linear(t)):6.2f}  {float(cubic(t)):6.2f}")

# ---------- C. Changing the inputs ----------
print("\n=== C. CHANGING THE INPUTS ===")

# C1: change one measurement (a spike at 11:00: 24.0 -> 27.0)
temp_spike = temperature.copy()
temp_spike[3] = 27.0
lin2 = interp1d(hours, temp_spike)
cub2 = CubicSpline(hours, temp_spike)
print("C1. Measurement at 11:00 changed from 24.0 to 27.0:")
print(f"  Linear at 10:30: {float(linear(10.5)):.2f} -> {float(lin2(10.5)):.2f} C")
print(f"  Cubic  at 10:30: {float(cubic(10.5)):.2f} -> {float(cub2(10.5)):.2f} C")
print(f"  Linear at 12:30: {float(linear(12.5)):.2f} -> {float(lin2(12.5)):.2f} C")
print(f"  Cubic  at 12:30: {float(cubic(12.5)):.2f} -> {float(cub2(12.5)):.2f} C")

# C2: use fewer measurements (every 2 hours) and predict the hours we left out
h2 = np.array([8, 10, 12, 14], dtype=float)
t2 = np.array([18.0, 22.5, 25.0, 25.0])
lin3 = interp1d(h2, t2)
cub3 = CubicSpline(h2, t2)
print("\nC2. Only every 2 hours used (08, 10, 12, 14); estimate the left-out hours:")
print("Time   Actual  Linear  err    Cubic   err")
for h, actual in ((9, 20.0), (11, 24.0), (13, 25.5)):
    le, ce = float(lin3(h)), float(cub3(h))
    print(f"{clock(h)}  {actual:6.2f}  {le:6.2f}  {le-actual:+5.2f}  {ce:6.2f}  {ce-actual:+5.2f}")

# C3: asking outside the measured range (extrapolation)
print("\nC3. Estimating outside the data range (at 14:30):")
try:
    linear(14.5)
except ValueError:
    print("  Linear interp1d: ValueError (value above the interpolation range)")
print(f"  CubicSpline extrapolates silently: {float(cubic(14.5)):.2f} C")

# ---------- E. Related case: solar panel output measured hourly ----------
# Assumed data (kW), hourly from 06:00 to 18:00
sun_hours = np.arange(6, 19, dtype=float)
power = np.array([0.0, 0.4, 1.2, 2.1, 2.8, 3.2, 3.3, 3.0, 2.4, 1.6, 0.8, 0.2, 0.0])
sun_linear = interp1d(sun_hours, power)
sun_cubic = CubicSpline(sun_hours, power)

print("\n=== E. REAL-WORLD CASE: solar panel output measured hourly ===")
for t in (10.75, 15.33333333):
    print(f"Estimate at {clock(t)}: linear {float(sun_linear(t)):.3f} kW | cubic {float(sun_cubic(t)):.3f} kW")
fine = np.linspace(6, 18, 721)
trapezoid = float(np.sum((power[:-1] + power[1:]) / 2 * np.diff(sun_hours)))
spline_energy = float(sun_cubic.integrate(6, 18))
print(f"Energy produced, trapezoid on hourly data: {trapezoid:.3f} kWh")
print(f"Energy produced, cubic spline integral:    {spline_energy:.3f} kWh")
print(f"Lowest cubic value between measurements:   {float(sun_cubic(fine).min()):.3f} kW")
print(f"Highest cubic value between measurements:  {float(sun_cubic(fine).max()):.3f} kW "
      f"(highest measurement {power.max():.1f} kW)")

sys.stdout = sys.__stdout__
out_file.close()

# ---------- Graphs ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.scatter(hours, temperature, color="black", zorder=3, label="Hourly measurements")
ax.plot(ten_minute_times, linear_estimates, label="Linear")
ax.plot(ten_minute_times, cubic_estimates, label="Cubic spline")
ax.scatter([target_time], [float(cubic(target_time))], color="red", zorder=4, label="10:30 estimate")
ax.set_xlabel("Hour of day"); ax.set_ylabel("Temperature (C)")
ax.set_title("Original problem: temperature interpolation")
ax.legend(); ax.grid(True)

ax = axes[1]
ax.scatter(sun_hours, power, color="black", zorder=3, label="Hourly measurements")
ax.plot(fine, sun_linear(fine), label="Linear")
ax.plot(fine, sun_cubic(fine), label="Cubic spline")
ax.set_xlabel("Hour of day"); ax.set_ylabel("Power (kW)")
ax.set_title("Real-world case: solar panel output")
ax.legend(); ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "task_13_diagram.png"), dpi=150)
plt.show()