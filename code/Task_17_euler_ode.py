import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------- Save everything printed to output/task_17_output.txt ----------
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

out_file = open(os.path.join(OUT_DIR, "task_17_output.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, out_file)

def euler(f, y0, t0, t_end, h):
    """Euler's method: y[i+1] = y[i] + h * f(t[i], y[i])."""
    n = int(round((t_end - t0) / h))
    t = t0 + h * np.arange(n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        y[i + 1] = y[i] + h * f(t[i], y[i])
    return t, y

# ---------- A. Simple example: dy/dt = y, y(0) = 1, h = 0.1 ----------
print("=== A. SIMPLE EXAMPLE: dy/dt = y, y(0)=1, h=0.1 ===")
ta, ya = euler(lambda t, y: y, 1.0, 0.0, 0.2, 0.1)
print(f"y1 = {ya[1]:.2f}, y2 = {ya[2]:.2f}   (exact y2 = e^0.2 = {np.exp(0.2):.4f})")

# ---------- B. Original problem: Newton's law of cooling ----------
# dT/dt = -k (T - T_ambient)
T_ambient = 25.0
T_initial = 90.0
k = 0.12
h = 1.0            # minutes
end_time = 30.0

def cooling(t, T, k=k, Ta=T_ambient):
    return -k * (T - Ta)

def exact_T(t, k=k, T0=T_initial, Ta=T_ambient):
    return Ta + (T0 - Ta) * np.exp(-k * t)

time, temperature = euler(cooling, T_initial, 0.0, end_time, h)
exact = exact_T(time)

print("\n=== B. ORIGINAL PROBLEM: cooling of a hot object ===")
print(f"Inputs: T_ambient = {T_ambient} C, T_initial = {T_initial} C, k = {k} per min, h = {h} min, end = {end_time} min")
print(f"Euler temperature after 30 min: {temperature[-1]:.2f} C")
print(f"Exact temperature after 30 min: {exact[-1]:.2f} C")
print(f"Final absolute error: {abs(temperature[-1] - exact[-1]):.4f} C")
print("\nTime(min)  Euler    Exact    Error")
for i in (0, 1, 2, 5, 10, 20, 30):
    print(f"{time[i]:7.0f}  {temperature[i]:7.2f}  {exact[i]:7.2f}  {temperature[i]-exact[i]:+6.3f}")

# ---------- C. Changing the inputs ----------
print("\n=== C. CHANGING THE INPUTS ===")

# C1: step size
print("C1. Changing the step size h (final error at 30 min):")
print("   h (min)  Euler T(30)  Exact T(30)  Abs error")
step_results = {}
for hh in (10.0, 5.0, 2.0, 1.0, 0.5, 0.1):
    t_, y_ = euler(cooling, T_initial, 0.0, end_time, hh)
    step_results[hh] = (t_, y_)
    print(f"   {hh:6.1f}   {y_[-1]:10.4f}  {exact_T(30.0):10.4f}  {abs(y_[-1]-exact_T(30.0)):9.5f}")

# C2: a step that is too large (Euler is stable only for h < 2/k)
print(f"\nC2. Stability: Euler needs h < 2/k = {2/k:.2f} min for this problem.")
for hh in (15.0, 20.0):
    t_, y_ = euler(cooling, T_initial, 0.0, 60.0, hh)
    print(f"   h = {hh:4.1f} min -> temperatures: {np.round(y_, 1)}")

# C3: change the cooling constant k
print("\nC3. Changing the cooling constant k (h = 1 min):")
print("   k      Euler T(30)  Exact T(30)")
for kk in (0.05, 0.12, 0.25):
    t_, y_ = euler(lambda t, T: cooling(t, T, k=kk), T_initial, 0.0, end_time, 1.0)
    print(f"   {kk:4.2f}   {y_[-1]:10.2f}  {exact_T(30.0, k=kk):10.2f}")

# C4: change the initial temperature
print("\nC4. Changing the initial temperature (h = 1 min):")
print("   T0     Euler T(30)  Exact T(30)")
for T0 in (60.0, 90.0, 120.0):
    t_, y_ = euler(cooling, T0, 0.0, end_time, 1.0)
    print(f"   {T0:5.0f}  {y_[-1]:10.2f}  {exact_T(30.0, T0=T0):10.2f}")

# ---------- E. Related case: logistic population growth ----------
# dP/dt = r P (1 - P/K): growth that slows as the population nears the capacity K
r, K, P0 = 0.4, 1000.0, 50.0
def logistic(t, P):
    return r * P * (1 - P / K)
def logistic_exact(t):
    return K / (1 + (K / P0 - 1) * np.exp(-r * t))

print("\n=== E. REAL-WORLD CASE: logistic population growth ===")
print(f"Inputs: r = {r} per day, K = {K:.0f}, P0 = {P0:.0f}, end = 30 days")
print("Step h (days)  Euler P(30)  Exact P(30)  Abs error")
pop_results = {}
for hh in (2.0, 1.0, 0.5):
    t_, p_ = euler(logistic, P0, 0.0, 30.0, hh)
    pop_results[hh] = (t_, p_)
    print(f"{hh:12.1f}  {p_[-1]:11.2f}  {logistic_exact(30.0):11.2f}  {abs(p_[-1]-logistic_exact(30.0)):9.3f}")
t1, p1 = pop_results[1.0]
half = t1[np.argmax(p1 >= K / 2)]
print(f"Euler (h=1) first reaches half the capacity ({K/2:.0f}) at day {half:.0f}; "
      f"exact solution reaches it at day {np.log(K/P0 - 1)/r:.2f}")

sys.stdout = sys.__stdout__
out_file.close()

# ---------- Graphs ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
t_fine = np.linspace(0, end_time, 400)

ax = axes[0, 0]
ax.plot(time, temperature, "o-", ms=4, label="Euler (h = 1 min)")
ax.plot(t_fine, exact_T(t_fine), label="Exact solution")
ax.axhline(T_ambient, color="gray", ls=":", label="Ambient 25 C")
ax.set_xlabel("Time (minutes)"); ax.set_ylabel("Temperature (C)")
ax.set_title("Original problem: cooling of a hot object")
ax.legend(); ax.grid(True)

ax = axes[0, 1]
for hh in (10.0, 5.0, 1.0, 0.1):
    t_, y_ = step_results[hh]
    ax.plot(t_, y_, label=f"h = {hh}")
ax.plot(t_fine, exact_T(t_fine), "k--", label="Exact")
ax.set_xlabel("Time (minutes)"); ax.set_ylabel("Temperature (C)")
ax.set_title("Effect of the step size"); ax.legend(); ax.grid(True)

ax = axes[1, 0]
t_big, y_big = euler(cooling, T_initial, 0.0, 60.0, 20.0)
t_ok, y_ok = euler(cooling, T_initial, 0.0, 60.0, 5.0)
ax.plot(t_big, y_big, "o-", color="tab:red", label="h = 20 (unstable)")
ax.plot(t_ok, y_ok, "o-", color="tab:green", label="h = 5 (stable)")
ax.plot(np.linspace(0, 60, 300), exact_T(np.linspace(0, 60, 300)), "k--", label="Exact")
ax.set_ylim(-170, 170)
ax.set_xlabel("Time (minutes)"); ax.set_ylabel("Temperature (C)")
ax.set_title("Too large a step: Euler becomes unstable"); ax.legend(); ax.grid(True)

ax = axes[1, 1]
tp = np.linspace(0, 30, 400)
ax.plot(tp, logistic_exact(tp), "k--", label="Exact")
for hh in (2.0, 0.5):
    t_, p_ = pop_results[hh]
    ax.plot(t_, p_, "o-", ms=3, label=f"Euler h = {hh} day")
ax.axhline(K, color="gray", ls=":", label="Capacity K")
ax.set_xlabel("Time (days)"); ax.set_ylabel("Population")
ax.set_title("Real-world case: logistic population growth"); ax.legend(); ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "task_17_diagram.png"), dpi=150)
plt.show()