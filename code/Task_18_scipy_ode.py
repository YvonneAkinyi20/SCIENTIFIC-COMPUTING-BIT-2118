import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------- Save everything printed to output/task_18_output.txt ----------
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

out_file = open(os.path.join(OUT_DIR, "task_18_output.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, out_file)

# ---------- A. Simple example: exponential decay dy/dt = -k y ----------
print("=== A. SIMPLE EXAMPLE: dy/dt = -0.5 y, y(0) = 10 ===")
decay = solve_ivp(lambda t, y: -0.5 * y, (0, 4), [10.0], t_eval=[0, 2, 4], rtol=1e-9, atol=1e-11)
print("t     :", decay.t)
print("y(t)  :", np.round(decay.y[0], 4), "  exact:", np.round(10 * np.exp(-0.5 * decay.t), 4))

# ---------- B. Original problem: RC charging circuit ----------
# dV/dt = (Vs - V) / (R*C)
Vs = 5.0          # supply voltage (V)
R = 1000.0        # resistance (ohms)
C = 0.001         # capacitance (farads)
V0 = 0.0          # initial capacitor voltage

def rc_circuit(t, V, Vs=Vs, R=R, C=C):
    return [(Vs - V[0]) / (R * C)]

t_eval = np.linspace(0, 5, 200)
solution = solve_ivp(rc_circuit, (0, 5), [V0], t_eval=t_eval, rtol=1e-9, atol=1e-11)
exact = Vs * (1 - np.exp(-solution.t / (R * C)))

print("\n=== B. ORIGINAL PROBLEM: RC charging circuit ===")
print(f"Inputs: Vs = {Vs} V, R = {R:.0f} ohm, C = {C} F, V0 = {V0} V, time 0 to 5 s")
print(f"Time constant tau = R*C = {R*C:.3f} s")
print("Solver success:", solution.success)
print(f"Capacitor voltage at 5 s: {solution.y[0, -1]:.4f} V")
print(f"Maximum error against exact solution: {np.max(np.abs(solution.y[0] - exact)):.3e} V")
print("\nTime(s)  Voltage(V)  Fraction of Vs")
table_t = [0.0, 1.0, 2.0, 3.0, 5.0]
table = solve_ivp(rc_circuit, (0, 5), [V0], t_eval=table_t, rtol=1e-9, atol=1e-11)
for tt, v in zip(table.t, table.y[0]):
    print(f"{tt:6.1f}  {v:9.4f}  {v/Vs:9.1%}")

# ---------- C. Changing the inputs ----------
print("\n=== C. CHANGING THE INPUTS ===")

# C1: change R and C (time constant)
print("C1. Changing R and C (voltage at 1 s and time to reach 90% of Vs):")
print("   R (ohm)   C (F)     tau (s)   V(1 s)   Time to 90% (s)")
for RR, CC in ((1000, 0.001), (2000, 0.001), (1000, 0.0005), (500, 0.001)):
    s = solve_ivp(lambda t, V: rc_circuit(t, V, R=RR, C=CC), (0, 10), [V0],
                  t_eval=np.linspace(0, 10, 4001), rtol=1e-9, atol=1e-11)
    v1 = float(np.interp(1.0, s.t, s.y[0]))
    t90 = float(s.t[np.argmax(s.y[0] >= 0.9 * Vs)])
    print(f"   {RR:7d}  {CC:8.4f}  {RR*CC:7.3f}  {v1:7.4f}  {t90:9.3f}")
print(f"   (theory: time to 90% = tau*ln(10) = {np.log(10):.3f} x tau)")

# C2: discharging (start at 5 V, supply 0 V)
dis = solve_ivp(lambda t, V: rc_circuit(t, V, Vs=0.0), (0, 5), [5.0], t_eval=t_eval,
                rtol=1e-9, atol=1e-11)
exact_dis = 5.0 * np.exp(-dis.t / (R * C))
print("\nC2. Discharging: start at 5 V, supply 0 V")
print(f"   Voltage at 1 s: {float(np.interp(1.0, dis.t, dis.y[0])):.4f} V (theory 5*e^-1 = {5*np.exp(-1):.4f} V)")
print(f"   Voltage at 5 s: {dis.y[0, -1]:.4f} V")
print(f"   Maximum error: {np.max(np.abs(dis.y[0] - exact_dis)):.3e} V")

# C3: tolerance and solver method
print("\nC3. Solver settings (charging circuit, error at 5 s grid, function evaluations):")
print("   Method  rtol     max error      nfev")
for method in ("RK45", "RK23", "Radau"):
    for rtol in (1e-3, 1e-6, 1e-9):
        s = solve_ivp(rc_circuit, (0, 5), [V0], t_eval=t_eval, method=method, rtol=rtol, atol=1e-11)
        err = np.max(np.abs(s.y[0] - Vs * (1 - np.exp(-s.t / (R * C)))))
        print(f"   {method:<6}  {rtol:.0e}  {err:12.3e}  {s.nfev:5d}")

# C4: a time-varying supply: 5 V square wave (switching on/off every 2 s)
def rc_square(t, V):
    vs = 5.0 if int(t // 2) % 2 == 0 else 0.0
    return [(vs - V[0]) / (R * C)]
sq = solve_ivp(rc_square, (0, 8), [0.0], t_eval=np.linspace(0, 8, 801), max_step=0.01)
print("\nC4. Supply switched on/off every 2 s (square wave):")
for tt in (2, 4, 6, 8):
    print(f"   V at {tt} s: {float(np.interp(tt, sq.t, sq.y[0])):.4f} V")

# ---------- E. Related case: SIR epidemic model (a system of three ODEs) ----------
# S' = -beta*S*I/N ; I' = beta*S*I/N - gamma*I ; R' = gamma*I
N = 1000.0
beta, gamma = 0.30, 0.10       # infection rate, recovery rate (per day)
def sir(t, y):
    S, I, Rr = y
    return [-beta * S * I / N,
            beta * S * I / N - gamma * I,
            gamma * I]

days = np.linspace(0, 160, 1601)
epi = solve_ivp(sir, (0, 160), [N - 1, 1, 0], t_eval=days, rtol=1e-9, atol=1e-11)
S, I, Rr = epi.y
peak_i = int(np.argmax(I))
print("\n=== E. REAL-WORLD CASE: SIR epidemic model ===")
print(f"Inputs: N = {N:.0f}, beta = {beta}/day, gamma = {gamma}/day, start with 1 infected")
print(f"Basic reproduction number R0 = beta/gamma = {beta/gamma:.1f}")
print("Solver success:", epi.success)
print(f"Peak infections: {I[peak_i]:.1f} people on day {epi.t[peak_i]:.1f}")
print(f"Total ever infected by day 160: {Rr[-1] + I[-1]:.1f} of {N:.0f} ({(Rr[-1]+I[-1])/N:.1%})")
print(f"Never infected (still susceptible): {S[-1]:.1f}")
print(f"Conservation check, max |S+I+R-N|: {np.max(np.abs(S + I + Rr - N)):.2e}")
for b in (0.15, 0.20, 0.30):
    e = solve_ivp(lambda t, y: [-b*y[0]*y[1]/N, b*y[0]*y[1]/N - gamma*y[1], gamma*y[1]],
                  (0, 400), [N - 1, 1, 0], t_eval=np.linspace(0, 400, 4001), rtol=1e-9, atol=1e-11)
    print(f"beta = {b:.2f} (R0 = {b/gamma:.1f}): peak {e.y[1].max():7.1f} infected, "
          f"total infected {(e.y[2, -1] + e.y[1, -1]):6.1f}")

sys.stdout = sys.__stdout__
out_file.close()

# ---------- Graphs ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
ax.plot(solution.t, solution.y[0], label="solve_ivp")
ax.plot(solution.t, exact, "--", label="Exact")
ax.axhline(Vs * (1 - np.exp(-1)), color="gray", ls=":")
ax.axvline(R * C, color="gray", ls=":", label="t = RC (63.2% of Vs)")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Capacitor voltage (V)")
ax.set_title("Original problem: RC circuit charging")
ax.legend(); ax.grid(True)

ax = axes[0, 1]
for RR, CC in ((500, 0.001), (1000, 0.001), (2000, 0.001)):
    s = solve_ivp(lambda t, V: rc_circuit(t, V, R=RR, C=CC), (0, 5), [V0], t_eval=t_eval,
                  rtol=1e-9, atol=1e-11)
    ax.plot(s.t, s.y[0], label=f"R = {RR} ohm (tau = {RR*CC:.1f} s)")
ax.plot(dis.t, dis.y[0], "k--", label="Discharging (R = 1000)")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Capacitor voltage (V)")
ax.set_title("Changing R, and discharging"); ax.legend(); ax.grid(True)

ax = axes[1, 0]
ax.plot(sq.t, sq.y[0], label="Capacitor voltage")
ax.step(np.linspace(0, 8, 801), [5.0 if int(t // 2) % 2 == 0 else 0.0 for t in np.linspace(0, 8, 801)],
        color="gray", ls=":", where="post", label="Supply voltage")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Voltage (V)")
ax.set_title("Square-wave supply"); ax.legend(); ax.grid(True)

ax = axes[1, 1]
ax.plot(epi.t, S, label="Susceptible")
ax.plot(epi.t, I, label="Infected")
ax.plot(epi.t, Rr, label="Recovered")
ax.axvline(epi.t[peak_i], color="gray", ls=":")
ax.set_xlabel("Time (days)"); ax.set_ylabel("People")
ax.set_title("Real-world case: SIR epidemic model"); ax.legend(); ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "task_18_diagram.png"), dpi=150)
plt.show()