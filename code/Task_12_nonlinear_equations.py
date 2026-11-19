import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# ---------- Save everything printed to output/task_12_output.txt ----------
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

out_file = open(os.path.join(OUT_DIR, "task_12_output.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, out_file)

# ---------- A. Polynomial example: x^2 - 5x + 6 = 0 ----------
print("=== A. POLYNOMIAL EXAMPLE: x^2 - 5x + 6 = 0 ===")
print("Roots:", np.roots([1, -5, 6]))

# ---------- B. Original problem: circle and parabola ----------
# Circle:   x^2 + y^2 = R2      (R2 = 4, so radius 2)
# Parabola: y = x^2 - 1
def make_equations(R2):
    def equations(z):
        x, y = z
        return [x**2 + y**2 - R2,
                y - (x**2 - 1)]
    return equations

equations = make_equations(4)

print("\n=== B. ORIGINAL PROBLEM: circle x^2+y^2=4 and parabola y=x^2-1 ===")
original_points = []
for guess in ([1.0, 1.0], [-1.0, 1.0]):
    solution = root(equations, guess)
    print("Initial guess:", guess)
    print("Intersection:", solution.x)
    print("Residuals:", np.array(equations(solution.x)))
    print("Converged:", solution.success)
    print()
    original_points.append(solution.x)

# ---------- C. Changing the inputs ----------
print("=== C. CHANGING THE INPUTS ===")
print("C1. Different starting guesses (circle radius 2):")
for guess in ([1.0, 1.0], [-1.0, 1.0], [3.0, 3.0], [0.0, -2.0]):
    sol = root(equations, guess)
    res = np.max(np.abs(equations(sol.x)))
    print(f"  guess {guess!s:<12} -> {np.round(sol.x, 6)}  "
          f"max residual {res:.2e}  converged: {sol.success}")

def all_intersections(R2):
    """Try many starting guesses and keep distinct, genuinely converged solutions."""
    eq = make_equations(R2)
    found = []
    for gx in (-2, -1, 0, 1, 2):
        for gy in (-2, -1, 0, 1, 2):
            s = root(eq, [gx, gy])
            if s.success and np.max(np.abs(eq(s.x))) < 1e-8:
                if not any(np.allclose(s.x, f, atol=1e-6) for f in found):
                    found.append(s.x)
    return found

print("\nC2. Changing the circle's size (many starting guesses):")
for R2 in (4, 1, 0.5):
    pts = all_intersections(R2)
    print(f"  x^2+y^2={R2}  (radius {np.sqrt(R2):.3f}) -> {len(pts)} intersection(s)")
    for p in sorted(pts, key=lambda p: (p[0], p[1])):
        print(f"      ({p[0]:.6f}, {p[1]:.6f})")

# ---------- E. Related case: road crossing a cell-tower coverage zone ----------
# Tower at the origin, coverage radius 5 km:   x^2 + y^2 = 25
# Road follows the curve:                      y = 0.1 x^2 - 3   (km)
def road_equations(z):
    x, y = z
    return [x**2 + y**2 - 25,
            y - (0.1 * x**2 - 3)]

print("\n=== E. REAL-WORLD CASE: road crossing a 5 km cell-tower coverage zone ===")
road_points = []
for guess in ([4.0, -1.0], [-4.0, -1.0]):
    sol = root(road_equations, guess)
    print("Initial guess:", guess)
    print(f"Boundary crossing: x = {sol.x[0]:.4f} km, y = {sol.x[1]:.4f} km")
    print("Residuals:", np.array(road_equations(sol.x)))
    print("Converged:", sol.success)
    print()
    road_points.append(sol.x)
width = abs(road_points[0][0] - road_points[1][0])
print(f"Road is inside coverage for x between {road_points[1][0]:.4f} and {road_points[0][0]:.4f} km")
print(f"Horizontal distance covered: {width:.4f} km")

sys.stdout = sys.__stdout__
out_file.close()

# ---------- Graphs ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
t = np.linspace(0, 2 * np.pi, 400)

ax = axes[0]
ax.plot(2 * np.cos(t), 2 * np.sin(t), label="Circle x²+y²=4")
xs = np.linspace(-2.2, 2.2, 400)
ax.plot(xs, xs**2 - 1, label="Parabola y=x²-1")
for p in original_points:
    ax.scatter(p[0], p[1], color="red", zorder=3)
    ax.annotate(f"({p[0]:.3f}, {p[1]:.3f})", p, textcoords="offset points", xytext=(6, 6))
ax.set_aspect("equal")
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
ax.set_title("Original problem: circle and parabola")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(loc="lower right")

ax = axes[1]
ax.plot(5 * np.cos(t), 5 * np.sin(t), label="Coverage zone (5 km)")
xr = np.linspace(-7, 7, 400)
ax.plot(xr, 0.1 * xr**2 - 3, label="Road y=0.1x²-3")
ax.scatter([0], [0], color="black", marker="^", s=80, label="Tower")
for p in road_points:
    ax.scatter(p[0], p[1], color="red", zorder=3)
    ax.annotate(f"({p[0]:.2f}, {p[1]:.2f})", p, textcoords="offset points", xytext=(6, 8))
ax.set_aspect("equal")
ax.set_xlim(-7, 7); ax.set_ylim(-7, 5)
ax.set_title("Real-world case: road and tower coverage")
ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)"); ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "task_12_diagram.png"), dpi=150)
plt.show()