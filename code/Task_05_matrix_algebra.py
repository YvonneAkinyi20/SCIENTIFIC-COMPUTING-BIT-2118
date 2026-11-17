"""
BIT 2118 Scientific Computing - Task D: Matrix Algebra and Matrix Operations

Part 0: Concept check (addition, multiplication, determinant, inverse)
Part 1: Original problem (rotate and scale the corners of a square)
Part 2: Changing the inputs and observing the results
Part 3: Related real-world case: electrical circuit (solving Ax = b)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # lets the script save the plot without opening a window
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)   # tidy printing of arrays


# ---------------------------------------------------------------
# PART 0: CONCEPT CHECK
# ---------------------------------------------------------------
print("=" * 60)
print("PART 0: MATRIX OPERATIONS")
print("=" * 60)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
x = np.array([5, 6])

print("A + B =\n", A + B)
print("A @ x =", A @ x, " (expected [17 39])")
print("A @ B =\n", A @ B)
print("Determinant of A:", round(np.linalg.det(A), 4))
A_inv = np.linalg.inv(A)
print("Inverse of A:\n", A_inv)
print("A @ A_inv equals the identity matrix?",
      np.allclose(A @ A_inv, np.eye(2)))


# ---------------------------------------------------------------
# PART 1: ORIGINAL PROBLEM
# ---------------------------------------------------------------
print()
print("=" * 60)
print("PART 1: ROTATE AND SCALE A SQUARE (original problem)")
print("=" * 60)

# Four corners of a square: each row is [x, y]
points = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
])

angle = np.deg2rad(45)
rotation = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle),  np.cos(angle)]
])
scale = np.array([
    [2.0, 0.0],
    [0.0, 1.5]
])

# Combined transformation (scale first, then rotate)
transformation = rotation @ scale

# Row-vector form requires multiplication by transpose
transformed_points = points @ transformation.T

print("Original points:\n", points)
print("Transformation matrix:\n", transformation)
print("Transformed points:\n", transformed_points)
print("Determinant:", np.linalg.det(transformation))
print("Expected determinant (2.0 x 1.5 x 1):", 2.0 * 1.5)


# ---------------------------------------------------------------
# PART 2: CHANGING THE INPUTS
# ---------------------------------------------------------------
print()
print("=" * 60)
print("PART 2: CHANGING THE INPUT VALUES")
print("=" * 60)


def make_rotation(degrees):
    a = np.deg2rad(degrees)
    return np.array([[np.cos(a), -np.sin(a)],
                     [np.sin(a),  np.cos(a)]])


# Change 1: different rotation angles (corner [1, 1] after rotation + scaling)
print("Change 1: different rotation angles (corner [1, 1] shown)")
print(f"{'angle':>8} {'new x':>10} {'new y':>10} {'determinant':>13}")
for deg in [0, 30, 45, 90, 180]:
    T = make_rotation(deg) @ scale
    p = T @ np.array([1.0, 1.0])
    print(f"{deg:>7}° {p[0]:>10.4f} {p[1]:>10.4f} {np.linalg.det(T):>13.4f}")

# Change 2: different scale matrices
print("\nChange 2: different scale matrices (no rotation)")
print(f"{'scale (sx, sy)':>16} {'determinant':>13} {'invertible?':>13}")
for sx, sy in [(1.0, 1.0), (2.0, 1.5), (-1.0, 1.0), (2.0, 0.0)]:
    S = np.array([[sx, 0.0], [0.0, sy]])
    d = np.linalg.det(S)
    print(f"{str((sx, sy)):>16} {d:>13.4f} {str(abs(d) > 1e-12):>13}")

# Change 3: order of multiplication
order_1 = rotation @ scale      # scale, then rotate
order_2 = scale @ rotation      # rotate, then scale
print("\nChange 3: order of multiplication")
print("rotation @ scale =\n", order_1)
print("scale @ rotation =\n", order_2)
print("Are they equal?", np.allclose(order_1, order_2))


# ---------------------------------------------------------------
# PART 3: RELATED REAL-WORLD CASE
# Electrical circuit: solving a system of equations Ax = b
# ---------------------------------------------------------------
print()
print("=" * 60)
print("PART 3: ELECTRICAL CIRCUIT (SOLVING Ax = b)")
print("=" * 60)

# Assumptions:
#   1. A two-loop circuit with resistors R1 = 4, R2 = 6, R3 = 5 ohms.
#      R2 is shared by both loops.
#   2. Loop 1 has a 12 V source; loop 2 has a 6 V source opposing it.
#   3. Mesh-current equations (Kirchhoff's voltage law):
#         (R1 + R2) I1 -        R2  I2 = 12
#              -R2  I1 + (R2 + R3) I2 = -6
R1, R2, R3 = 4.0, 6.0, 5.0
V1, V2 = 12.0, 6.0

A_circuit = np.array([
    [R1 + R2, -R2],
    [-R2, R2 + R3]
])
b_circuit = np.array([V1, -V2])

det_circuit = np.linalg.det(A_circuit)
currents = np.linalg.solve(A_circuit, b_circuit)
residual = A_circuit @ currents - b_circuit

print("Coefficient matrix A:\n", A_circuit)
print("Right-hand side b:", b_circuit)
print(f"Determinant of A: {det_circuit:.2f} (non-zero, so a unique solution exists)")
print(f"Loop current I1: {currents[0]:.4f} A")
print(f"Loop current I2: {currents[1]:.4f} A")
print("Check A @ I - b (should be ~0):", residual)
print(f"Current through shared resistor R2: {currents[0] - currents[1]:.4f} A")

# Change an input: what if R3 is increased from 5 to 20 ohms?
A_new = np.array([[R1 + R2, -R2], [-R2, R2 + 20.0]])
I_new = np.linalg.solve(A_new, b_circuit)
print(f"\nWith R3 = 20 ohms: I1 = {I_new[0]:.4f} A, I2 = {I_new[1]:.4f} A")

# Change an input: a singular system (no unique solution)
A_bad = np.array([[2.0, 4.0], [1.0, 2.0]])
print("\nSingular matrix [[2, 4], [1, 2]]: determinant =",
      round(np.linalg.det(A_bad), 4))
try:
    np.linalg.solve(A_bad, np.array([1.0, 1.0]))
except np.linalg.LinAlgError as error:
    print("np.linalg.solve raised an error:", error)


# Graph: original square and transformed square (from Part 1)
def closed(p):
    return np.vstack([p, p[0]])        # repeat first point to close the shape


plt.figure(figsize=(6, 6))
plt.plot(*closed(points).T, marker="o", label="Original square")
plt.plot(*closed(transformed_points).T, marker="s",
         label="Rotated 45° and scaled (2, 1.5)")
plt.axhline(0, color="grey", linewidth=0.8)
plt.axvline(0, color="grey", linewidth=0.8)
plt.gca().set_aspect("equal")
plt.title("Transforming a Square with Matrix Multiplication")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "matrix_transformation_plot.png"), dpi=150)
print("\nGraph saved as output/matrix_transformation_plot.png")