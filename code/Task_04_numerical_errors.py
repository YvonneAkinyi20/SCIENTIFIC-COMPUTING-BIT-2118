"""
BIT 2118 Scientific Computing - Task C: Floating-Point Arithmetic and Numerical Errors

Part 1: Original problem (0.1 + 0.2 == 0.3, error measures, unstable formula)
Part 2: Changing the inputs and observing the results
Part 3: Related real-world case: banking, adding 10-cent payments
"""

import os
from decimal import Decimal
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # lets the script save the plot without opening a window
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# PART 1: ORIGINAL PROBLEM
# ---------------------------------------------------------------
print("=" * 60)
print("PART 1: FLOATING-POINT ERRORS (original problem)")
print("=" * 60)

# Floating-point comparison
calculated = 0.1 + 0.2
expected = 0.3
print("0.1 + 0.2 =", calculated)
print("Exact comparison:", calculated == expected)
print("Tolerance-based comparison:", np.isclose(calculated, expected))

# Absolute and relative error
true_value = np.sqrt(2)
approx_value = 1.414
absolute_error = abs(true_value - approx_value)
relative_error = absolute_error / abs(true_value)
print(f"Absolute error: {absolute_error:.8f}")
print(f"Relative error: {relative_error:.8%}")

# Numerically unstable vs stable formulation
print("\nUnstable (direct) vs stable formula for (sqrt(1+x) - 1) / x:")
for x in [1e-4, 1e-8, 1e-12]:
    direct = (np.sqrt(1 + x) - 1) / x
    stable = 1 / (np.sqrt(1 + x) + 1)
    print(f"x={x:.0e}: direct={direct:.12f}, stable={stable:.12f}")


# ---------------------------------------------------------------
# PART 2: CHANGING THE INPUTS
# ---------------------------------------------------------------
print()
print("=" * 60)
print("PART 2: CHANGING THE INPUT VALUES")
print("=" * 60)

# Change 1: other comparisons
print("Change 1: other floating-point comparisons")
ten_tenths = 0.0
for _ in range(10):          # explicit loop: same result on every Python version
    ten_tenths += 0.1
tests = [
    ("0.5 + 0.25 == 0.75", 0.5 + 0.25, 0.75),
    ("0.1 * 3 == 0.3", 0.1 * 3, 0.3),
    ("sum of ten 0.1s == 1.0", ten_tenths, 1.0),
    ("0.1 + 0.7 == 0.8", 0.1 + 0.7, 0.8),
]
print(f"{'test':<26} {'exact ==':>10} {'isclose':>10}")
for label, a, b in tests:
    print(f"{label:<26} {str(a == b):>10} {str(bool(np.isclose(a, b))):>10}")

# Change 2: better and worse approximations of sqrt(2)
print("\nChange 2: different approximations of sqrt(2)")
print(f"{'approximation':>14} {'absolute error':>16} {'relative error':>16}")
for approx in [1.4, 1.41, 1.414, 1.41421]:
    abs_err = abs(np.sqrt(2) - approx)
    print(f"{approx:>14} {abs_err:>16.8f} {abs_err / np.sqrt(2):>16.8%}")

# Change 3: more extreme values of x in the unstable formula
print("\nChange 3: more extreme x values (stable formula is the reference)")
print(f"{'x':>8} {'direct':>16} {'stable':>16} {'relative error':>16}")
for x in [1e-4, 1e-8, 1e-12, 1e-15, 1e-16]:
    direct = (np.sqrt(1 + x) - 1) / x
    stable = 1 / (np.sqrt(1 + x) + 1)
    print(f"{x:>8.0e} {direct:>16.10f} {stable:>16.10f} "
          f"{abs(direct - stable) / stable:>16.2e}")


# ---------------------------------------------------------------
# PART 3: RELATED REAL-WORLD CASE
# Banking: adding 10-cent payments
# ---------------------------------------------------------------
print()
print("=" * 60)
print("PART 3: BANKING - ADDING 10-CENT PAYMENTS")
print("=" * 60)

# Assumptions:
#   1. A bank adds up many payments of exactly 0.10 dollars.
#   2. The exact total after n payments is n * 0.10 dollars,
#      which equals n cents * 10 stored as whole numbers (integers).
#   3. Float arithmetic is compared with integer cents, Decimal and
#      math.fsum.
n = 1_000_000
float_total = 0.0
for _ in range(n):
    float_total += 0.1                       # plain float: errors accumulate

cents_total = sum([10] * n) / 100            # integer cents, divide once
decimal_total = sum([Decimal("0.10")] * n)   # exact decimal arithmetic
fsum_total = math.fsum([0.1] * n)            # error-compensated summation
exact_total = 100000.0                       # 1,000,000 x 0.10

print(f"Exact total:          {exact_total:.10f}")
print(f"Plain float sum:      {float_total:.10f}  "
      f"(error {abs(float_total - exact_total):.2e})")
print(f"Integer cents:        {cents_total:.10f}  "
      f"(error {abs(cents_total - exact_total):.2e})")
print(f"Decimal:              {float(decimal_total):.10f}  "
      f"(error {abs(float(decimal_total) - exact_total):.2e})")
print(f"math.fsum:            {fsum_total:.10f}  "
      f"(error {abs(fsum_total - exact_total):.2e})")
print("Plain float equals exact total?", float_total == exact_total)

if float_total != exact_total:
    print("Interpretation: the plain float total is slightly wrong. Banks "
          "store money as integer cents or Decimal to avoid this.")

# Change an input: how does the error grow with the number of payments?
counts = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
print("\nPayments -> error of plain float sum (vs exact n/10):")
exact_errors = []
for count in counts:
    total = 0.0
    for _ in range(count):
        total += 0.1
    exact_errors.append(abs(total - count / 10))
    print(f"{count:>10,} -> {exact_errors[-1]:.3e}")

# Graph: error of plain float sum against number of payments
plt.figure(figsize=(9, 4.5))
plt.loglog(counts, [max(e, 1e-16) for e in exact_errors], marker="o",
           label="Plain float sum")
plt.title("Accumulated Round-Off Error When Adding 0.10 Repeatedly")
plt.xlabel("Number of payments added")
plt.ylabel("Absolute error (dollars)")
plt.grid(alpha=0.3, which="both")
plt.legend()
plt.tight_layout()
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "roundoff_error_plot.png"), dpi=150)
print("\nGraph saved as output/roundoff_error_plot.png")