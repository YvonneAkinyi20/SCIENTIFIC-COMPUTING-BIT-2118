import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# ---------- Save everything printed to output/task_11_output.txt ----------
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

out_file = open(os.path.join(OUT_DIR, "task_11_output.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, out_file)

# ---------- Original problem: IRR of a project ----------
cash_flows = [-10000, 3000, 3500, 4000, 2500]

def npv(rate, cf=cash_flows):
    return sum(c / (1 + rate)**t for t, c in enumerate(cf))

def d_npv(rate, cf=cash_flows):
    return sum(-t * c / (1 + rate)**(t + 1) for t, c in enumerate(cf) if t > 0)

newton = root_scalar(npv, fprime=d_npv, x0=0.10, method="newton", xtol=1e-12)
secant = root_scalar(npv, x0=0.05, x1=0.20, method="secant", xtol=1e-12)

print("=== B. ORIGINAL PROBLEM ===")
print(f"IRR using Newton-Raphson: {newton.root:.6%}")
print(f"IRR using Secant method:  {secant.root:.6%}")
print(f"NPV at Newton solution:   {npv(newton.root):.6f}")
print(f"Newton iterations: {newton.iterations} | Secant iterations: {secant.iterations}")

# ---------- C. Changing the inputs ----------
print("\n=== C. CHANGING THE INPUTS ===")
def solve(cf, x0=0.10):
    n = root_scalar(lambda r: npv(r, cf), fprime=lambda r: d_npv(r, cf),
                    x0=x0, method="newton", xtol=1e-12)
    s = root_scalar(lambda r: npv(r, cf), x0=0.05, x1=0.20,
                    method="secant", xtol=1e-12)
    return n, s

cases = {
    "Baseline":         [-10000, 3000, 3500, 4000, 2500],
    "Last inflow 3500": [-10000, 3000, 3500, 4000, 3500],
    "Invest 12000":     [-12000, 3000, 3500, 4000, 2500],
}
for name, cf in cases.items():
    n, s = solve(cf)
    print(f"{name:<17}: Newton {n.root:.6%} ({n.iterations} it) | "
          f"Secant {s.root:.6%} ({s.iterations} it)")

print()
for x0 in (0.01, 0.10, 0.50, 1.00):
    n, _ = solve(cash_flows, x0=x0)
    print(f"Newton start x0={x0:<4}: root {n.root:.6%} in {n.iterations} iterations")

# ---------- E. Related case: true cost of a loan with an upfront fee ----------
loan = [95000] + [-9200] * 12     # receive 100,000 minus 5% fee, repay 9,200 x 12 months
ln = root_scalar(lambda r: npv(r, loan), fprime=lambda r: d_npv(r, loan),
                 x0=0.02, method="newton", xtol=1e-12)
ls = root_scalar(lambda r: npv(r, loan), x0=0.01, x1=0.05,
                 method="secant", xtol=1e-12)
print("\n=== E. LOAN CASE (KES 100,000, 5% fee, 12 x 9,200) ===")
print(f"Monthly rate (Newton): {ln.root:.6%}  in {ln.iterations} iterations")
print(f"Monthly rate (Secant): {ls.root:.6%}  in {ls.iterations} iterations")
print(f"Nominal APR (x12):     {ln.root*12:.4%}")
print(f"Effective annual rate: {(1 + ln.root)**12 - 1:.4%}")

sys.stdout = sys.__stdout__
out_file.close()

# ---------- Graphs ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

r = np.linspace(-0.05, 0.40, 400)
axes[0].plot(r * 100, [npv(x) for x in r], label="NPV(r)")
axes[0].axhline(0, color="gray", lw=0.8)
axes[0].scatter([newton.root * 100], [0], color="red", zorder=3,
                label=f"IRR = {newton.root:.4%}")
axes[0].set_title("Original problem: NPV vs discount rate")
axes[0].set_xlabel("Discount rate r (%)")
axes[0].set_ylabel("NPV")
axes[0].legend()

r2 = np.linspace(0.001, 0.06, 400)
axes[1].plot(r2 * 100, [npv(x, loan) for x in r2], label="NPV(r)")
axes[1].axhline(0, color="gray", lw=0.8)
axes[1].axvline(ln.root * 100, color="green", ls="--",
                label=f"Root = {ln.root:.3%} per month")
axes[1].set_title("Loan case: NPV vs monthly rate")
axes[1].set_xlabel("Monthly interest rate r (%)")
axes[1].set_ylabel("NPV to borrower (KES)")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "task_11_diagram.png"), dpi=150)
plt.show()