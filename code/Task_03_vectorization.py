"""
BIT 2118 Scientific Computing - Task B: Mathematical Functions and Vectorization

Part 1: Original problem (network latency, ms -> seconds -> squared)
Part 2: Changing the inputs and observing the results
Part 3: Related real-world case: web-server response-time monitoring
"""

import os
import timeit
import numpy as np
import matplotlib
matplotlib.use("Agg")  # lets the script save the plot without opening a window
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# PART 0: MATHEMATICAL CONCEPT CHECK
# sin^2(x) + cos^2(x) = 1 for every x, evaluated on a whole vector at once
# ---------------------------------------------------------------
print("=" * 60)
print("PART 0: VECTORIZED MATHEMATICAL FUNCTIONS")
print("=" * 60)

x = np.array([0, np.pi / 2, np.pi])
print("sin(x) for x = [0, pi/2, pi]:", np.round(np.sin(x), 4))

x_big = np.linspace(0, 2 * np.pi, 1_000_000)
y = np.sin(x_big) ** 2 + np.cos(x_big) ** 2
print("sin^2 + cos^2 over 1,000,000 values -> min:", y.min(),
      " max:", y.max())
print("All values equal to 1 (within rounding)?", np.allclose(y, 1.0))


# ---------------------------------------------------------------
# PART 1: ORIGINAL PROBLEM
# ---------------------------------------------------------------
print()
print("=" * 60)
print("PART 1: NETWORK LATENCY (original problem)")
print("=" * 60)

# Simulated network latency readings in milliseconds
rng = np.random.default_rng(42)
latency_ms = rng.uniform(5, 250, 1_000_000)


def loop_version(data):
    squared = []
    for value in data:
        seconds = value / 1000.0
        squared.append(seconds ** 2)
    return squared


def vectorized_version(data):
    seconds = data / 1000.0
    return seconds ** 2


loop_time = timeit.timeit(lambda: loop_version(latency_ms), number=1)
vector_time = timeit.timeit(lambda: vectorized_version(latency_ms), number=3) / 3

result = vectorized_version(latency_ms)
print("First five squared latency values:", result[:5])
print(f"Loop time: {loop_time:.4f} s")
print(f"Vectorized time: {vector_time:.4f} s")
print(f"Approximate speed-up: {loop_time / vector_time:.1f}x")

# Check that both methods give the same answer
print("Loop and vectorized results match?",
      np.allclose(loop_version(latency_ms), result))


# ---------------------------------------------------------------
# PART 2: CHANGING THE INPUTS
# ---------------------------------------------------------------
print()
print("=" * 60)
print("PART 2: CHANGING THE INPUT VALUES")
print("=" * 60)

# Change 1: different data sizes - how does the speed-up change?
print("Change 1: number of measurements")
print(f"{'n':>12} {'loop (s)':>12} {'vector (s)':>12} {'speed-up':>10}")
for n in [100, 10_000, 1_000_000]:
    data = rng.uniform(5, 250, n)
    t_loop = timeit.timeit(lambda: loop_version(data), number=1)
    t_vec = timeit.timeit(lambda: vectorized_version(data), number=5) / 5
    print(f"{n:>12,} {t_loop:>12.6f} {t_vec:>12.6f} {t_loop / t_vec:>9.1f}x")

# Change 2: different latency range (slow network: 100-900 ms)
slow_ms = rng.uniform(100, 900, 1_000_000)
slow_sq = vectorized_version(slow_ms)
print("\nChange 2: slow network (100-900 ms)")
print(f"  Mean squared latency: {slow_sq.mean():.4f} s^2 "
      f"(original: {result.mean():.4f} s^2)")

# Change 3: add one extreme outlier (10,000 ms = 10 s)
with_outlier = latency_ms.copy()
with_outlier[0] = 10_000.0
out_sq = vectorized_version(with_outlier)
print("\nChange 3: one outlier of 10,000 ms")
print(f"  Max squared value: {out_sq.max():.4f} s^2 "
      f"(original: {result.max():.4f} s^2)")
print(f"  Mean squared value: {out_sq.mean():.6f} s^2 "
      f"(original: {result.mean():.6f} s^2)")

# Change 4: constant latency of 100 ms
const_sq = vectorized_version(np.full(5, 100.0))
print("\nChange 4: constant 100 ms")
print("  Squared values (s^2):", const_sq)


# ---------------------------------------------------------------
# PART 3: RELATED REAL-WORLD CASE
# Web-server response-time monitoring
# ---------------------------------------------------------------
print()
print("=" * 60)
print("PART 3: WEB-SERVER RESPONSE-TIME MONITORING")
print("=" * 60)

# Assumptions:
#   1. The server logs the response time of one million requests (ms).
#   2. The target response time is 120 ms.
#   3. Responses slower than 200 ms break the service agreement (SLA).
#   4. Most requests are fast; a few are slow (modelled with a
#      lognormal distribution).
TARGET_MS = 120.0
SLA_MS = 200.0

response_ms = rng.lognormal(mean=np.log(110), sigma=0.35, size=1_000_000)

errors_s = (response_ms - TARGET_MS) / 1000.0     # error in seconds
squared_error = errors_s ** 2                     # squared error (s^2)
rmse_ms = np.sqrt(squared_error.mean()) * 1000    # root mean squared error

over_sla = response_ms > SLA_MS                   # boolean array
percent_over = 100 * np.mean(over_sla)

print(f"Mean response time: {response_ms.mean():.2f} ms")
print(f"Median response time: {np.median(response_ms):.2f} ms")
print(f"95th percentile: {np.percentile(response_ms, 95):.2f} ms")
print(f"Root mean squared error vs {TARGET_MS:.0f} ms target: {rmse_ms:.2f} ms")
print(f"Requests slower than {SLA_MS:.0f} ms: {int(over_sla.sum()):,} "
      f"({percent_over:.2f}%)")

if percent_over > 5:
    print("Interpretation: too many slow requests; investigate the server.")
else:
    print("Interpretation: most requests meet the SLA; the server is "
          "performing acceptably.")

# Change an input: what if the SLA were stricter (150 ms)?
strict_ms = 150.0
print(f"\nWith a stricter limit of {strict_ms:.0f} ms, requests above the "
      f"limit: {100 * np.mean(response_ms > strict_ms):.2f}%")

# Graph: histogram of response times with target and SLA lines
plt.figure(figsize=(9, 4.5))
plt.hist(response_ms, bins=100, range=(0, 400), color="steelblue", alpha=0.8)
plt.axvline(TARGET_MS, color="green", linestyle="--",
            label=f"Target ({TARGET_MS:.0f} ms)")
plt.axvline(SLA_MS, color="red", linestyle="--",
            label=f"SLA limit ({SLA_MS:.0f} ms)")
plt.title("Web-Server Response Times (1,000,000 requests)")
plt.xlabel("Response time (ms)")
plt.ylabel("Number of requests")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "response_time_histogram.png"), dpi=150)
print("\nGraph saved as output/response_time_histogram.png")