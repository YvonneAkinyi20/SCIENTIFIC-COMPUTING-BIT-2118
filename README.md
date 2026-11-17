
---

## Task B: Mathematical Functions and Vectorization

**Program file:** `code/Task_03_vectorization.py`
**Output graph:** `output/response_time_histogram.png`

### 1. Mathematical Concept

**Vectorization** means applying a mathematical operation to many values at the same time. If `yᵢ = sin²(xᵢ) + cos²(xᵢ)`, the identity gives `yᵢ = 1` for every `xᵢ`. Instead of looping through `x1, x2, ..., xn`, NumPy evaluates the whole vector in one expression. Vectorized code runs in compiled C code inside NumPy, which is why it is so much faster than a Python `for` loop.

### 2. Example

For `x = [0, π/2, π]`, `np.sin(x)` gives approximately `[0, 1, 0]`. The same operation works whether `x` has 3 values or one million values.

### 3. Real-World Applications

Large datasets, simulations, signal processing, image processing, numerical models, and machine learning, where millions of arithmetic operations are repeated.

### 4. Real-World Problem and Solution Approach

**Problem:** A network-monitoring system has one million latency measurements and needs to convert each from milliseconds to seconds and square the values for an error calculation.

**Solution approach:** Use vectorized NumPy expressions such as `seconds = latency_ms / 1000` and `squared = seconds ** 2`. This is simpler and usually much faster than Python loops.

### 5. Solution Code

See `code/Task_03_vectorization.py`. It contains both a loop version and a vectorized version, times them with `timeit`, and checks that they give the same answer.

**Output:**

```
First five squared latency values: [0.03787665 0.01266192 0.04637842 0.03092504 0.00078812]
Loop time: 0.1921 s
Vectorized time: 0.0067 s
Approximate speed-up: 28.5x
Loop and vectorized results match? True
```

**Interpretation:** Both methods give identical results, but the vectorized version is about 28 times faster on one million values (the exact speed-up depends on the computer).

### 6. Changing the Inputs and Observing the Results

| Change made | Result |
|-------------|--------|
| 100 values | Speed-up about 6x (too little data for vectorization to matter much) |
| 10,000 values | Speed-up about 85x |
| 1,000,000 values | Speed-up about 45x |
| Slow network (100 to 900 ms) | Mean squared latency rises from 0.0213 s² to 0.3032 s² |
| One outlier of 10,000 ms | Maximum squared value jumps from 0.0625 s² to 100 s², but the mean barely changes (0.021259 to 0.021359 s²) |
| Constant 100 ms | Every squared value is 0.01 s² |

**Observations:**

- Vectorization helps most on larger datasets. The speed-up varies between runs, but it is always large.
- Squaring magnifies large values, so a single outlier dominates the maximum squared error.
- Because squaring is a nonlinear operation, doubling the latency makes the squared value four times larger.

### 7. Related Real-World Case: Web-Server Response-Time Monitoring

**Scenario:** A web server logs the response time of one million requests. The target is 120 ms, and any response slower than 200 ms breaks the service agreement (SLA).

**Assumptions:**

1. The server logs one million response times in milliseconds.
2. The target response time is 120 ms.
3. Responses over 200 ms break the SLA.
4. Most requests are fast and a few are slow (modelled with a lognormal distribution).

**Approach:** Compute the error from the target, square it, and take the root mean squared error (RMSE), all as vectorized expressions. Use a boolean array (`response_ms > SLA_MS`) to count the slow requests.

**Results:**

```
Mean response time: 116.98 ms
Median response time: 110.08 ms
95th percentile: 195.59 ms
Root mean squared error vs 120 ms target: 42.30 ms
Requests slower than 200 ms: 43,665 (4.37%)
```

**Interpretation:** Only 4.37% of requests exceed 200 ms, so the server is performing acceptably. With a stricter limit of 150 ms, 18.83% of requests would be too slow.

![Web-server response times](output/response_time_histogram.png)

### 8. How to Run

```bash
pip install numpy matplotlib
python code/Task_03_vectorization.py
```


---

## Task C: Floating-Point Arithmetic and Numerical Errors

**Program file:** `code/Task_04_numerical_errors.py`
**Output graph:** `output/roundoff_error_plot.png`

### 1. Mathematical Concept

Computers store most real numbers with a finite number of binary digits, so some values are only approximations.

- **Absolute error** = |true value − approximate value|
- **Relative error** = absolute error / |true value|
- **Truncation error** comes from replacing an exact mathematical process with an approximation.
- **Round-off error** comes from finite computer precision.

### 2. Example

If √2 = 1.41421356 but we use 1.414, the absolute error is about 0.00021356. The relative error is about 0.000151, or 0.0151%.

### 3. Real-World Applications

Scientific simulations, banking calculations, GPS/navigation, engineering design, iterative algorithms, and any system where accumulated numerical error can affect a result.

### 4. Real-World Problem and Solution Approach

**Problem:** A program compares `0.1 + 0.2` with `0.3` and unexpectedly reports `False`.

**Solution approach:** This happens because binary floating-point representation is approximate. Compare using `np.isclose()` or a suitable tolerance instead of exact equality.

### 5. Solution Code

See `code/Task_04_numerical_errors.py`. It shows the failed exact comparison, the tolerance-based fix, the absolute and relative error of an approximation of √2, and an unstable formula next to a stable rewrite of it.

**Output:**

```
0.1 + 0.2 = 0.30000000000000004
Exact comparison: False
Tolerance-based comparison: True
Absolute error: 0.00021356
Relative error: 0.01510114%

x=1e-04: direct=0.499987500624, stable=0.499987500625
x=1e-08: direct=0.499999996961, stable=0.499999998750
x=1e-12: direct=0.500044450291, stable=0.500000000000
```

**Interpretation:** `0.1 + 0.2` is stored as `0.30000000000000004`, so exact equality fails but `np.isclose()` succeeds. The direct formula `(√(1+x) − 1)/x` loses accuracy as `x` shrinks because it subtracts two nearly equal numbers (catastrophic cancellation). The algebraically equivalent form `1/(√(1+x) + 1)` avoids the subtraction and stays accurate.

### 6. Changing the Inputs and Observing the Results

**Other comparisons:**

| Test | Exact `==` | `np.isclose` |
|------|-----------|--------------|
| 0.5 + 0.25 == 0.75 | True | True |
| 0.1 * 3 == 0.3 | False | True |
| ten 0.1s added == 1.0 | False | True |
| 0.1 + 0.7 == 0.8 | False | True |

**Different approximations of √2:**

| Approximation | Absolute error | Relative error |
|---------------|----------------|----------------|
| 1.4 | 0.01421356 | 1.0051% |
| 1.41 | 0.00421356 | 0.2979% |
| 1.414 | 0.00021356 | 0.0151% |
| 1.41421 | 0.00000356 | 0.0003% |

**Unstable formula at more extreme `x`:**

| x | Relative error of direct formula |
|---|----------------------------------|
| 1e-04 | 2.0e-12 |
| 1e-08 | 3.6e-09 |
| 1e-12 | 8.9e-05 |
| 1e-15 | 1.1e-01 (11%) |
| 1e-16 | 1.0 (100%, the answer is 0) |

**Observations:**

- Numbers that are exact in binary (0.5, 0.25, 0.75) compare correctly. Numbers such as 0.1, 0.3 and 0.7 cannot be stored exactly, so exact equality is unreliable for them.
- Each extra correct digit in an approximation reduces the error, but it never reaches zero.
- The direct formula gets worse and worse as `x` shrinks, until at `x = 1e-16` it returns 0 because `1 + x` rounds to exactly 1. The stable formula is unaffected.

### 7. Related Real-World Case: Banking, Adding 10-Cent Payments

**Scenario:** A bank adds up one million payments of exactly $0.10. The exact total is $100,000.00.

**Assumptions:**

1. Every payment is exactly $0.10.
2. The exact total is n × 0.10.
3. Plain floats are compared with integer cents, `Decimal` and `math.fsum`.

**Approach:** Add the payments four different ways and compare each total with the exact value.

**Results:**

```
Plain float sum:      100000.0000013329  (error 1.33e-06)
Integer cents:        100000.0000000000  (error 0.00e+00)
Decimal:              100000.0000000000  (error 0.00e+00)
math.fsum:            100000.0000000000  (error 0.00e+00)
```

**Error of the plain float sum as the number of payments grows:**

| Payments | Error |
|----------|-------|
| 10 | 1.1e-16 |
| 1,000 | 1.4e-12 |
| 100,000 | 1.9e-08 |
| 1,000,000 | 1.3e-06 |

**Interpretation:** A tiny error in each addition accumulates, so the plain float total is wrong by about 0.0000013 dollars after a million payments. This is small, but a bank cannot accept it, and equality checks on the total would fail. Banks avoid it by storing money as integer cents or `Decimal`.

![Accumulated round-off error](output/roundoff_error_plot.png)

### 8. How to Run

```bash
pip install numpy matplotlib
python code/Task_04_numerical_errors.py
```