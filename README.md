
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