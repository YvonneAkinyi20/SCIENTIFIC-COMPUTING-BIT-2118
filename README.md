
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

## Task D: Matrix Algebra and Matrix Operations

**Program file:** `code/Task_05_matrix_algebra.py`
**Output graph:** `output/task_05_diagram.png`

### 1. Mathematical Concept

A matrix organizes values in rows and columns. Matrix addition combines corresponding elements. Matrix multiplication combines rows of the first matrix with columns of the second: `cᵢⱼ = Σ aᵢₖ bₖⱼ`. The determinant indicates whether a square matrix is invertible (`det = 0` means it is not). The inverse `A⁻¹` satisfies `A A⁻¹ = I`. For a 2-D transformation, `|det|` is also the factor by which areas are multiplied.

### 2. Example

For `A = [[1,2],[3,4]]` and `x = [5,6]`: `Ax = [1×5 + 2×6, 3×5 + 4×6] = [17, 39]`. Also `det(A) = 1×4 − 2×3 = −2`, so `A` is invertible.

### 3. Real-World Applications

Computer graphics transformations, machine learning, cryptography, network models, robotics, electrical circuits and solving systems of equations.

### 4. Real-World Problem and Solution Approach

**Problem:** A graphics program needs to rotate and scale a set of 2-D coordinates representing points on an object.

**Solution approach:** Store the points as coordinate vectors, build a rotation matrix and a scaling matrix, combine them (`T = R @ S`), and multiply every point by `T` so all points are transformed consistently. With points stored as rows, use `points @ T.T`.

### 5. Solution Code

See `code/Task_05_matrix_algebra.py`. It checks the matrix concepts, transforms the four corners of a square (rotate 45°, scale by 2 and 1.5), tests changed inputs, solves an electrical-circuit case, and draws one summary diagram.

**Output:**

```
Part 0: A @ x = [17. 39.]   det(A) = -2.0   A @ inverse(A) = identity
Transformation matrix (rotation @ scale):
[[ 1.4142 -1.0607]
 [ 1.4142  1.0607]]
Transformed points:
[[ 0.      0.    ]
 [ 1.4142  1.4142]
 [ 0.3536  2.4749]
 [-1.0607  1.0607]]
Determinant: 3.0000
```

![Task 05 summary](output/task_05_diagram.png)

**Interpretation:** Every point of the square is rotated by 45° and stretched by 2 in one direction and 1.5 in the other. The determinant is 3 (= 2 × 1.5 × 1), so the transformed shape has three times the area of the unit square. Rotation alone never changes area (its determinant is 1), so the area change comes from the scaling.

### 6. Changing the Inputs and Observing the Results

| Change made | Determinant | Result |
|---|---|---|
| Angle 45° → 90° | 3 | Same area. The square turns to a different orientation: corners `(0,0), (0,2), (−1.5,2), (−1.5,0)`. |
| Scale (1, 1), pure rotation | 1 | Same size, only rotated. |
| Scale (2, −1.5) | −3 | The area still grows by 3, but the shape is a mirror image (reflection). |
| Order swapped (`S @ R`) | 3 | Same area but a different shape and position. Matrix multiplication is not commutative. |
| Scale (2, 0) | 0 | The square collapses to a line and the matrix cannot be inverted. |

**Observations:**

- The sign of the determinant shows whether the shape is flipped, and `det = 0` shows the transformation destroys information.
- Order matters: `R @ S ≠ S @ R`.
- When `det ≠ 0`, the inverse matrix undoes the transformation exactly (maximum recovery error about 5×10⁻¹⁷).

### 7. Related Real-World Case: Electrical Circuit (Mesh Currents)

**Scenario:** A circuit with three loops has the resistances and sources below. Mesh-current analysis gives a linear system `R I = V`, which is solved with the same matrix tools.

**Assumptions:**

- Three independent loops with currents I₁, I₂, I₃ in amperes.
- Resistance matrix `R = [[30, −20, 0], [−20, 50, −10], [0, −10, 30]]` ohms.
- Source voltages `V = [10, 0, −5]` volts.

**Results:**

| Case | I₁ | I₂ | I₃ |
|---|---|---|---|
| Original | 0.4333 A | 0.1500 A | −0.1167 A |
| First source 10 V → 20 V | 0.9000 A | 0.3500 A | −0.0500 A |
| R[1,1] 30 → 60 ohm | 0.1806 A | 0.0417 A | −0.1528 A |

**Interpretation:** `det(R) = 30000`, which is non-zero, so there is one unique set of loop currents, and the residual is about 10⁻¹⁵. Doubling the first source raises all the currents, since the loops share resistors. Raising the loop-1 resistance reduces the current in loop 1 most. A negative current means the current flows opposite to the direction chosen for that loop.

### 8. How to Run

```
pip install numpy matplotlib
python code/Task_05_matrix_algebra.py
```

## Task E: Solution of Systems of Linear Equations

**Program file:** `code/Task_06_linear_systems.py`
**Output graph:** `output/task_06_diagram.png`

### 1. Mathematical Concept

Many problems can be written as `Ax = b`, where `A` contains the coefficients, `x` contains the unknowns and `b` contains the known results. Solving the system means finding `x` so that multiplying `A` by `x` reproduces `b`. A unique solution exists when `det(A) ≠ 0`. With two unknowns, each equation is a straight line and the solution is the point where the lines cross.

### 2. Example

`2x + 3y = 8` and `4x + y = 6` become `A = [[2,3],[4,1]]` and `b = [8,6]`. The solution is `x = 1`, `y = 2`, because `2(1) + 3(2) = 8` and `4(1) + 2 = 6`.

### 3. Real-World Applications

Electrical circuit analysis, resource allocation, traffic-flow models, balancing chemical equations, computer graphics and estimating unknown parameters.

### 4. Real-World Problem and Solution Approach

**Problem:** Two server types consume different amounts of CPU and memory. Given the total CPU and memory usage, determine how many servers of each type are active.

**Solution approach:** Let `x` be the number of Type A servers and `y` the number of Type B servers. Form two simultaneous equations from the resource totals and solve `Ax = b` with `np.linalg.solve()`. Verify with `A @ x` and the residual `b − A @ x`.

| | CPU units | Memory (GB) |
|---|---|---|
| Type A | 4 | 8 |
| Type B | 2 | 12 |
| Observed total | 20 | 72 |

`4x + 2y = 20` and `8x + 12y = 72`

### 5. Solution Code

See `code/Task_06_linear_systems.py`. It solves the textbook example and the server problem, tests variations of the input values, handles singular systems, flags answers that are not realistic (fractional or negative server counts), and draws one summary diagram.

**Output:**

```
Type A servers: 3
Type B servers: 4
Verification A @ solution: [20. 72.]
Residual: [0. 0.]
Residual norm: 0.0
```

![Task 06 summary](output/task_06_diagram.png)

**Interpretation:** There are 3 Type A servers and 4 Type B servers. Check: CPU = 4(3) + 2(4) = 20 and memory = 8(3) + 12(4) = 72. The residual is zero, so the solution reproduces the observed totals exactly. On the graph, the two lines cross at the point (3, 4).

### 6. Changing the Inputs and Observing the Results

| Variation | Input change | Type A | Type B | Status |
|---|---|---|---|---|
| 1. Original | `b = [20, 72]` | 3 | 4 | Valid |
| 2. Different coefficients | `A = [[3,1],[2,5]]`, `b = [14, 24]` | 3.5385 | 3.3846 | Solved, but not whole |
| 3. Fractional answer | `b = [21, 70]` | 3.5 | 3.5 | Solved, but not whole |
| 4. Negative answer | `b = [2, −8]` | 1.25 | −1.5 | Solved, but not realistic |
| 5. Singular | `A = [[4,2],[8,4]]`, `b = [20, 50]` | none | none | No unique solution |

**Observations:**

- A system can be solved mathematically and still not make sense in the real world. Server counts must be whole, non-negative numbers, so variations 2, 3 and 4 are rejected even though `np.linalg.solve()` returns an answer for them.
- In variation 5, the second row of `A` is exactly twice the first row, so `det(A) = 0`. The two lines are parallel, never cross, and `np.linalg.solve()` raises an error, which the program catches.
- The original `A` has `det = 32`, which is non-zero, so any `b` gives a unique answer. Only the interpretation of that answer can fail.

### 7. Related Real-World Case: Three Server Types (CPU, Memory and Power)

**Scenario:** A data centre runs three server types and records the total CPU, memory and power use. The task is to find how many of each type are active.

**Assumptions:**

- `x`, `y` and `z` are the numbers of Type A, B and C servers.
- Each type uses fixed amounts of CPU, memory and power.
- The observed totals are 22 CPU units, 80 GB memory and 21 power units.

| | CPU | Memory (GB) | Power |
|---|---|---|---|
| Type A | 4 | 8 | 3 |
| Type B | 2 | 12 | 2 |
| Type C | 1 | 4 | 2 |
| Observed total | 22 | 80 | 21 |

**System

## Task F: Gaussian Elimination and LU Decomposition

**Program file:** `code/Task_07_gaussian_lu.py`
**Output graph:** `output/task_07_diagram.png`

### 1. Mathematical Concept

Gaussian elimination transforms `Ax = b` into an equivalent upper-triangular system by removing variables one row at a time, then solves it by back-substitution. LU decomposition writes `PA = LU`, where `L` is lower triangular, `U` is upper triangular and `P` records any row swaps (pivoting). Once `A` is factored, every new `b` is solved in two cheap steps: forward substitution `Ly = Pb`, then back substitution `Ux = y`.

### 2. Example

For `A = [[4,-1,0],[-1,4,-1],[0,-1,3]]` and `b = [15,10,10]`:

- Row 2 ← Row 2 + 0.25 × Row 1 removes x from row 2 and gives `3.75y − z = 13.75`.
- Row 3 ← Row 3 + 0.2667 × Row 2 removes y from row 3 and gives `2.7333z = 13.6667`.
- The last equation gives `z = 5`. Back-substitution gives `y = 5` and `x = 5`.

The multipliers (0.25 and 0.2667) are the entries of `L`, and the reduced matrix is `U`.

### 3. Real-World Applications

Engineering simulations, repeated linear-system solutions, finite-element methods, economic models and scientific models with many equations.

### 4. Real-World Problem and Solution Approach

**Problem:** A simulation uses the same coefficient matrix `A` every second but receives a new measurement vector `b`.

**Solution approach:** Factor `A` once with `lu_factor`. Reuse the factors with `lu_solve` for each new `b`, instead of repeating the full elimination every time.

### 5. Solution Code

See `code/Task_07_gaussian_lu.py`. It factors `A` once, solves three measurement sets, checks the residuals, compares with `np.linalg.solve`, tests changed inputs, solves a heated-rod case, times repeated solves, and draws one summary diagram.

**Output:**

```
L = [[ 1.      0.      0.    ]
     [-0.25    1.      0.    ]
     [ 0.     -0.2667  1.    ]]
U = [[ 4.     -1.      0.    ]
     [ 0.      3.75   -1.    ]
     [ 0.      0.      2.7333]]
Set 1: b = [15. 10. 10.] -> x = [5.     5.     5.    ]  (residual 0)
Set 2: b = [16. 11.  9.] -> x = [5.3171 5.2683 4.7561] (residual 1.8e-15)
Set 3: b = [14. 12. 11.] -> x = [4.9024 5.6098 5.5366] (residual 0)
```

![Task 07 summary](output/task_07_diagram.png)

**Interpretation:** Each solution satisfies all

## Task G: Iterative Methods for Linear Systems

**Program file:** `code/Task_08_iterative_linear.py`
**Output graph:** `output/task_08_diagram.png`

### 1. Mathematical Concept

Jacobi and Gauss-Seidel methods start with an initial guess and repeatedly improve it. For each equation, solve for one unknown using the current estimates of the others. Jacobi uses only the old estimates in a sweep, while Gauss-Seidel uses each new value immediately. The process stops when the change between successive estimates is below a chosen tolerance. The methods are guaranteed to converge when the matrix is strictly diagonally dominant.

### 2. Example

If successive estimates of x are 1.0, 1.8, 1.97, 1.999 and 2.000, the values are converging toward 2.

### 3. Real-World Applications

Large sparse systems from heat transfer, electrical networks, structural models, image reconstruction and scientific simulations where direct matrix factorization may be expensive.

### 4. Real-World Problem and Solution Approach

**Problem:** A grid model of temperatures produces thousands of linear equations, most of whose coefficients are zero.

**Solution approach:** Use an iterative method such as Jacobi or Gauss-Seidel. Track the error in each iteration (the maximum change in any unknown) and stop once it falls below the tolerance.

### 5. Solution Code

See `code/Task_08_iterative_linear.py`. It solves a 5-point temperature model with both methods, compares them with the direct solution, tests changed inputs, solves a 2-D plate case, and draws one summary diagram.

**Model:** five interior grid points, left boundary 100 °C, right boundary 20 °C, `A = tridiag(−1, 2, −1)` and `b = [100, 0, 0, 0, 20]`.

**Output:**

```
Jacobi temperatures       : [86.6667 73.3333 60. 46.6667 33.3333]   (150 iterations)
Gauss-Seidel temperatures : [86.6667 73.3333 60. 46.6667 33.3333]   (75 iterations)
Direct solution           : [86.6667 73.3333 60. 46.6667 33.3333]
Max difference Jacobi vs direct       : 3.41e-08
Max difference Gauss-Seidel vs direct : 2.89e-08
```

![Task 08 summary](output/task_08_diagram.png)

**Interpretation:** Both methods reach the direct solution, and the temperature falls in a straight line from 100 °C to 20 °C across the grid. The error drops steadily at each iteration. Gauss-Seidel needs 75 iterations while Jacobi needs 150, about half as many, because it reuses the newest values straight away. The remaining difference of about 10⁻⁸ equals the tolerance.

### 6. Changing the Inputs and Observing the Results

| Change made | Result |
|---|---|
| Tolerance 10⁻² → 10⁻⁴ → 10⁻⁸ → 10⁻¹² | Jacobi: 54, 86, 150, 214 iterations. Gauss-Seidel: 27, 43, 75, 107. |
| Starting guess all zeros → all 60 | Jacobi needs 32 iterations instead of 150. Starting at the exact answer needs 1. |
| Boundaries 100/20 → 200/50 | Gauss-Seidel gives `[175, 150, 125, 100, 75]` in 78 iterations. |
| Grid of 5, 10, 20, 40 points | Jacobi: 150, 486, 1672, 5935 iterations. Gauss-Seidel: 75, 245, 842, 2993. |
| `A = [[1,2],[3,1]]`, `b = [3,4]` (not diagonally dominant) | Both methods diverge. The change grows from 144 to 11 664 to 1 119 744 (Jacobi, at iterations 5, 10 and 15). The direct solution `[1, 1]` works fine. |

**Observations:**

- A stricter tolerance costs more iterations, but only by a roughly constant amount for each extra digit of accuracy.
- A good starting guess saves work.
- Bigger grids converge more slowly, so very large problems need many iterations.
- Gauss-Seidel needs about half the iterations of Jacobi in every test.
- Diagonal dominance matters. Without it, an iterative method can blow up, even though the system itself has a solution.

### 7. Related Real-World Case: Temperature of a 2-D Heated Metal Plate

**Scenario:** A square metal plate has its top edge held at 100 °C and its other three edges at 20 °C. We want the steady-state temperature at every interior point. Each point's temperature is the average of its four neighbours, which gives one equation per grid point.

**Assumptions:**

- Steady state, no heat sources inside the plate, uniform material.
- 20 × 20 interior grid, so 400 unknowns and a 400 × 400 matrix.
- The 5-point Laplace stencil (each row has a 4 on the diagonal and four −1 entries), so about 98.8% of the coefficients are zero.
- Tolerance 10⁻⁶.

**Results:**

| Method | Iterations | Max error vs direct solution |
|---|---|---|
| Jacobi | 1202 | 8.8 × 10⁻⁵ |
| Gauss-Seidel | 631 | 4.4 × 10⁻⁵ |

The hottest interior point is 92.30 °C, next to the 100 °C edge, and the centre of the plate is 38.42 °C.

**Interpretation:** The heatmap shows heat spreading from the hot top edge, with temperatures dropping toward the cooler edges. Because the matrix is almost entirely zeros, each iteration only needs the four neighbours of each point. For plates with thousands or millions of points, this is much cheaper than a direct factorisation, which is why iterative methods are used for large sparse systems. Gauss-Seidel again needs about half the iterations of Jacobi.

### 8. How to Run

```
pip install numpy matplotlib
python code/Task_08_iterative_linear.py
```

## Task H: Eigenvalues and Eigenvectors

**Program file:** `code/Task_09_eigenvalues.py`
**Output graph:** `output/task_09_diagram.png`

### 1. Mathematical Concept

For a square matrix `A`, an eigenvector `v` is a non-zero vector whose direction is unchanged by the transformation `A`. It satisfies `Av = λv`, where `λ` is the eigenvalue. The eigenvalue tells how much the vector is stretched, compressed or reversed. For a connected network, the eigenvector of the largest eigenvalue (the dominant eigenvector) has all positive entries and can be used as a measure of node importance.

### 2. Example

If `Av = 3v`, then `v` is an eigenvector and 3 is its eigenvalue. Applying `A` triples the magnitude of `v` without changing its direction. For `A = [[2,1],[1,2]]` and `v = [1,1]`: `Av = [3,3] = 3v`.

### 3. Real-World Applications

Google PageRank concepts, vibration analysis, stability of dynamical systems, principal component analysis, image compression, quantum models and network centrality.

### 4. Real-World Problem and Solution Approach

**Problem:** A small computer network needs to identify which nodes are structurally most influential from their pattern of connections.

**Solution approach:** Represent the network as an adjacency matrix, compute its dominant eigenvalue and eigenvector with `np.linalg.eig`, and normalize the dominant eigenvector so the scores add up to 1. This gives an eigenvector-based node-importance measure. Verify the result with `Av = λv`.

### 5. Solution Code

See `code/Task_09_eigenvalues.py`. It checks the eigenvector idea on a small matrix, computes the importance of each node, tests changed networks, solves a building-vibration case, and draws one summary diagram.

**Network:** nodes A, B, C, D with links A–B, A–C, B–C, B–D and C–D. A and D have 2 links each, and B and C have 3 each.

**Output:**

```
All eigenvalues: [ 2.5616 -0.     -1.     -1.5616]
Dominant eigenvalue: 2.5616
Eigenvector-based importance:
   A: 0.2192   (number of links = 2)
   B: 0.2808   (number of links = 3)
   C: 0.2808   (number of links = 3)
   D: 0.2192   (number of links = 2)
Verification error ||A v - lambda v||: 1.51e-15
```

![Task 09 summary](output/task_09_diagram.png)

**Interpretation:** The dominant eigenvalue is 2.5616. B and C are the most influential nodes (0.2808 each), and A and D are less influential (0.2192 each), which matches the network, where B and C have the most links. The importance score is not just the link count: a node scores higher when its neighbours are themselves well connected. The verification error is about 10⁻¹⁵, so `Av = λv` holds to machine precision. The graph also shows that power iteration (repeatedly multiplying by `A`) finds the same vector without calling `eig`.

### 6. Changing the Inputs and Observing the Results

| Change made | Dominant eigenvalue | Importance (A, B, C, D) | Result |
|---|---|---|---|
| Original network | 2.5616 | 0.219, 0.281, 0.281, 0.219 | B and C lead. |
| Add link A–D | 3.0000 | 0.25 each | Every node now has 3 links, so all are equal. |
| Remove link B–C | 2.0000 | 0.25 each | The network is a 4-node ring, so all are equal. |
| Star centred on B | 1.7321 | 0.211, 0.366, 0.211, 0.211 | B is clearly the most influential. |
| Link B–C weight 3 | 4.0000 | 0.167, 0.333, 0.333, 0.167 | B and C gain even more importance. |
| Disconnected (A–B and C–D only) | 1 (repeated) | not meaningful | The eigenvector is not unique. |

**Observations:**

- Adding links raises the dominant eigenvalue (2.56 → 3.00), and removing links lowers it (2.56 → 2.00).
- When every node is equivalent, the scores are equal.
- In a star, the hub gets the highest score.
- For a disconnected network the dominant eigenvalue is repeated, so the importance scores are not defined.
- Eigenvectors can come out with either sign, so the code flips the sign and uses absolute values.

### 7. Related Real-World Case: Vibration of a 3-Storey Building

**Scenario:** A structural engineer needs the natural frequencies and vibration shapes of a three-storey building, to check them against earthquake or wind frequencies. These come from an eigenvalue problem `(K/m) v = ω² v`.

**Assumptions:**

- Three floors, each with a mass of 10 000 kg, connected by springs of stiffness 2 × 10⁶ N/m.
- Linear behaviour, sideways motion only, no damping.
- Stiffness matrix `K = 2×10⁶ × [[2,−1,0],[−1,2,−1],[0,−1,1]]`.
- The eigenvalues of `K/m` are ω² (in rad²/s²), and the frequency is `f = ω / 2π`.

**Results:**

| Mode | ω² | Frequency | Shape (floors 1, 2, 3) |
|---|---|---|---|
| 1 | 39.61 | 1.002 Hz | 0.328, 0.591, 0.737 |
| 2 | 310.99 | 2.807 Hz | −0.737, −0.328, 0.591 |
| 3 | 649.40 | 4.056 Hz | 0.591, −0.737, 0.328 |

Doubling the stiffness multiplies every frequency by √2 (1.002 → 1.417 Hz for mode 1). Doubling the floor mass divides every frequency by √2 (1.002 → 0.708 Hz).

**Interpretation:** The lowest mode (1.00 Hz) has all floors moving the same way, with the top floor moving most. This is the mode most easily excited by wind or earthquakes. Higher modes have floors moving in opposite directions and vibrate faster. A stiffer building vibrates faster and a heavier building vibrates slower, which is why engineers change stiffness or mass to move the natural frequencies away from dangerous forcing frequencies.

### 8. How to Run

```
pip install numpy matplotlib
python code/Task_09_eigenvalues.py
```

## Task I: Root Finding: Bisection and Fixed-Point Methods

**Program file:** `code/Task_10_bisection_fixed_point.py`
**Output graph:** `output/task_10_diagram.png`

### 1. Mathematical Concept

A root of `f(x)` is a value `r` for which `f(r) = 0`. Bisection starts with an interval `[a, b]` where `f(a)` and `f(b)` have opposite signs, repeatedly halves the interval and keeps the half that contains a root. Fixed-point iteration rewrites the equation as `x = g(x)` and repeatedly calculates `xₙ₊₁ = g(xₙ)`. Bisection always converges once a root is bracketed, but slowly. Fixed-point iteration is faster when it converges, but only if `|g′(x)| < 1` near the root.

### 2. Example

For `f(x) = x³ − x − 2`, `f(1) = −2` and `f(2) = 4`, so a root lies between 1 and 2. The first midpoints are 1.5, 1.75, 1.625, 1.5625, ... and they close in on the root 1.521380.

### 3. Real-World Applications

Finding break-even points, solving engineering equations, computing interest rates, calibration, physical equilibrium and threshold values.

### 4. Real-World Problem and Solution Approach

**Problem:** Determine the temperature at which a nonlinear heat-balance equation becomes zero.

**Solution approach:** Define the heat-balance function `f(T) = T − 25 − 10·exp(−T/20)`, bracket a sign change in `[20, 40]` and apply the bisection method. Stop when the interval or the function value is sufficiently small. As a check, rewrite the equation as `T = 25 + 10·exp(−T/20)` and use fixed-point iteration.

### 5. Solution Code

See `code/Task_10_bisection_fixed_point.py`. It writes out both methods, compares them with `scipy`, tests changed inputs, solves a break-even case, and draws one summary diagram.

**Output:**

```
f(20) = -8.6788, f(40) = 13.6466  -> sign change in [20, 40]
Bisection (own code)     : 27.525211 C   (38 iterations)
Bisection (scipy)        : 27.525211 C   (38 iterations)
Fixed-point iteration    : 27.525211 C   (13 iterations)
Check f(root) = -7.55e-11
|g'(T)| at the root = 0.126  (< 1, so fixed-point iteration converges)
```

![Task 10 summary](output/task_10_diagram.png)

**Interpretation:** The heat balance is zero at an equilibrium temperature of 27.5252 °C, and both methods agree. Bisection needs 38 halvings, because each step only gains one binary digit of accuracy. Fixed-point iteration needs only 13 steps, because `|g′(T)| ≈ 0.126` at the root, so each step cuts the error by about a factor of 8. On the cobweb graph, the iteration spirals into the point where `g(T)` crosses the line `y = T`.

### 6. Changing the Inputs and Observing the Results

| Change made | Result |
|---|---|
| Tolerance 10⁻² → 10⁻⁴ → 10⁻⁶ → 10⁻¹⁰ → 10⁻¹⁴ | Bisection: 11, 18, 25, 38, 51 iterations. Fixed point: 4, 6, 9, 13, 18. |
| Bracket `[20, 40]` → `[0, 100]` | 38 → 40 iterations. A wider bracket costs only a little extra. |
| Fixed-point start 25, 30, 100 | Same answer (27.525211 °C) in 13, 13 and 14 iterations. |
| Bracket `[30, 40]` (no sign change) | The code raises a `ValueError`, because no root is guaranteed. |
| Different rearrangement `T = −20 ln((T−25)/10)` | Does not converge. The values jump between about −81 and 599 (`|g′| ≈ 7.9 > 1`). |
| Stronger heat source (10 → 40) | Root moves to 32.7706 °C. Bisection 39 iterations, fixed point 27. |

**Observations:**

- Each extra digit of accuracy costs bisection about 3.3 more halvings, because the interval halves every time.
- Fixed-point iteration is faster, but only for a rearrangement with `|g′| < 1`. The same equation can converge or fail depending on how it is rewritten.
- Bisection needs a sign change, and it fails safely with an error when there isn't one.
- The starting guess hardly matters when fixed point converges.
- A stronger heat source makes `|g′|` larger, so fixed point needs more steps (27 instead of 13).

### 7. Related Real-World Case: Break-Even Points of a Business

**Scenario:** A company sells `q` units. The price falls as volume grows, so revenue is `R(q) = 80q − 0.03q²`. The cost is `C(q) = 5000 + 20q + 0.02q²`. The break-even points are where profit `f(q) = R(q) − C(q) = 60q − 0.05q² − 5000` equals zero.

**Assumptions:**

- Every unit produced is sold.
- Prices and costs follow the formulas above.
- There are two break-even points, so each is bracketed separately: `[0, 500]` and `[500, 1500]`.
- Tolerance 10⁻⁸.

**Results:**

| Method | q₁ (lower) | q₂ (upper) |
|---|---|---|
| Bisection | 90.0980 units | 1109.9020 units |
| Exact (quadratic formula) | 90.0980 units | 1109.9020 units |

The company makes a profit between about 90 and 1110 units. The most profitable volume is 600 units, with a profit of 13 000.

**Interpretation:** Below 90 units the fixed costs are not covered, and above 1110 units the falling price and rising cost wipe out the profit again. Bisection finds both roots as long as each is bracketed. Fixed-point iteration `q = (5000 + 0.05q²)/60` finds only the lower root (in 16 steps from `q₀ = 200`). From `q₀ = 1200` it runs away, because `|g′|` is larger than 1 at the upper root. This shows why bisection is the safer method when there are several roots.

### 8. How to Run

```
pip install numpy scipy matplotlib
python code/Task_10_bisection_fixed_point.py
```

## Task J (Task 11): Root Finding with Newton-Raphson and Secant Methods

**Program:** `code/Task_11_newton_secant.py`
**Outputs:** `output/task_11_output.txt`, `output/task_11_diagram.png`

### A. Concept
Newton-Raphson improves a guess using x(n+1) = x(n) - f(x(n)) / f'(x(n)). It follows the tangent line toward the root. The secant method does the same but estimates the slope from the two latest points, so no derivative is needed. Both usually converge much faster than bisection when started well.

**Problem:** A project costs 10,000 now and returns 3,000, 3,500, 4,000 and 2,500 at the end of years 1 to 4. Find the interest rate (the internal rate of return, IRR) that makes the net present value (NPV) equal to zero.

### B. Input values and mathematical operation

| Input | Value | Meaning |
|---|---|---|
| `cash_flows` | `[-10000, 3000, 3500, 4000, 2500]` | investment at t = 0, then yearly inflows |
| Newton start `x0` | 0.10 | first guess (10%) |
| Secant starts `x0`, `x1` | 0.05 and 0.20 | two first guesses |
| `xtol` | 1e-12 | stopping tolerance |

- NPV(r) = sum of CF(t) / (1 + r)^t. This is the function f(r) whose root is wanted.
- d_npv(r) = sum of -t * CF(t) / (1 + r)^(t+1). This is f'(r), used by Newton.
- Newton repeats r_new = r - NPV(r) / d_npv(r).
- Secant repeats r_new = r(n) - NPV(r(n)) * (r(n) - r(n-1)) / (NPV(r(n)) - NPV(r(n-1))).

Check by hand: NPV(0.10) = 332.63 and d_npv(0.10) = -22,143.92, so the first Newton step gives

## Task K (Task 12): Polynomial and Nonlinear Equation Solving

**Program:** `code/Task_12_nonlinear_equations.py`
**Outputs:** `output/task_12_output.txt`, `output/task_12_diagram.png`

### A. Concept
A polynomial equation has the form a(n)x^n + ... + a(1)x + a(0) = 0. A nonlinear system has equations that are not all linear in the unknowns. Numerical solvers search for values that make every residual close to zero. For example, x² - 5x + 6 = 0 has roots 2 and 3, and `np.roots([1, -5, 6])` returns them.

**Problem:** Find the intersection points of a circle and a parabola. This gives two nonlinear equations in x and y, solved with `scipy.optimize.root` from more than one starting guess.

### B. Input values and mathematical operation

| Input | Value | Meaning |
|---|---|---|
| Circle | x² + y² = 4 | circle of radius 2 centred at the origin |
| Parabola | y = x² - 1 | upward parabola with vertex at (0, -1) |
| Starting guesses | [1, 1] and [-1, 1] | first guesses for (x, y) |

The two equations are written as residuals that must both equal zero:
- r1(x, y) = x² + y² - 4
- r2(x, y) = y - (x² - 1)

`root` adjusts (x, y) until both residuals are near zero. Check by hand: substituting x² = y + 1 into the circle gives y² + y - 3 = 0, so y = (-1 + √13)/2 = 1.3028 (the other value, y = -2.3028, would need x² = -1.3028, which has no real solution). Then x = ±√2.3028 = ±1.5175.

**Result:**
```
Initial guess: [1.0, 1.0]
Intersection: [1.51748991 1.30277564]
Residuals: [-2.47801779e-13 -2.41584530e-13]
Converged: True

Initial guess: [-1.0, 1.0]
Intersection: [-1.51748991  1.30277564]
Residuals: [-2.47801779e-13 -2.41584530e-13]
Converged: True
```

### C. Changing the inputs

**C1. Different starting guesses (circle of radius 2):**

| Guess | Result | Max residual | Converged |
|---|---|---|---|
| [1, 1] | (1.51749, 1.302776) | 2.48e-13 | True |
| [-1, 1] | (-1.51749, 1.302776) | 2.48e-13 | True |
| [3, 3] | (1.51749, 1.302776) | 2.31e-12 | True |
| [0, -2] | (0, -2) | 1.00 | False |

**C2. Changing the circle's size (25 starting guesses each):**

| Circle | Number of intersections | Points |
|---|---|---|
| x² + y² = 4 | 2 | (±1.517490, 1.302776) |
| x² + y² = 1 | 3 | (±1, 0) and (0, -1) |
| x² + y² = 0.5 | 0 | none |

- Starting guesses on the upper side lead to the two real intersections, even from far away like [3, 3]. Different starting guesses can lead to different roots.
- The guess [0, -2] failed (`Converged: False`, residual 1.00). The solver stopped without making both residuals zero. This is why the program must **check the residuals and the converged flag**, not just print a number.
- A smaller circle (radius 1) gives 3 intersections, and an even smaller one (radius 0.707) gives none, because the circle no longer meets the parabola. In that case no starting guess can succeed.

### D. Meaning of the output and the graph
Each pair of numbers is a point that lies on both the circle and the parabola. The residuals of about 10⁻¹³ mean the equations are satisfied to machine precision, and `Converged: True` confirms it. The two answers are mirror images, (1.5175, 1.3028) and (-1.5175, 1.3028), because both curves are symmetric about the y-axis.

**Graph (left panel of `output/task_12_diagram.png`):** the circle and the parabola cross at exactly two red points in the upper half. The parabola also dips down through the bottom of the circle but never meets it there, which matches the fact that the second root of y has no real x.

![Task 12 graphs](output/task_12_diagram.png)

### E. Related real-world case: road crossing a cell-tower coverage zone

**Scenario:** A cell tower at the origin covers a circular area of radius 5 km. A road curves along y = 0.1x² - 3 (distances in km). Find where the road enters and leaves the coverage zone.

**Assumptions:**
- The coverage zone is a perfect circle of radius 5 km with no obstacles.
- The road is modelled by a parabola.
- Both equations use the same coordinates in km, with the tower at (0, 0).

**Input data:** circle x² + y² = 25, road y = 0.1x² - 3, starting guesses [4, -1] and [-4, -1].

**Output:**
```
Boundary crossing: x = 4.9721 km, y = -0.5279 km
Boundary crossing: x = -4.9721 km, y = -0.5279 km
Road is inside coverage for x between -4.9721 and 4.9721 km
Horizontal distance covered: 9.9441 km
```

**Interpretation:** The road enters the coverage zone at about (-4.97, -0.53) and leaves at about (4.97, -0.53). Between those points a phone on the road has coverage, over a horizontal stretch of about 9.94 km. The road's lowest point (0, -3) is 3 km from the tower, which is inside the 5 km range, so the road is covered in the middle and uncovered at the two ends. In the right panel of the graph, the red points mark where the road crosses the coverage boundary. This is the same method as the circle and parabola problem, applied to a different physical situation.

### F. Submission note
The discussion for this task was done as group work, but the code, experiments and write-up here are my own and are submitted in my individual GitHub repository. No group repository is used.

## Task L (Task 13): Interpolation

**Program:** `code/Task_13_interpolation.py`
**Outputs:** `output/task_13_output.txt`, `output/task_13_diagram.png`

### A. Concept
Interpolation estimates values between known data points. Linear interpolation draws a straight line between neighbouring points. Polynomial interpolation fits one polynomial through several points. A cubic spline joins smooth cubic curves so the curve stays smooth where the pieces meet. For example, if the temperature is 20 C at 10:00 and 24 C at 12:00, linear interpolation estimates 22 C at 11:00 (the program confirms this).

**Problem:** A sensor records the temperature every hour, but an application needs estimates every 10 minutes. We build interpolation functions with `scipy.interpolate` from the hourly measurements and evaluate them at the in-between times.

### B. Input values and mathematical operation

| Input | Value |
|---|---|
| `hours` | 8, 9, 10, 11, 12, 13, 14 |
| `temperature` (C) | 18.0, 20.0, 22.5, 24.0, 25.0, 25.5, 25.0 |
| Evaluation times | every 10 minutes (1/6 hour) from 08:00 to 14:00, which is 37 points |
| Target time | 10:30 (`10.5`) |

- **Linear** (`interp1d`): between two points (x0, y0) and (x1, y1), the estimate is y = y0 + (y1 - y0) * (x - x0) / (x1 - x0).
- **Cubic spline** (`CubicSpline`): a separate cubic polynomial on each hour, chosen so that the curve and its first and second derivatives match where the pieces join.

Check by hand: at 10:30 the linear estimate is the midpoint of 22.5 and 24.0, which is 23.25.

**Result:**
```
Linear estimate at 10:30: 23.25 C
Cubic estimate at 10:30:  23.36 C
Number of 10-minute estimates: 37
```

### C. Changing the inputs

**C1. Change one measurement (11:00 changed from 24.0 to 27.0):**

| Estimate | Before | After |
|---|---|---|
| Linear at 10:30 | 23.25 C | 24.75 C |
| Cubic at 10:30 | 23.36 C | 25.18 C |
| Linear at 12:30 | 25.25 C | 25.25 C |
| Cubic at 12:30 | 25.34 C | 24.86 C |

**C2. Use only every 2 hours (08, 10, 12, 14) and estimate the hours left out:**

| Time | Actual | Linear | Error | Cubic | Error |
|---|---|---|---|---|---|
| 09:00 | 20.00 | 20.25 | +0.25 | 20.47 | +0.47 |
| 11:00 | 24.00 | 23.75 | -0.25 | 24.03 | +0.03 |
| 13:00 | 25.50 | 25.00 | -0.50 | 25.34 | -0.16 |

**C3. Estimating outside the data range (14:30):** `interp1d` raises a `ValueError`, while `CubicSpline` silently extrapolates to 24.20 C.

- Linear interpolation only reacts to the two neighbouring points: changing 11:00 left the linear estimate at 12:30 unchanged. The cubic spline is global, so the same change also moved the estimate at 12:30 (25.34 to 24.86).
- With fewer data points, the cubic spline was usually closer to the true values (at 11:00 its error was +0.03 against -0.25 for linear), but not always: at 09:00 linear was closer.
- Extrapolating beyond the measured times is unreliable and should be avoided.

### D. Meaning of the output and the graph
At 10:30 the temperature is estimated at about **23.25 C** (linear) or **23.36 C** (cubic). Both are reasonable because they lie between the 10:00 reading (22.5 C) and the 11:00 reading (24.0 C). The cubic value is slightly higher because the temperature is rising quickly at that point, and the spline curve bends upward with it. In total, the program produces 37 estimates covering every 10 minutes from 08:00 to 14:00.

**Graph (left panel of `output/task_13_diagram.png`):** the black dots are the hourly measurements. The linear estimate is a set of straight segments with corners at each measurement. The cubic spline is a smooth curve through the same points, which flattens naturally around the 13:00 peak. Both pass exactly through the measured points and differ only in between.

![Task 13 graphs](output/task_13_diagram.png)

### E. Related real-world case: solar panel power output

**Scenario:** A solar panel's power output (kW) is recorded every hour from 06:00 to 18:00. We need the output at times between readings and the total energy produced in the day.

**Assumptions:**
- The measurements below are assumed (illustrative) data, not real readings.
- Output is zero at 06:00 and 18:00 and peaks around midday.
- No clouds or other sudden changes happen between readings.

**Input data (kW, hourly from 06:00):** 0.0, 0.4, 1.2, 2.1, 2.8, 3.2, 3.3, 3.0, 2.4, 1.6, 0.8, 0.2, 0.0

**Output:**
```
Estimate at 10:45: linear 3.100 kW | cubic 3.126 kW
Estimate at 15:20: linear 1.333 kW | cubic 1.323 kW
Energy produced, trapezoid on hourly data: 21.000 kWh
Energy produced, cubic spline integral:    21.003 kWh
Lowest cubic value between measurements:   -0.004 kW
Highest cubic value between measurements:  3.311 kW (highest measurement 3.3 kW)
```

**Interpretation:** The panel is estimated to produce about 3.1 kW at 10:45 and about 1.3 kW at 15:20. Over the day it produces about **21.0 kWh**, and both methods agree closely, so the choice of method hardly matters for the total. The cubic spline does show a small weakness: between 06:00 and 07:00 it dips to -0.004 kW, which is impossible for real power, and it slightly overshoots the highest measurement (3.311 kW against 3.3 kW). In practice such estimates should be clipped at zero, and linear interpolation is the safer choice when values must never leave the measured range.

### F. Submission note
The discussion for this task was done as group work, but the code, experiments and write-up here are my own and are submitted in my individual GitHub repository. No group repository is used.

## Task M: Curve Fitting and Approximation

**Program file:** `code/Task_14_curve_fitting.py`
**Output graph:** `output/task_14_diagram.png`

### 1. Mathematical Concept

Curve fitting chooses model parameters so that a mathematical function follows observed data as closely as possible. In least squares, the parameters minimize `Σ(yᵢ − ŷᵢ)²`, the sum of squared residuals between the observations `yᵢ` and the model predictions `ŷᵢ`. The root mean squared error `RMSE = √(mean of the squared residuals)` measures the typical size of the error in the units of the data, and `R²` measures the share of the variation that the model explains.

### 2. Example

If data approximately follows `y = a + bx`, least squares chooses `a` and `b` that minimize the squared vertical differences between the measured points and the fitted line. For `x = 1…5` and `y = [2.2, 4.1, 5.9, 8.2, 9.8]`, the best line is `y = 0.250 + 1.930x`, with a sum of squared residuals of 0.0830. A different line such as `y = 2.1x` is worse, with a sum of 0.7100.

### 3. Real-World Applications

Trend estimation, calibration, performance modelling, growth curves, sensor models, demand forecasting and experimental science.

### 4. Real-World Problem and Solution Approach

**Problem:** Measure server response time at different user loads and estimate a mathematical relationship between load and response time.

**Solution approach:** Choose a plausible model (here a quadratic, because the delay grows faster as the load increases), fit its parameters with `scipy.optimize.curve_fit`, plot the observed and fitted values, and inspect the residuals.

| Concurrent users | 50 | 100 | 150 | 200 | 250 | 300 | 350 |
|---|---|---|---|---|---|---|---|
| Response time (ms) | 82 | 95 | 115 | 145 | 185 | 238 | 305 |

### 5. Solution Code

See `code/Task_14_curve_fitting.py`. It fits the quadratic model, reports the RMSE, R² and parameter errors, compares other models, tests an outlier and fewer data points, fits a cooling-coffee case, and draws one summary diagram.

**Output:**

```
Fitted model: y = 0.002167x^2 -0.1345x +85.00
Parameter standard errors: a +/- 0.000089, b +/- 0.0366, c +/- 3.19
RMSE: 1.55 ms
R^2 : 0.99958
Residuals (ms): [-1.69  1.79  1.43  0.24 -1.79 -1.64  1.67]
Predicted response at 400 users: 377.86 ms
Predicted response at 500 users: 559.40 ms
```

![Task 14 summary](output/task_14_diagram.png)

**Interpretation:** The fitted curve is `y = 0.002167x² − 0.1345x + 85.00`. The model explains 99.96% of the variation (R² = 0.9996), and the typical error is only 1.55 ms. The residuals are small and have mixed signs, so there is no pattern the model has missed. The curve says the response time starts around 85 ms with no load and climbs faster and faster, reaching about 378 ms at 400 users. The prediction at 400 users is only slightly beyond the measured range (up to 350), so it is reasonably trustworthy, but the 500-user prediction is much less certain.

### 6. Changing the Inputs and Observing the Results

| Change made | RMSE | At 400 users | At 500 users | Leave-one-out RMSE |
|---|---|---|---|---|
| Linear model | 18.83 ms | 312.9 ms | 386.1 ms | 30.64 ms |
| Quadratic model (original) | 1.55 ms | 377.9 ms | 559.4 ms | 4.16 ms |
| Cubic model | 0.14 ms | 387.9 ms | 607.7 ms | 0.27 ms |
| Exponential model | 1.43 ms | 400.4 ms | 698.6 ms | 4.26 ms |
| Degree-5 polynomial | 0.14 ms | 389.4 ms | 630.3 ms | 5.98 ms |

| Other change | Result |
|---|---|
| Outlier: the 250-user value 185 → 300 ms | The curve changes to `y = 0.000524x² + 0.6048x + 35.71`, the RMSE jumps from 1.55 to 35.96 ms, and the 400-user prediction moves from 377.9 to 361.4 ms. |
| Fit only 50 to 200 users, predict the rest | Errors grow with distance: −2.2 ms at 250 users, −8.8 ms at 300 and −20.9 ms at 350. |
| Add ±15 ms noise to the measurements | RMSE rises from 1.55 to 11.43 ms, and the standard error of `a` rises from 0.000089 to 0.000660. |

**Observations:**

- The straight line is clearly too simple, with a fit error of 18.8 ms. The curved models fit far better.
- The cubic and degree-5 curves both reach an RMSE of 0.14 ms on the data, but the leave-one-out test (fit without one point, then predict it) gives 0.27 ms for the cubic and 5.98 ms for degree 5. A fit that matches the data closely can still be overfitting.
- The models agree inside the measured range but disagree outside it: the predictions at 500 users range from 386 to 699 ms, so extrapolating needs care.
- One bad measurement changes every parameter, because squaring the residuals gives large errors a large weight.
- More noise in the data gives a larger RMSE and less certain parameters.

### 7. Related Real-World Case: Cooling of a Cup of Coffee

**Scenario:** A cup of coffee cools down in a room. Newton's law of cooling says that the temperature follows `T(t) = T_room + (T₀ − T_room)·e^(−kt)`. We measure the temperature over 30 minutes and fit the model to estimate the room temperature and the cooling rate `k`.

**Assumptions:**

- The room temperature is constant.
- The cooling rate is proportional to the difference between the coffee and the room.
- The measurements have small random errors (about ±0.5 °C).

| Time (min) | 0 | 2 | 4 | 6 | 8 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| Temperature (°C) | 90.2 | 80.7 | 73.9 | 66.2 | 61.3 | 55.4 | 46.3 | 38.5 | 30.6 |

**Results:**

```
Fitted: T(t) = 22.39 + 67.65 exp(-0.0706 t)
Room temperature estimate: 22.39 +/- 1.00 C
Cooling constant k = 0.0706 +/- 0.0023 per minute
Time constant 1/k = 14.2 min;  half-life of the temperature difference = 9.8 min
RMSE exponential: 0.39 C   (a straight line gives 5.47 C)
Time for the coffee to cool to 45 C: 15.5 minutes
Predicted temperature after 60 minutes: 23.4 C
```

**Interpretation:** The fit estimates the room temperature at 22.4 °C and the cooling constant at 0.0706 per minute, so the gap between the coffee and the room halves about every 9.8 minutes. The exponential curve follows the data to within 0.39 °C, while a straight line is off by 5.47 °C, because cooling slows down as the coffee approaches room temperature. The curve predicts that the coffee reaches 45 °C after about 15.5 minutes and is almost at room temperature (23.4 °C) after an hour. This model has a physical reason behind it, which makes predictions beyond the data more trustworthy than for a purely empirical curve.

### 8. How to Run

```
pip install numpy scipy matplotlib
python code/Task_14_curve_fitting.py
```

## Task N: Numerical Differentiation

**Program file:** `code/Task_15_numerical_differentiation.py`
**Output graph:** `output/task_15_diagram.png`

### 1. Mathematical Concept

A derivative measures how quickly a quantity changes. When a formula for the derivative is unavailable, it can be approximated from nearby values.

- Forward difference: `f′(x) ≈ [f(x+h) − f(x)] / h`
- Backward difference: `f′(x) ≈ [f(x) − f(x−h)] / h`
- Central difference: `f′(x) ≈ [f(x+h) − f(x−h)] / (2h)`, which is usually more accurate for the same `h`.

The forward and backward errors shrink in proportion to `h`, while the central error shrinks in proportion to `h²`.

### 2. Example

If distance changes from 100 m to 112 m in 2 s, the average rate of change is (112 − 100) / 2 = 6 m/s. Numerical differentiation applies the same idea over very small intervals. For `f(x) = x²` at `x = 1` (exact derivative 2), the forward difference with `h = 0.1` gives 2.1 (error 0.1), and the central difference gives exactly 2.

### 3. Real-World Applications

Estimating speed from position data, acceleration from velocity, gradients in optimization, rates of change in sensors and sensitivity analysis.

### 4. Real-World Problem and Solution Approach

**Problem:** A GPS device records position at closely spaced times. Estimate the instantaneous vehicle speed.

**Solution approach:** Apply central differences to neighbouring position measurements and divide by the time interval. Use a forward difference at the first point and a backward difference at the last point, where only one neighbour exists. Compare different step sizes to understand the error.

| Time (s) | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Position (m) | 0.0 | 4.8 | 10.2 | 16.1 | 22.7 | 30.0 | 38.0 |

### 5. Solution Code

See `code/Task_15_numerical_differentiation.py`. It estimates the speed from the GPS data, checks it against `np.gradient`, studies the effect of step size, noise and the end formulas, estimates the speed and deceleration of a braking car, and draws one summary diagram.

**Output:**

```
 t (s)  position (m)  speed (m/s)     method  smooth ref
     0           0.0         4.80    forward        4.40
     1           4.8         5.10    central        5.04
     2          10.2         5.65    central        5.68
     3          16.1         6.25    central        6.32
     4          22.7         6.95    central        6.96
     5          30.0         7.65    central        7.60
     6          38.0         8.00   backward        8.24
Same result as np.gradient: True
```

![Task 15 summary](output/task_15_diagram.png)

**Interpretation:** The vehicle speeds up steadily from about 4.8 m/s to 8.0 m/s over 6 seconds, an acceleration of about 0.64 m/s². At 3 s the speed is 6.25 m/s (22.5 km/h). The "smooth reference" column is the derivative of a quadratic curve fitted to the positions, and it is only a guide, since the true speed is unknown. The central estimates agree with it to within 0.07 m/s. The end points differ more (0.40 m/s at 0 s and 0.24 m/s at 6 s), because the forward and backward formulas are only first-order accurate and have just one neighbouring point to use.

### 6. Changing the Inputs and Observing the Results

| Change made | Result |
|---|---|
| Step size on `sin(x)` at `x = 1`, h = 10⁻¹ → 10⁻⁴ | Forward error 4.3×10⁻² → 4.2×10⁻⁵ (proportional to h). Central error 9.0×10⁻⁴ → 9.0×10⁻¹⁰ (proportional to h²). |
| Step size even smaller, h = 10⁻¹⁰, 10⁻¹², 10⁻¹⁵ | The error grows again: forward 5.9×10⁻⁸, 4.3×10⁻⁵, 1.5×10⁻². The best values are 2.5×10⁻⁹ (forward, at h ≈ 6×10⁻⁹) and 3.8×10⁻¹³ (central, at h ≈ 6×10⁻⁶). |
| Noise of about ±0.3 m added to the GPS positions | With 2 s central differences the speeds change by up to 0.3 m/s, for example 4.8 instead of 5.1 at 1 s. The RMS error against the smooth reference is 0.23 m/s. |
| Wider span (4 s instead of 2 s) on the same noisy data | The RMS error falls to 0.01 m/s. A wider span averages out the noise. |
| Use every 2nd GPS point (2 s interval) | Interior speeds stay accurate (5.67 and 6.95 m/s at 2 s and 4 s, against 5.68 and 6.96), but the end points get worse (7.65 against 8.24 at 6 s). |
| Second-order formula at the ends | At 0 s: 4.50 instead of 4.80 (reference 4.40). At 6 s: 8.35 instead of 8.00 (reference 8.24). |

**Observations:**

- Central differences are more accurate than forward or backward differences for the same step size.
- Making `h` smaller improves the answer only up to a point. Below about 10⁻⁶ to 10⁻⁸, round-off error from subtracting nearly equal numbers takes over and the answer gets worse.
- Differentiating noisy data amplifies the noise, so noisy measurements should use a wider span or be smoothed first.
- The end points are the least accurate, and a second-order one-sided formula improves them.

### 7. Related Real-World Case: Speed and Deceleration of a Braking Car

**Scenario:** A GPS unit records the position of a braking car every 0.5 s. We estimate its speed and its acceleration. The speed is the first derivative of the position, and the acceleration is the second derivative, calculated with `[x(t+h) − 2x(t) + x(t−h)] / h²`.

**Assumptions:**

- The car starts at 20 m/s and slows down at a constant 4 m/s², so the true position is `x = 20t − 2t²`.
- Each GPS position has a small error of up to ±0.04 m.
- Times from 0 to 4 s, with `h = 0.5` s.

**Results:**

| t (s) | Speed estimate (m/s) | True speed (m/s) | Acceleration estimate (m/s²) |
|---|---|---|---|
| 0.5 | 18.02 | 18.00 | −3.52 |
| 1.0 | 16.01 | 16.00 | −4.52 |
| 2.0 | 11.98 | 12.00 | −4.48 |
| 3.0 | 8.01 | 8.00 | −4.44 |
| 3.5 | 5.99 | 6.00 | −3.64 |

The speed estimates have an RMS error of only 0.014 m/s. The acceleration estimates range from −4.52 to −3.48 m/s², with a mean of −3.95 m/s² (true value −4.00). A straight line through the speed estimates gives a deceleration of 4.01 m/s², which means the car stops after about 5.0 s and its stopping distance is about 49.9 m (true value 50 m).

**Interpretation:** The speed estimates are very accurate, but single acceleration estimates swing between −3.5 and −4.5 m/s² because the second difference divides the position errors by `h² = 0.25`, which magnifies them. The average of many estimates, or the slope of a line through the speed estimates, is much more reliable. This shows that every extra derivative makes the noise problem worse.

### 8. How to Run

```
pip install numpy matplotlib
python code/Task_15_numerical_differentiation.py
```

## Task O (Task 16): Numerical Integration

**Program:** `code/Task_16_numerical_integration.py`
**Outputs:** `output/task_16_output.txt`, `output/task_16_diagram.png`

### A. Concept
An integral is an accumulated quantity, the area under a curve. The trapezoidal rule approximates that area with trapezoids. Simpson's rule fits parabolic segments through the points and is often more accurate for smooth functions. SciPy's `quad` uses adaptive quadrature, which refines the calculation automatically. For example, constant power of 2 kW for 3 hours gives ∫P dt = 2 × 3 = 6 kWh, which the program confirms. For changing power, many small contributions are added.

**Problem:** A smart meter records power at regular intervals. Estimate the total energy consumed.

### B. Input values and mathematical operation

| Input | Value |
|---|---|
| `time_hours` | 0, 2, 4, 6, 8, 10, 12 |
| `power_kw` | 0.8, 1.0, 1.4, 1.8, 1.6, 1.2, 0.9 |

- Trapezoidal rule: energy ≈ Σ (P(i) + P(i+1))/2 · Δt.
- Simpson's rule: energy ≈ (h/3)·[P0 + 4(P1 + P3 + P5) + 2(P2 + P4) + P6] for 6 intervals of width h = 2 hours.

Check by hand: trapezoid = 2·(0.4 + 1.0 + 1.4 + 1.8 + 1.6 + 1.2 + 0.45) = 2 × 7.85 = 15.7 kWh. Simpson = (2/3)·(0.8 + 4×4.0 + 2×3.0 + 0.9) = (2/3) × 23.7 = 15.8 kWh.

**Result:**
```
Energy using trapezoidal rule: 15.700 kWh
Energy using Simpson's rule:   15.800 kWh
Difference between estimates:  0.100 kWh
```

### C. Changing the inputs

**C1. One reading changed (06:00 from 1.8 kW to 2.4 kW):**

| Rule | Before | After |
|---|---|---|
| Trapezoid | 15.700 kWh | 16.900 kWh |
| Simpson | 15.800 kWh | 17.400 kWh |

**C2. Reading interval, using the smooth function P(t) = 1.2 + 0.7·sin(πt/12) with a known exact answer of 19.747606 kWh:**

| Interval | Trapezoid error | Simpson error |
|---|---|---|
| 6 h | 1.147606 | 0.252394 |
| 3 h | 0.277758 | 0.012192 |
| 2 h | 0.122735 | 0.002308 |
| 1 h | 0.030578 | 0.000141 |

**C3. Adaptive `quad`** on the same function gives 19.747606 kWh with an error estimate of 2.2e-13.

- Raising one reading increases the energy, and Simpson reacts more strongly (+1.6 kWh against +1.2 kWh for the trapezoid) because it gives weight 4 to odd-numbered readings and the 06:00 reading is one of them.
- Shorter reading intervals give smaller errors for both rules. For this smooth curve, Simpson's error shrinks much faster: at 2 hours it is 0.0023 kWh against 0.123 kWh for the trapezoid.
- `quad` is best when the function itself is known, since it picks its own points. With meter data we only have the readings, so we must choose between the trapezoid and Simpson.

### D. Meaning of the output and the graph
The household used about **15.7 to 15.8 kWh** over the 12 hours, which is the area under the power curve. Both rules agree within 0.1 kWh (under 1%), so the estimate is reliable here. Power rises from 0.8 kW at the start, peaks at 1.8 kW at 06:00 and falls back to 0.9 kW, and the shaded area in the left panel of `output/task_16_diagram.png` is the energy used. The area unit works out as kW × hours = kWh.

![Task 16 graphs](output/task_16_diagram.png)

### E. Related real-world case: distance of a city bus from speed readings

**Scenario:** A city bus logs its speed every 5 minutes for one hour. Distance is the integral of speed over time, so we estimate the distance travelled.

**Assumptions:**
- The speed readings are assumed (illustrative) data.
- The bus stops at 30 minutes and at the start and end, and speed changes smoothly between readings.
- Time is converted to hours so that km/h × h = km.

**Input data:** speeds (km/h) at 0, 5, ..., 60 minutes: 0, 18, 32, 40, 45, 30, 0, 12, 35, 48, 50, 36, 0.

**Output:**
```
Distance using trapezoidal rule: 28.833 km
Distance using Simpson's rule:   29.444 km
Difference between estimates:    0.611 km
Average speed over the hour:     28.83 km/h
```

**Interpretation:** The bus travelled about **28.8 to 29.4 km** in the hour, an average speed of about 28.8 km/h, which is realistic for a city route with stops. The two rules differ by 0.611 km (about 2%), a bigger gap than for the smooth meter data. This is because the speed drops to zero and changes sharply between readings, which parabolas fit less well. For data with sudden changes like this, the trapezoidal rule is the safer choice, and shorter reading intervals would narrow the gap.

### F. Submission note
The discussion for this task was done as group work, but the code, experiments and write-up here are my own and are submitted in my individual GitHub repository. No group repository is used.