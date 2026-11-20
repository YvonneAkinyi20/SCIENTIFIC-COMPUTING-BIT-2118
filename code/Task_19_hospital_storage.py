
# ============================================================
# BIT 2118 - SCIENTIFIC COMPUTING
# TASK 19: NUMERICAL OPTIMIZATION
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path
import sys


# ============================================================
# 1. CREATE OUTPUT FOLDER
# ============================================================

# Locate the main project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Create output folder if it does not exist
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SAVE RESULTS TO TEXT FILE AND TERMINAL
# ============================================================

class Tee:

    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


# ============================================================
# 3. GENERAL NUMERICAL OPTIMIZATION FUNCTION
# ============================================================

def optimize_problem(
    title,
    cost_a,
    cost_b,
    capacity_a,
    capacity_b,
    required_capacity,
    label_a,
    label_b,
    graph_filename
):

    # --------------------------------------------------------
    # OBJECTIVE FUNCTION
    # Minimize total operating cost
    # --------------------------------------------------------

    def objective(z):

        x, y = z

        return cost_a*x + cost_b*y

    # --------------------------------------------------------
    # CAPACITY CONSTRAINT
    # Total capacity must meet the requirement
    # --------------------------------------------------------

    def capacity_constraint(z):

        x, y = z

        return (
            capacity_a*x +
            capacity_b*y -
            required_capacity
        )

    constraints = {
        "type": "ineq",
        "fun": capacity_constraint
    }

    # Allocations cannot be negative
    bounds = [
        (0, None),
        (0, None)
    ]

    # --------------------------------------------------------
    # PERFORM OPTIMIZATION USING SCIPY
    # --------------------------------------------------------

    solution = minimize(

        objective,

        x0=[5, 3],

        method="SLSQP",

        bounds=bounds,

        constraints=constraints

    )

    if not solution.success:

        print("Optimization failed:", solution.message)
        return

    # Obtain optimal values
    x, y = solution.x

    # Calculate total capacity
    total_capacity = (
        capacity_a*x +
        capacity_b*y
    )

    # --------------------------------------------------------
    # FIND BEST WHOLE-UNIT SOLUTION
    # --------------------------------------------------------

    best_integer = None

    for a in range(21):

        for b in range(21):

            capacity = (
                capacity_a*a +
                capacity_b*b
            )

            if capacity >= required_capacity:

                total_cost = (
                    cost_a*a +
                    cost_b*b
                )

                if (
                    best_integer is None
                    or total_cost < best_integer[0]
                ):

                    best_integer = (
                        total_cost,
                        a,
                        b,
                        capacity
                    )

    # --------------------------------------------------------
    # DISPLAY OPTIMIZATION RESULTS
    # --------------------------------------------------------

    print("\n" + "="*60)

    print(title)

    print("="*60)

    print("\nINPUT VALUES")

    print(f"Cost of {label_a}: {cost_a}")

    print(f"Cost of {label_b}: {cost_b}")

    print(f"Capacity of {label_a}: {capacity_a}")

    print(f"Capacity of {label_b}: {capacity_b}")

    print(f"Required capacity: {required_capacity}")

    print("\nCONTINUOUS OPTIMIZATION RESULTS")

    print("Optimization success:", solution.success)

    print(f"{label_a} allocation: {x:.3f}")

    print(f"{label_b} allocation: {y:.3f}")

    print(f"Total capacity: {total_capacity:.2f}")

    print(f"Minimum planning cost: {solution.fun:.2f}")

    print("\nBEST WHOLE-UNIT SOLUTION")

    if best_integer is not None:

        print(f"Minimum cost: {best_integer[0]}")

        print(f"{label_a} allocation: {best_integer[1]}")

        print(f"{label_b} allocation: {best_integer[2]}")

        print(f"Total capacity: {best_integer[3]}")

    # --------------------------------------------------------
    # 4. GENERATE VISUALIZATION GRAPH
    # --------------------------------------------------------

    # Set graph ranges
    max_x = max(14, required_capacity / capacity_a * 1.2)

    max_y = max(10, required_capacity / capacity_b * 1.2)

    X = np.linspace(0, max_x, 300)

    Y = np.linspace(0, max_y, 300)

    xx, yy = np.meshgrid(X, Y)

    # Calculate capacity at every graph point
    capacity_grid = (
        capacity_a*xx +
        capacity_b*yy
    )

    # Identify feasible region
    feasible = (
        capacity_grid >= required_capacity
    )

    # Create graph
    plt.figure(figsize=(10, 7))

    # Shade feasible region
    plt.contourf(
        xx,
        yy,
        feasible.astype(int),
        levels=[0.5, 1.5],
        alpha=0.25
    )

    # --------------------------------------------------------
    # PLOT CAPACITY CONSTRAINT
    # --------------------------------------------------------

    boundary_y = (
        required_capacity - capacity_a*X
    ) / capacity_b

    plt.plot(
        X,
        boundary_y,
        linewidth=2,
        label="Capacity constraint"
    )

    # --------------------------------------------------------
    # PLOT MINIMUM COST LINE
    # --------------------------------------------------------

    optimal_cost = solution.fun

    cost_y = (
        optimal_cost - cost_a*X
    ) / cost_b

    plt.plot(
        X,
        cost_y,
        linestyle="--",
        linewidth=2,
        label="Minimum cost line"
    )

    # --------------------------------------------------------
    # MARK CONTINUOUS OPTIMAL SOLUTION
    # --------------------------------------------------------

    plt.scatter(
        x,
        y,
        color="red",
        s=120,
        label="Continuous optimal solution",
        zorder=5
    )

    # --------------------------------------------------------
    # MARK BEST WHOLE-UNIT SOLUTION
    # --------------------------------------------------------

    if best_integer is not None:

        _, a, b, _ = best_integer

        plt.scatter(
            a,
            b,
            color="green",
            marker="*",
            s=200,
            label="Best whole-unit solution",
            zorder=6
        )

    # --------------------------------------------------------
    # GRAPH LABELS AND TITLE
    # --------------------------------------------------------

    plt.xlabel(label_a)

    plt.ylabel(label_b)

    plt.title(title)

    plt.xlim(0, max_x)

    plt.ylim(0, max_y)

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    # --------------------------------------------------------
    # SAVE GRAPH AUTOMATICALLY
    # --------------------------------------------------------

    graph_path = OUTPUT_DIR / graph_filename

    plt.savefig(
        graph_path,
        dpi=300,
        bbox_inches="tight"
    )

    print("\nGraph saved:", graph_path.name)

    # Display graph
    plt.show()

    plt.close()


# ============================================================
# 5. RUN ALL EXPERIMENTS AND SAVE RESULTS
# ============================================================

output_file_path = OUTPUT_DIR / "task_19_output.txt"

with open(
    output_file_path,
    "w",
    encoding="utf-8"
) as output_file:

    # Save results both in terminal and text file
    original_stdout = sys.stdout

    sys.stdout = Tee(original_stdout, output_file)

    try:

        # ====================================================
        # EXPERIMENT 1: ORIGINAL CLOUD SERVER OPTIMIZATION
        # ====================================================

        optimize_problem(

            title="EXPERIMENT 1: ORIGINAL CLOUD SERVER OPTIMIZATION",

            cost_a=12,

            cost_b=18,

            capacity_a=100,

            capacity_b=170,

            required_capacity=1000,

            label_a="Type A Servers",

            label_b="Type B Servers",

            graph_filename="task_19_original.png"

        )

        # ====================================================
        # EXPERIMENT 2: MODIFIED CLOUD SERVER OPTIMIZATION
        # ====================================================

        optimize_problem(

            title="EXPERIMENT 2: MODIFIED CLOUD SERVER OPTIMIZATION",

            cost_a=12,

            cost_b=18,

            capacity_a=100,

            capacity_b=170,

            required_capacity=1200,

            label_a="Type A Servers",

            label_b="Type B Servers",

            graph_filename="task_19_modified.png"

        )

        # ====================================================
        # EXPERIMENT 3: HOSPITAL CLOUD STORAGE OPTIMIZATION
        # ====================================================

        optimize_problem(

            title="EXPERIMENT 3: HOSPITAL CLOUD STORAGE OPTIMIZATION",

            cost_a=8,

            cost_b=11,

            capacity_a=100,

            capacity_b=170,

            required_capacity=1200,

            label_a="Storage A Units",

            label_b="Storage B Units",

            graph_filename="task_19_hospital.png"

        )

        # ====================================================
        # FINAL MESSAGE
        # ====================================================

        print("\n" + "="*60)

        print("ALL OPTIMIZATION EXPERIMENTS COMPLETED SUCCESSFULLY!")

        print("All numerical results and graphs have been saved.")

        print("="*60)

    finally:

        sys.stdout = original_stdout


print("\nOUTPUT FILES LOCATION:")

print(OUTPUT_DIR)

print("\nCheck your output folder for all saved files.")


# ============================================================
# END OF TASK 19
# ============================================================
