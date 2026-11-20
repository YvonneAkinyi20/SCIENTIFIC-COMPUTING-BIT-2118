
# ============================================================
# BIT 2118 - SCIENTIFIC COMPUTING
# TASK 20: RANDOM NUMBERS AND MONTE CARLO METHODS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


# ============================================================
# 1. CREATE OUTPUT FOLDER
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. MONTE CARLO SIMULATION FUNCTION
# ============================================================

def monte_carlo_simulation(
    title,
    capacity,
    mean_demand,
    std_demand,
    N,
    seed,
    graph_filename
):

    # Generate random numbers
    rng = np.random.default_rng(seed)

    demand = rng.normal(
        mean_demand,
        std_demand,
        N
    )

    # Identify overloaded cases
    overloaded = demand > capacity

    # Calculate overload probability
    overload_probability = overloaded.mean()

    overload_count = overloaded.sum()

    successful_cases = N - overload_count

    # Calculate running probability
    running_probability = (
        np.cumsum(overloaded) /
        np.arange(1, N + 1)
    )

    # Select points for graph
    indices = np.unique(
        np.logspace(
            1,
            np.log10(N),
            200
        ).astype(int)
    ) - 1

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 60)

    print(title)

    print("=" * 60)

    print("\nINPUT VALUES")

    print("Server capacity:", capacity)

    print("Mean demand:", mean_demand)

    print("Standard deviation:", std_demand)

    print("Number of simulations:", N)

    print("Random seed:", seed)

    print("\nSIMULATION RESULTS")

    print(
        f"Estimated overload probability: "
        f"{overload_probability:.4%}"
    )

    print(
        f"Overload percentage: "
        f"{overload_probability * 100:.2f}%"
    )

    print(
        f"Number of overload cases: "
        f"{overload_count} out of {N}"
    )

    print(
        f"Successful cases: "
        f"{successful_cases} out of {N}"
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print("\nINTERPRETATION")

    if overload_probability > 0.5:

        print("The server has a high overload risk.")

    elif overload_probability > 0.1:

        print("The server has a moderate overload risk.")

    else:

        print("The server has a relatively low overload risk.")

    # ========================================================
    # 3. CREATE VISUALIZATION GRAPHS
    # ========================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6)
    )

    # --------------------------------------------------------
    # GRAPH 1: PROBABILITY CONVERGENCE
    # --------------------------------------------------------

    axes[0].semilogx(
        indices + 1,
        running_probability[indices],
        label="Estimated probability"
    )

    axes[0].axhline(
        overload_probability,
        linestyle="--",
        label="Final estimated probability"
    )

    axes[0].set_xlabel(
        "Number of simulations"
    )

    axes[0].set_ylabel(
        "Estimated overload probability"
    )

    axes[0].set_title(
        "Monte Carlo Probability Convergence"
    )

    axes[0].grid(True)

    axes[0].legend()

    # --------------------------------------------------------
    # GRAPH 2: DISTRIBUTION OF NETWORK DEMAND
    # --------------------------------------------------------

    axes[1].hist(
        demand,
        bins=50,
        edgecolor="black",
        alpha=0.7
    )

    axes[1].axvline(
        capacity,
        linestyle="--",
        linewidth=2,
        label="Server capacity"
    )

    axes[1].set_xlabel(
        "Incoming network demand"
    )

    axes[1].set_ylabel(
        "Frequency"
    )

    axes[1].set_title(
        "Distribution of Random Network Demand"
    )

    axes[1].grid(True)

    axes[1].legend()

    # --------------------------------------------------------
    # SAVE GRAPH
    # --------------------------------------------------------

    plt.suptitle(title)

    plt.tight_layout()

    graph_path = OUTPUT_DIR / graph_filename

    plt.savefig(
        graph_path,
        dpi=300,
        bbox_inches="tight"
    )

    print("\nGraph saved:", graph_filename)

    plt.show()

    plt.close()

    return overload_probability


# ============================================================
# 4. RUN EXPERIMENTS AND SAVE RESULTS
# ============================================================

output_file_path = OUTPUT_DIR / "task_20_output.txt"

original_stdout = sys.stdout


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


with open(
    output_file_path,
    "w",
    encoding="utf-8"
) as output_file:

    sys.stdout = Tee(
        original_stdout,
        output_file
    )

    try:

        # ====================================================
        # EXPERIMENT 1: ORIGINAL NETWORK SERVER
        # ====================================================

        original_probability = monte_carlo_simulation(

            title="EXPERIMENT 1: ORIGINAL NETWORK SERVER",

            capacity=1000.0,

            mean_demand=850.0,

            std_demand=120.0,

            N=100_000,

            seed=42,

            graph_filename="task_20_original.png"

        )

        # ====================================================
        # EXPERIMENT 2: MODIFIED NETWORK SERVER
        # ====================================================

        modified_probability = monte_carlo_simulation(

            title="EXPERIMENT 2: MODIFIED NETWORK SERVER",

            capacity=900.0,

            mean_demand=850.0,

            std_demand=120.0,

            N=100_000,

            seed=42,

            graph_filename="task_20_modified.png"

        )

        # ====================================================
        # EXPERIMENT 3: HOSPITAL INFORMATION SERVER
        # ====================================================

        hospital_probability = monte_carlo_simulation(

            title="EXPERIMENT 3: HOSPITAL INFORMATION SERVER",

            capacity=800.0,

            mean_demand=650.0,

            std_demand=90.0,

            N=100_000,

            seed=43,

            graph_filename="task_20_hospital.png"

        )

        # ====================================================
        # 5. COMPARE ALL EXPERIMENTS
        # ====================================================

        print("\n" + "=" * 60)

        print("COMPARISON OF ALL EXPERIMENTS")

        print("=" * 60)

        print(
            f"Original network overload probability: "
            f"{original_probability:.4%}"
        )

        print(
            f"Modified network overload probability: "
            f"{modified_probability:.4%}"
        )

        print(
            f"Hospital server overload probability: "
            f"{hospital_probability:.4%}"
        )

        difference = (
            modified_probability -
            original_probability
        ) * 100

        print("\nCHANGE IN OVERLOAD PROBABILITY")

        print(
            f"Increase after reducing capacity: "
            f"{difference:.2f} percentage points"
        )

        print("\nALL EXPERIMENTS COMPLETED SUCCESSFULLY!")

        print("Results and graphs have been saved.")

    finally:

        sys.stdout = original_stdout


# ============================================================
# 6. DISPLAY OUTPUT FOLDER LOCATION
# ============================================================

print("\nOUTPUT FILES LOCATION:")

print(OUTPUT_DIR)

print("\nCheck your output folder for all saved files.")


# ============================================================
# END OF TASK 20
# ============================================================
