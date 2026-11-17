"""
BIT 2118 Scientific Computing - Task A: NumPy Arrays and Numerical Computation

Part 1: Original problem (weather station, 24 hourly temperatures)
Part 2: Changing the inputs and observing the results (Instruction C)
Part 3: Related real-world case: cold-storage room monitoring (Instruction E)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # lets the script save the plot without opening a window
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# PART 1: ORIGINAL PROBLEM
# ---------------------------------------------------------------
print("=" * 60)
print("PART 1: WEATHER STATION (original problem)")
print("=" * 60)

# Input: 24 hourly temperature readings in degrees Celsius
temperatures_c = np.array([
    18.5, 18.0, 17.8, 17.5, 17.9, 19.0,
    21.5, 23.8, 25.6, 27.1, 28.3, 29.0,
    29.4, 29.1, 28.5, 27.6, 26.2, 24.8,
    23.4, 22.1, 21.0, 20.2, 19.6, 19.0
])

# Operations: mean, minimum, maximum
mean_temp = np.mean(temperatures_c)
min_temp = np.min(temperatures_c)
max_temp = np.max(temperatures_c)

# Vectorized Celsius-to-Fahrenheit conversion: F = 1.8C + 32
temperatures_f = 1.8 * temperatures_c + 32

print(f"Daily mean temperature: {mean_temp:.2f} C")
print(f"Minimum temperature: {min_temp:.2f} C")
print(f"Maximum temperature: {max_temp:.2f} C")
print("First six Fahrenheit readings:", temperatures_f[:6])


# ---------------------------------------------------------------
# PART 2: CHANGING THE INPUTS (Instruction C)
# ---------------------------------------------------------------
print()
print("=" * 60)
print("PART 2: CHANGING THE INPUT VALUES")
print("=" * 60)


def summarize(label, readings):
    """Print mean, min and max for a set of readings."""
    print(f"{label:<38} mean={np.mean(readings):6.2f}  "
          f"min={np.min(readings):6.2f}  max={np.max(readings):6.2f}")


summarize("Original", temperatures_c)

# Change 1: add 10 C to every reading (a hotter day)
summarize("All readings +10 C", temperatures_c + 10)

# Change 2: replace the highest reading with an outlier of 60 C
with_outlier = temperatures_c.copy()
with_outlier[np.argmax(with_outlier)] = 60.0
summarize("Highest reading changed to 60 C", with_outlier)

# Change 3: constant temperature all day
summarize("Constant 20 C all day", np.full(24, 20.0))

# Change 4: freezing day (subtract 25 C from every reading)
summarize("All readings -25 C", temperatures_c - 25)


# ---------------------------------------------------------------
# PART 3: RELATED REAL-WORLD CASE (Instruction E)
# Cold-storage room monitoring
# ---------------------------------------------------------------
print()
print("=" * 60)
print("PART 3: COLD-STORAGE ROOM MONITORING")
print("=" * 60)

# Assumptions:
#   1. A sensor records the room temperature once every hour for 24 hours.
#   2. Readings are in degrees Celsius.
#   3. Stored produce is safe only at or below 5.0 C.
#   4. The door is opened for deliveries around hours 8-10, so the
#      temperature rises then.
SAFE_LIMIT_C = 5.0

room_c = np.array([
    3.2, 3.0, 2.9, 3.1, 3.0, 3.3,
    3.5, 4.2, 5.8, 6.4, 5.1, 4.3,
    3.9, 3.6, 3.5, 3.4, 3.6, 3.7,
    3.5, 3.3, 3.2, 3.1, 3.0, 3.0
])
hours = np.arange(24)

room_f = 1.8 * room_c + 32

above_limit = room_c > SAFE_LIMIT_C          # boolean array (True/False per hour)
hours_above = int(np.sum(above_limit))       # True counts as 1
worst_hour = int(np.argmax(room_c))

print(f"Mean temperature: {np.mean(room_c):.2f} C ({np.mean(room_f):.2f} F)")
print(f"Minimum temperature: {np.min(room_c):.2f} C")
print(f"Maximum temperature: {np.max(room_c):.2f} C at hour {worst_hour}")
print(f"Safe limit: {SAFE_LIMIT_C:.1f} C")
print(f"Hours above the safe limit: {hours_above}")
print("Hours where limit was exceeded:", hours[above_limit])

if hours_above > 0:
    print("Interpretation: the room went above the safe limit, so the "
          "produce may be at risk. Check the door and cooling system.")
else:
    print("Interpretation: the room stayed within the safe limit all day.")

# Change an input: what if the safe limit were stricter (4.0 C)?
stricter_limit = 4.0
print(f"\nWith a stricter limit of {stricter_limit:.1f} C, hours above the "
      f"limit: {int(np.sum(room_c > stricter_limit))}")

# Graph: temperature against hour, with the safe limit as a dashed line
plt.figure(figsize=(9, 4.5))
plt.plot(hours, room_c, marker="o", label="Room temperature")
plt.axhline(SAFE_LIMIT_C, color="red", linestyle="--",
            label=f"Safe limit ({SAFE_LIMIT_C} C)")
plt.scatter(hours[above_limit], room_c[above_limit], color="red", zorder=3,
            label="Above limit")
plt.title("Cold-Storage Room Temperature Over 24 Hours")
plt.xlabel("Hour of day")
plt.ylabel("Temperature (C)")
plt.xticks(hours)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
# Save the graph in the repo's "output" folder (next to the "code" folder)
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "cold_storage_plot.png"), dpi=150)
print("\nGraph saved as output/cold_storage_plot.png")