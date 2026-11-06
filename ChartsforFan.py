import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Table 3.5 & 3.6 Data ---
metaheuristic_data = {
    "Fan": {
        "Optimizer": ["PSO", "WOA", "HHO", "ZOA"],
        "Accuracy": [0.9773, 0.9745, 0.9788, 0.9816],
        "Best Fitness": [0.0227, 0.0255, 0.0212, 0.0184],
        "Selection Size": [790, 920, 240, 550],
        "Execution Time": [2250, 2040, 1050, 2300]
    },
    "Pump": {
        "Optimizer": ["WOA", "HHO", "PSO", "ZOA"],
        "Accuracy": [0.8856, 0.9502, 0.8800, 0.9080],
        "Best Fitness": [0.1144, 0.0498, 0.1200, 0.0920],
        "Selection Size": [520, 550, 780, 490],
        "Execution Time": [900, 1450, 1480, 1290]
    },
    "Valve": {
        "Optimizer": ["PSO", "WOA", "HHO", "ZOA"],
        "Accuracy": [0.9903, 0.9871, 0.9871, 0.9935],
        "Best Fitness": [0.0097, 0.0129, 0.0129, 0.0065],
        "Selection Size": [740, 790, 510, 230],
        "Execution Time": [1100, 990, 980, 900]
    },
    "Slider": {
        "Optimizer": ["PSO", "WOA", "HHO", "ZOA"],
        "Accuracy": [0.9864, 0.9847, 0.9830, 0.9915],
        "Best Fitness": [0.0136, 0.0153, 0.0170, 0.0085],
        "Selection Size": [730, 140, 510, 460],
        "Execution Time": [1900, 400, 1680, 1480]
    }
}

# Print Tables
for machine, metrics in metaheuristic_data.items():
    print(f"\n--- Comparative Optimization Analysis for {machine} ---")
    df = pd.DataFrame(metrics)
    print(df.to_string(index=False))

# Plot Table 3.6 Charts for Fan
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
colors = ["blue", "orange", "green", "red"]

fan_df = pd.DataFrame(metaheuristic_data["Fan"])

axes[0, 0].bar(fan_df["Optimizer"], fan_df["Accuracy"], color=colors)
axes[0, 0].set_title("Comparison of Accuracy")
axes[0, 0].set_ylim(0, 1.05)

axes[0, 1].bar(fan_df["Optimizer"], fan_df["Best Fitness"], color=colors)
axes[0, 1].set_title("Comparison of Best Fitness")

axes[1, 0].bar(fan_df["Optimizer"], fan_df["Selection Size"], color=colors)
axes[1, 0].set_title("Comparison of Selection Size")

axes[1, 1].bar(fan_df["Optimizer"], fan_df["Execution Time"], color=colors)
axes[1, 1].set_title("Comparison of Execution Time")

plt.tight_layout()
plt.show()