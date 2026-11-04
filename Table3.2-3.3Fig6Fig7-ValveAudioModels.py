import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Table 3.2: Valve Audio Models ---
table_3_2_data = [
    {"Valve ID": "id00", "Model": "YAMNet", "Accuracy": 0.96, "Precision": 0.83, "Recall": 0.79, "F1 Score": 0.81},
    {"Valve ID": "id00", "Model": "VGGish", "Accuracy": 0.89, "Precision": 0.50, "Recall": 0.08, "F1 Score": 0.14},
    {"Valve ID": "id00", "Model": "CREPE", "Accuracy": 0.8918, "Precision": 0.00, "Recall": 0.00, "F1 Score": 0.00},
    {"Valve ID": "id02", "Model": "YAMNet", "Accuracy": 0.97, "Precision": 0.913, "Recall": 0.875, "F1 Score": 0.89},
    {"Valve ID": "id02", "Model": "VGGish", "Accuracy": 0.89, "Precision": 0.87, "Recall": 0.29, "F1 Score": 0.44},
    {"Valve ID": "id02", "Model": "CREPE", "Accuracy": 0.8493, "Precision": 0.00, "Recall": 0.00, "F1 Score": 0.00},
    {"Valve ID": "id04", "Model": "YAMNet", "Accuracy": 0.95, "Precision": 0.93, "Recall": 0.54, "F1 Score": 0.68},
    {"Valve ID": "id04", "Model": "VGGish", "Accuracy": 0.88, "Precision": 0.00, "Recall": 0.00, "F1 Score": 0.00},
    {"Valve ID": "id04", "Model": "CREPE", "Accuracy": 0.89, "Precision": 0.00, "Recall": 0.00, "F1 Score": 0.00},
    {"Valve ID": "id06", "Model": "YAMNet", "Accuracy": 0.94, "Precision": 0.79, "Recall": 0.63, "F1 Score": 0.70},
    {"Valve ID": "id06", "Model": "VGGish", "Accuracy": 0.88, "Precision": 0.00, "Recall": 0.00, "F1 Score": 0.00},
    {"Valve ID": "id06", "Model": "CREPE", "Accuracy": 0.89, "Precision": 0.00, "Recall": 0.00, "F1 Score": 0.00},
]
print("--- Table 3.2: Audio Models across Valve IDs ---")
print(pd.DataFrame(table_3_2_data).to_string(index=False))

# --- Table 3.3: MIMII Machine Anomaly Detection ---
table_3_3_data = [
    {"Machine": "Fan", "Accuracy": 0.69, "Precision": 0.42, "Recall": 0.12, "F1 Score": 0.19, "AUC": 0.58},
    {"Machine": "Slider", "Accuracy": 0.98, "Precision": 0.97, "Recall": 0.94, "F1 Score": 0.96, "AUC": 1.00},
    {"Machine": "Valve", "Accuracy": 0.96, "Precision": 0.83, "Recall": 0.79, "F1 Score": 0.81, "AUC": 0.99},
    {"Machine": "Pump", "Accuracy": 0.95, "Precision": 0.82, "Recall": 0.78, "F1 Score": 0.80, "AUC": 0.99},
]
print("\n--- Table 3.3: Metrics across Machines ---")
print(pd.DataFrame(table_3_3_data).to_string(index=False))

# --- Figure 7: Confusion Matrices (YAMNet) ---
cms = {
    "Fan": np.array([[188, 14], [72, 10]]),
    "Slider": np.array([[212, 2], [4, 67]]),
    "Valve": np.array([[194, 4], [5, 19]]),
    "Pump": np.array([[194, 4], [5, 19]])
}

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for idx, (machine, cm) in enumerate(cms.items()):
    ax = axes[idx // 2, idx % 2]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax,
                xticklabels=["Normal", "Abnormal"], yticklabels=["Normal", "Abnormal"])
    ax.set_title(f"Confusion Matrix - YAMNet ({machine})")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

plt.tight_layout()
plt.show()