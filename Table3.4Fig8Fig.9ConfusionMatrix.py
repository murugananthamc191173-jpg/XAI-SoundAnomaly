import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Table 3.4 Data ---
table_3_4_data = [
    {"Machine": "Fan", "Optimal Threshold": 0.264, "Accuracy": 0.79, "Precision": 0.66, "Recall": 0.57, "F1 Score": 0.61, "AUC": 0.98},
    {"Machine": "Valve", "Optimal Threshold": 0.085, "Accuracy": 0.96, "Precision": 0.71, "Recall": 1.00, "F1 Score": 0.83, "AUC": 0.99},
    {"Machine": "Pump", "Optimal Threshold": 0.146, "Accuracy": 0.97, "Precision": 0.87, "Recall": 0.93, "F1 Score": 0.90, "AUC": 1.00},
    {"Machine": "Slider", "Optimal Threshold": 0.090, "Accuracy": 0.99, "Precision": 0.99, "Recall": 0.97, "F1 Score": 0.98, "AUC": 0.99},
]
df_3_4 = pd.DataFrame(table_3_4_data)
print("--- Table 3.4: XGBoost Anomaly Detection Performance ---")
print(df_3_4.to_string(index=False))

# --- Figure 9: Confusion Matrices for XGBoost ---
cms_xgb = {
    "Fan": np.array([[178, 24], [35, 47]]),
    "Slider": np.array([[188, 10], [0, 24]]),
    "Valve": np.array([[197, 4], [2, 27]]),
    "Pump": np.array([[213, 1], [2, 69]])
}

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for idx, (machine, cm) in enumerate(cms_xgb.items()):
    ax = axes[idx // 2, idx % 2]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax,
                xticklabels=["Normal", "Abnormal"], yticklabels=["Normal", "Abnormal"])
    ax.set_title(f"Confusion Matrix - XGBoost ({machine})")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

plt.tight_layout()
plt.show()