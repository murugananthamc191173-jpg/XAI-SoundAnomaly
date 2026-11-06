import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

# --- Table 3.1 Data Construction ---
table_3_1_data = [
    # id00
    {"Fan ID": "id00", "Model": "SVM", "Accuracy": 0.89, "Precision": 0.00, "Recall": 0.00, "F1 Score": 0.00},
    {"Fan ID": "id00", "Model": "RF", "Accuracy": 0.89, "Precision": 1.00, "Recall": 0.042, "F1 Score": 0.080},
    {"Fan ID": "id00", "Model": "XGBoost", "Accuracy": 0.91, "Precision": 0.67, "Recall": 0.33, "F1 Score": 0.44},
    # id02
    {"Fan ID": "id02", "Model": "SVM", "Accuracy": 0.74, "Precision": 0.00, "Recall": 0.00, "F1 Score": 0.00},
    {"Fan ID": "id02", "Model": "RF", "Accuracy": 0.92, "Precision": 1.00, "Recall": 0.70, "F1 Score": 0.83},
    {"Fan ID": "id02", "Model": "XGBoost", "Accuracy": 0.95, "Precision": 0.98, "Recall": 0.83, "F1 Score": 0.90},
    # id04
    {"Fan ID": "id04", "Model": "SVM", "Accuracy": 0.75, "Precision": 0.00, "Recall": 0.00, "F1 Score": 0.00},
    {"Fan ID": "id04", "Model": "RF", "Accuracy": 0.93, "Precision": 0.98, "Recall": 0.73, "F1 Score": 0.84},
    {"Fan ID": "id04", "Model": "XGBoost", "Accuracy": 0.94, "Precision": 0.93, "Recall": 0.81, "F1 Score": 0.87},
    # id06
    {"Fan ID": "id06", "Model": "SVM", "Accuracy": 0.74, "Precision": 0.00, "Recall": 0.00, "F1 Score": 0.00},
    {"Fan ID": "id06", "Model": "RF", "Accuracy": 0.996, "Precision": 1.00, "Recall": 0.99, "F1 Score": 0.99},
    {"Fan ID": "id06", "Model": "XGBoost", "Accuracy": 0.99, "Precision": 0.98, "Recall": 0.97, "F1 Score": 0.97},
]

df_table_3_1 = pd.DataFrame(table_3_1_data)
print("--- Table 3.1: Model Performance Comparison by Fan ID ---")
print(df_table_3_1.to_string(index=False))

# --- Figure 5: ROC Curves for Fan IDs ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fan_ids = ["id00", "id02", "id04", "id06"]
auc_scores = {
    "id00": {"XGBoost": 0.97, "SVM": 0.58, "RF": 0.97},
    "id02": {"SVM": 0.81, "RF": 0.71, "XGBoost": 0.79},
    "id04": {"XGBoost": 0.99, "SVM": 0.72, "RF": 0.99},
    "id06": {"XGBoost": 1.00, "SVM": 0.79, "RF": 1.00},
}

np.random.seed(42)
for idx, fid in enumerate(fan_ids):
    ax = axes[idx // 2, idx % 2]
    fpr_base = np.linspace(0, 1, 100)
    
    for model, color in zip(["XGBoost", "SVM", "RF"], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        auc_val = auc_scores[fid][model]
        # Synthetic ROC curve matching AUC
        tpr = np.power(fpr_base, (1 - auc_val) / auc_val) if auc_val < 1.0 else np.ones_like(fpr_base)
        tpr[0] = 0.0
        ax.plot(fpr_base, tpr, label=f"{model} (AUC = {auc_val:.2f})", color=color, lw=1.5)
        
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_title(f"ROC Curve Comparison - Fan {fid}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True)
    ax.legend(loc="lower right")

plt.tight_layout()
plt.show()