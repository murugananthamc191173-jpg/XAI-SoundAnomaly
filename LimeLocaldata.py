import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data for LIME local explanation (from Fig 16 & 17)
lime_features = [
    "mfcc_0 > 1.05",
    "autocorrelation > 0.55",
    "yamnet_embedding_813 <= -0.02",
    "yamnet_embedding_11 <= -0.02",
    "yamnet_embedding_803 <= -0.02",
    "yamnet_embedding_168 <= -0.02",
    "yamnet_embedding_766 <= -0.03",
    "yamnet_embedding_196 <= -0.02",
    "yamnet_embedding_165 <= -0.03",
    "yamnet_embedding_604 <= -0.02"
]

weights = [0.33, 0.31, 0.28, -0.27, 0.25, 0.22, 0.19, 0.18, -0.18, 0.12]

df_lime = pd.DataFrame({"Feature Rule": lime_features, "Weight": weights})
print("--- LIME Table Explanation ---")
print(df_lime.to_string(index=False))

# --- Figure 17: LIME Bar Chart Plot ---
plt.figure(figsize=(9, 5))
colors = ['green' if w > 0 else 'red' for w in weights]
plt.barh(lime_features[::-1], weights[::-1], color=colors[::-1])
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel("Local Contribution Weight")
plt.title("Fig 17: Local Explanation for Class Anomalous (LIME Graph)")
plt.tight_layout()
plt.show()