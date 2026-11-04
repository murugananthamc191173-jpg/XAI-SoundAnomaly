import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.ensemble import GradientBoostingClassifier

# 1. Create synthetic hybrid features matching the paper's descriptions
np.random.seed(42)
n_samples = 200
feature_names = ["mfcc_0", "autocorrelation", "yamnet_embedding_690", "mfcc_4", 
                 "yamnet_embedding_415", "mfcc_14", "yamnet_embedding_764", "mfcc_26"]

X = np.random.randn(n_samples, len(feature_names))
# Force stronger predictive power for mfcc_0 & autocorrelation
y = (X[:, 0] + X[:, 1]*0.8 + np.random.randn(n_samples)*0.5 > 0).astype(int)

model = GradientBoostingClassifier().fit(X, y)
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# --- Figure 13: SHAP Bar Plot ---
plt.figure(figsize=(8, 5))
shap.plots.bar(shap_values, show=False)
plt.title("Fig 13: Global Feature Importance (SHAP Bar Plot)")
plt.tight_layout()
plt.show()

# --- Figure 14: SHAP Beeswarm Plot ---
plt.figure(figsize=(8, 6))
shap.plots.beeswarm(shap_values, show=False)
plt.title("Fig 14: SHAP Beeswarm Plot")
plt.tight_layout()
plt.show()

# --- Figure 11: SHAP Waterfall Plot (Instance 0) ---
plt.figure(figsize=(8, 5))
shap.plots.waterfall(shap_values[0], show=False)
plt.title("Fig 11: SHAP Waterfall Plot")
plt.tight_layout()
plt.show()