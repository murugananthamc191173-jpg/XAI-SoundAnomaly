import pandas as pd

table_3_7_data = [
    {"Year": "2019", "Paper / Work": "MIMII Baseline (Purohit et al.)", "Model / Technique": "Autoencoder", "Dataset / Machine": "MIMII (Fan, Pump, Valve, Slider)", "Metrics Reported": "AUC", "Performance Summary": "Fan=65.83%, Pump=72.89%, Valve=66.28%, Slider≈70%", "XAI Used": "No"},
    {"Year": "2020", "Paper / Work": "Jalali et al. (DCASE2020)", "Model / Technique": "Ensemble Autoencoders + Fusion", "Dataset / Machine": "DCASE2020 / MIMII", "Metrics Reported": "AUC", "Performance Summary": "Mean AUC ≈ 73.51%", "XAI Used": "No"},
    {"Year": "2021", "Paper / Work": "Dohi et al.", "Model / Technique": "Normalizing Flows (MAF, Glow)", "Dataset / Machine": "MIMII (overall)", "Metrics Reported": "AUC", "Performance Summary": "+4.6% (MAF), +5.8% (Glow) vs baseline", "XAI Used": "No"},
    {"Year": "2021", "Paper / Work": "Thoidis et al.", "Model / Technique": "Semi-supervised", "Dataset / Machine": "MIMII (overall)", "Metrics Reported": "AUC", "Performance Summary": "Mean AUC ≈ 0.91", "XAI Used": "No"},
    {"Year": "2022", "Paper / Work": "Liu et al. (STgram-MFN)", "Model / Technique": "Spectral-Temporal + MFN", "Dataset / Machine": "MIMII (Fan, Pump, Slider, Valve)", "Metrics Reported": "AUC", "Performance Summary": "Fan=81.39%, Pump=83.48%, Slider=98.22%, Valve=98.83%", "XAI Used": "No"},
    {"Year": "2023", "Paper / Work": "Choi & Choi (Noisy-ArcMix)", "Model / Technique": "Angular Loss + Mixup", "Dataset / Machine": "DCASE2020 / MIMII", "Metrics Reported": "AUC", "Performance Summary": "≈ 94.6% AUC (avg)", "XAI Used": "No"},
    {"Year": "2025", "Paper / Work": "Wu et al.", "Model / Technique": "Few-Shot ASD", "Dataset / Machine": "MIMII (all types)", "Metrics Reported": "AUC", "Performance Summary": "Competitive per-type AUCs (> 90%)", "XAI Used": "No"},
    {"Year": "2025", "Paper / Work": "Proposed (Current Work)", "Model / Technique": "XGBoost + ZOA + Diffusion + SHAP/LIME", "Dataset / Machine": "MIMII (Fan, Valve, Pump, Slider)", "Metrics Reported": "Acc / Prec / Rec / F1", "Performance Summary": "Fan(Acc=0.79, F1=0.61); Valve(0.96, 0.83); Pump(0.97, 0.90); Slider(0.99, 0.98)", "XAI Used": "Yes (SHAP + LIME)"}
]

df_3_7 = pd.DataFrame(table_3_7_data)
print("--- Table 3.7: Comparison Chart with Prior Works ---")
print(df_3_7.to_string(index=False))