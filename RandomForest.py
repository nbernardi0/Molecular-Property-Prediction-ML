# Random Forest Regression on Molecular Properties (with optional k-fold cross-validation + memory monitoring + memory integral)
import os
import numpy as np
import time
import psutil
import threading
import pandas as pd
import matplotlib.pyplot as plt
from numpy import trapz   # per l'integrale numerico

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from Plots import plot_parity_plots, plot_rf_r2_vs_time

# ================= Configuration =================
info = "FR"                # "atom", "FG", "FR"
n_estimators_list = [100]  # multiple n_estimators can be used
k = 5
CV = False                 # True = use cross-validation, False = use single train/test split

cv = KFold(n_splits=k, shuffle=True, random_state=42)

# ------------- Path Management -------------
model_dir = os.path.join("Results", "ML", "RF", info)
os.makedirs(model_dir, exist_ok=True)
plots_dir = os.path.join("Plots", "ML", "RF", info)
os.makedirs(plots_dir, exist_ok=True)

# ================= Data Loading =================
X = np.load("X_features_filtered.npy")
Y_labels = np.load("Y_labels.npy")

num_atoms = 5
num_func_groups = 20

# input selection
if info == "atom":
    X = X[:, :num_atoms]
elif info == "FG":
    X = X[:, :num_atoms + num_func_groups]
elif info == "FR":
    X = X
else:
    raise ValueError("info must be 'atom', 'FG', or 'FR'")

# Train/test split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_labels, test_size=0.2, random_state=42)

property_names = ['μ (D)', 'α (a₀³)', 'ε_HOMO (Ha)', 'ε_LUMO (Ha)','ε_gap (Ha)', '⟨r²⟩ (a₀²)', 'zpve (Ha)', 'U₀ (Ha)','U (Ha)', 'H (Ha)', 'G (Ha)', 'Cᵥ (cal/mol·K)']

# ================= Memory Monitoring =================
mem_log = []
stop_flag = False

def monitor_memory(interval=1):
    process = psutil.Process(os.getpid())
    while not stop_flag:
        mem_mb = process.memory_info().rss / (1024**2)  # RSS in MB
        mem_log.append((time.time(), mem_mb))
        time.sleep(interval)

# Start memory monitor thread
monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
monitor_thread.start()

# ================= Loop over n_estimators =================
results = []
for n_estimators in n_estimators_list:
    print(f"\n[INFO] Training Random Forest with n_estimators = {n_estimators}")

    start_time = time.time()
    r2_scores_mean = []
    r2_scores_std = []
    feature_importances_all = []

    if CV:  # ---------- CROSS-VALIDATION MODE ----------
        for i in range(Y_labels.shape[1]): # loop over each single property
            model = RandomForestRegressor(n_estimators, random_state=42, n_jobs=-1)
            scores = cross_val_score(model, X, Y_labels[:, i], cv=cv, scoring="r2")
            r2_scores_mean.append(np.mean(scores))
            r2_scores_std.append(np.std(scores))

        # No parity plot or feature importance here (CV does not produce Y_pred)
        r2_global = np.mean(r2_scores_mean)
        training_time = time.time() - start_time

    else:   # ---------- TRAIN/TEST SPLIT MODE ----------
        Y_pred = np.zeros_like(Y_test)

        for i in range(Y_labels.shape[1]):
            model = RandomForestRegressor(n_estimators, random_state=42, n_jobs=-1)
            model.fit(X_train, Y_train[:, i])
            Y_pred[:, i] = model.predict(X_test)
            feature_importances_all.append(model.feature_importances_)

        feature_importances_all = np.array(feature_importances_all)

        training_time = time.time() - start_time
        r2_global = r2_score(Y_test, Y_pred, multioutput='uniform_average')

        # R2 for parity plots
        r2_scores = [r2_score(Y_test[:, i], Y_pred[:, i]) for i in range(Y_test.shape[1])]

        # Plotting
        plot_parity_plots(Y_test, Y_pred, r2_scores, property_names, plots_dir)

        # ================= Feature Importances =================
        fig, axes = plt.subplots(4, 3, figsize=(18, 12))
        fig.suptitle(f'Feature Importance per Property', fontsize=18)
        for i, ax in enumerate(axes.ravel()):
            ax.bar(range(X.shape[1]), feature_importances_all[i],
                   color='skyblue', edgecolor='k')
            ax.set_title(property_names[i], fontsize=12)
            ax.set_xlabel('Feature Index', fontsize=9)
            ax.set_ylabel('Importance', fontsize=9)
            ax.grid(True, axis='y')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(plots_dir,
                    f"feature_importance_RF_{n_estimators}.pdf"),
                    format="pdf", bbox_inches="tight")
        plt.close()

    # Save results
    results.append({
        "n_estimators": n_estimators,
        "R2_global": r2_global,
        "Training_time": training_time,
        "R2_means": r2_scores_mean,
        "R2_stds": r2_scores_std
    })

# ================= Stop Memory Monitoring =================
stop_flag = True
monitor_thread.join()

# Save memory usage log
mem_df = pd.DataFrame(mem_log, columns=["timestamp", "memory_MB"])
mem_df["time_s"] = mem_df["timestamp"] - mem_df["timestamp"].iloc[0]
mem_csv_path = os.path.join(model_dir, "memory_usage.csv")
mem_df.to_csv(mem_csv_path, index=False)

# ================= Memory Integral =================
mem_area = trapz(mem_df["memory_MB"], x=mem_df["time_s"])
print(f"[INFO] Integrated memory usage (area under curve) = {mem_area:.2f} MB*s")

# Plot memory usage
plt.figure(figsize=(8, 5))
plt.plot(mem_df["time_s"], mem_df["memory_MB"], label="RAM used (MB)")
plt.xlabel("Time [s]")
plt.ylabel("Memory [MB]")
plt.title("Memory usage during Random Forest training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "memory_usage.png"))
plt.show()

# ================= Save summary =================
summary_file = os.path.join(model_dir, "results_summary.txt")
with open(summary_file, "w", encoding="utf-8") as f:
    f.write("Random Forest Results Summary\n")
    f.write("============================\n\n")
    for res in results:
        f.write(f"n_estimators = {res['n_estimators']}\n")
        f.write(f"  R2_global      = {res['R2_global']:.4f}\n")
        f.write(f"  Training time  = {res['Training_time']:.2f} s\n")
        if CV:
            f.write(f"  CV R2 means    = {res['R2_means']}\n")
            f.write(f"  CV R2 stds     = {res['R2_stds']}\n")
        f.write("----------------------------\n")
    f.write(f"\nIntegrated memory usage = {mem_area:.2f} MB*s\n")

print(f"\n[INFO] Summary saved to {summary_file}")

# ================= Global Plot (R2 vs time) =================
n_estimators_vals = [res["n_estimators"] for res in results]
R2_global_vals = [res["R2_global"] for res in results]
Training_time_vals = [res["Training_time"] for res in results]

plots_dir = os.path.join(plots_dir, info)
plot_rf_r2_vs_time(n_estimators_vals, R2_global_vals, Training_time_vals, plots_dir)
