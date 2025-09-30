import os
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from Plots import plot_parity_plots, plot_coefficients_grid

# ------------- Configuration -------------
info = "FR"                   # "atom", "FG", "FR"
model_choice = "ridge"        # "linear", "ridge", "lasso"
alpha_lasso = 0.0001          
alpha_ridge = 1
k = 5                         # fold numbers for cross validation

# ------------- Path Management -------------
model_dir = os.path.join("Results", "ML", info, model_choice)  # folder for model results
os.makedirs(model_dir, exist_ok=True)
plots_dir = os.path.join("Plots", "ML", info, model_choice)   # folder for plots
os.makedirs(plots_dir, exist_ok=True)

# ------------- Data Loading -------------
X = np.load("X_features_filtered.npy")
Y_labels = np.load("Y_labels.npy")

num_atoms = 5
num_func_groups = 20  
if info == "atom":
    X = X[:, :num_atoms]
elif info == "FG":
    X = X[:, :num_atoms + num_func_groups]
elif info == "FR":
    X = X
else:
    raise ValueError("info must be 'atom', 'FG', or 'FR'")

# ------------- Cross Validation -------------
kf = KFold(n_splits=k, shuffle=True, random_state=42)  # K-Fold cross-validation
fold_mse = []
r2_matrix = []  # to store R2 scores for each fold and property

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):  # here we loop over each fold to scale each fold

    X_train, X_val = X[train_idx], X[val_idx]                # we split according to the fold
    Y_train, Y_val = Y_labels[train_idx], Y_labels[val_idx]

    # Scaling per fold
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)

    scaler_Y = MinMaxScaler()
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    Y_val_scaled = scaler_Y.transform(Y_val)

    if model_choice == "linear":
        model_obj = LinearRegression()
    elif model_choice == "ridge":
        model_obj = Ridge(alpha=alpha_ridge)
    elif model_choice == "lasso":
        model_obj = Lasso(alpha=alpha_lasso, max_iter=500)
    else:
        raise ValueError("Invalid model_choice")

    model_obj.fit(X_train_scaled, Y_train_scaled)
    Y_val_pred_scaled = model_obj.predict(X_val_scaled)

    # MSE on scaled data
    fold_mse.append(mean_squared_error(Y_val_scaled, Y_val_pred_scaled))
    # R2 for each property on scaled data
    r2_fold = [r2_score(Y_val_scaled[:, i], Y_val_pred_scaled[:, i]) for i in range(Y_val.shape[1])] # r2 for each property by 
    r2_matrix.append(r2_fold)

# Mean and standard deviation for MSE and R2
mean_cv_mse = np.mean(fold_mse)
std_cv_mse = np.std(fold_mse)
r2_matrix = np.array(r2_matrix)               
mean_r2_scores = np.mean(r2_matrix, axis=0) # axis=0 to sum the different rows
std_r2_scores = np.std(r2_matrix, axis=0)

# ------------- Train Final Model ------------- <--- Over entire dataset with no validation split (to produce coefficients and parity plots)
scaler_X_final = MinMaxScaler()
X_scaled = scaler_X_final.fit_transform(X)
scaler_Y_final = MinMaxScaler()
Y_scaled = scaler_Y_final.fit_transform(Y_labels)

start = timer()
if model_choice == "linear":
    final_model = LinearRegression()
elif model_choice == "ridge":
    final_model = Ridge(alpha=alpha_ridge)
elif model_choice == "lasso":
    final_model = Lasso(alpha=alpha_lasso, max_iter=5000)
else:
    raise ValueError("Invalid model_choice")

final_model.fit(X_scaled, Y_scaled)
training_time = timer() - start
Y_pred_scaled = final_model.predict(X_scaled)
mse = mean_squared_error(Y_scaled, Y_pred_scaled)

Y_pred_original = scaler_Y_final.inverse_transform(Y_pred_scaled)

# ------------------ Plot section ----------------------
property_names = ['μ (D)', 'α (a₀³)', 'ε_HOMO (Ha)', 'ε_LUMO (Ha)','ε_gap (Ha)','⟨r²⟩ (a₀²)', 'zpve (Ha)', 'U₀ (Ha)','U (Ha)', 'H (Ha)', 'G (Ha)', 'Cᵥ (cal/mol·K)']

plot_parity_plots(Y_labels, Y_pred_original, mean_r2_scores, property_names, plots_dir)

coef = final_model.coef_
plot_coefficients_grid(coef, property_names, model_choice, plots_dir,alpha=(alpha_ridge if model_choice=="ridge" else alpha_lasso))

# ------------- Save results -------------
results_file = os.path.join(model_dir, "results_cv.txt") # to open the results folder and save the result file
with open(results_file, "w", encoding="utf-8") as f:
    f.write("===== Regression Results with K-Fold (scaled) =====\n\n")
    f.write(f"Model type: {model_choice}\n")
    if model_choice in ["ridge", "lasso"]:
        f.write(f"Alpha: {alpha_ridge if model_choice=='ridge' else alpha_lasso}\n")
    f.write(f"Mean CV MSE (scaled): {mean_cv_mse:.6f} ± {std_cv_mse:.6f}\n")
    f.write(f"MSE (global, scaled 0-1): {mse:.6f}\n")
    f.write(f"Training time: {training_time:.2f} seconds\n\n")

    f.write("Mean ± Std R2 scores per property (scaled):\n")
    for name, r2m, r2s in zip(property_names, mean_r2_scores, std_r2_scores):
        f.write(f"{name}: {r2m:.4f} ± {r2s:.4f}\n")

print(f"[INFO] Results saved in {results_file}")
