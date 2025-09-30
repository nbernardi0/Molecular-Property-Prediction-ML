import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import os
import json
import time
from sklearn.metrics import r2_score

# SETTINGS
epochs = 2000
patience = int(0.05 * epochs)
activation_function = "tanh"
info = "FR"                    # "atom", "FG", "FR"
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                 # Seed for multiple runs

#architectures = {2500: {1: [27], 2: [22, 22], 3: [19, 19, 19]},  # dictionary: target_params -> {num_layers: [hidden_layer_sizes]} ex. architectures[2500][2]=[22,22]
#                 5000: {1: [54], 2: [38, 38], 3: [32, 32, 32]},
#                 7500: {1: [81], 2: [51, 51], 3: [42, 42, 42]},
#                 10000: {1: [107], 2: [63, 63], 3: [51, 51, 51]},
#                 12500: {1: [134], 2: [74, 74], 3: [59, 59, 59]} }

architectures = {5000: {1: [54]}}

property_names = ['μ (D)', 'α (a0^3)', 'ε_HOMO (Ha)', 'ε_LUMO (Ha)', 'ε_gap (Ha)', '<r^2> (a0^2)', 'zpve (Ha)', 'U0 (Ha)', 'U (Ha)', 'H (Ha)', 'G (Ha)', 'Cv (cal/molK)'] # property names

# DATA LOADING AND SCALING
X = np.load("X_features_filtered.npy")
Y_labels = np.load("Y_labels.npy")

num_atoms = 5
num_func_groups = 20  # after filtering

# input selection
if info == "atom":   
    X = X[:, :num_atoms]                   # Select atomic features
elif info == "FG":
    X = X[:, :num_atoms + num_func_groups] # Select atomic and functional group features
elif info == "FR":
    X = X                                  # Select all features
else:
    raise ValueError("info must be 'atom', 'FG', or 'FR'")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_labels, test_size=0.2, random_state=42) # Split the data into training and testing sets

scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train) # we fit over X_train to avoid data leakage
X_test_scaled = scaler_X.transform(X_test)

scaler_Y = MinMaxScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

# LOOP OVER ARCHITECTURES
results_summary = [] # to store results

for target_params, layers_dict in architectures.items(): # for key, object in dictionary.items():
    for num_layers, hidden_layers in layers_dict.items():
        print(f"\n===== Training Target={target_params}, Layers={num_layers}, Arch={hidden_layers} =====")

        run_r2_all = []         # R2 for each property
        run_mse_scaled = []     # Scaled MSE
        run_mse_original = []   # Original MSE
        run_times = []

        for seed in SEEDS:
            tf.keras.utils.set_random_seed(seed) # to make results reproducible

            # Model definition
            net = tf.keras.models.Sequential()
            net.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
            for units in hidden_layers:
                net.add(tf.keras.layers.Dense(units, activation=activation_function))
            net.add(tf.keras.layers.Dense(Y_labels.shape[1], activation="linear"))

            net.compile(optimizer='adam', loss='mse')

            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience,restore_best_weights=True) # restore_best_weights=True to use the best model weights

            start_time = time.time() # start time measurement
            history = net.fit(X_train_scaled, Y_train_scaled, validation_split=0.2, epochs=epochs, batch_size=32, verbose=1,callbacks=[early_stop])
            training_time = time.time() - start_time # end time measurement

            # Predictions
            Y_pred_scaled = net.predict(X_test_scaled, verbose=0)
            Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)    # back to original scale

            # Metrics (for 1 seed)
            r2_scores = [r2_score(Y_test[:, i], Y_pred[:, i]) for i in range(Y_test.shape[1])] # Y_test/pred[:, i] corresponds to each property, Y_test.shape[1]=12 (tot num of properties)
            mse_test_scaled = np.mean((Y_test_scaled - Y_pred_scaled) ** 2)
            mse_test_original = np.mean((Y_test - Y_pred) ** 2)

            run_r2_all.append(r2_scores)                 # run_r2_all = [[seed1_prop1, seed1_prop2,...], [seed2_prop1, seed2_prop2,...], ...]
            run_mse_scaled.append(mse_test_scaled)
            run_mse_original.append(mse_test_original)
            run_times.append(training_time)

        # Medie e std per proprietà
        r2_mean_all = np.mean(run_r2_all, axis=0)    # mean for each property (ex. mean(seed1_prop1, seed2_prop1,...))
        r2_std_all = np.std(run_r2_all, axis=0)      # std for each property

        # Append results
        results_summary.append({                    # results_summary is a list of dictionaries (each dictionary corresponds to one architecture)
            "TargetParams": int(target_params),
            "NumLayers": int(num_layers),
            "HiddenLayers": [int(x) for x in hidden_layers],
            "ParamsActual": int(target_params),

            # Property metrics
            "PropertyNames": property_names,
            "R2_mean_all": r2_mean_all.tolist(),
            "R2_std_all": r2_std_all.tolist(),

            # Global metrics
            "MSE_global_scaled_mean": float(np.mean(run_mse_scaled)),
            "MSE_global_scaled_std": float(np.std(run_mse_scaled)),
            "MSE_global_original_mean": float(np.mean(run_mse_original)),
            "MSE_global_original_std": float(np.std(run_mse_original)),

            # Training time
            "Time_mean": float(np.mean(run_times)),
            "Time_std": float(np.std(run_times))
        })

        # Save single-architecture summary
        out_dir = os.path.join("Results_multi", info, activation_function, f"{target_params}_params", f"{num_layers}layers")
        os.makedirs(out_dir, exist_ok=True)  # create directory if it doesn't exist
        with open(os.path.join(out_dir, "summary.txt"), "w") as f: f.write(json.dumps(results_summary[-1], indent=4)) # save in a .txt file the last dictionary added to results_summary

# Final Global Summary
summary_file = os.path.join(out_dir, "summary.txt")
with open(summary_file, "w", encoding="utf-8") as f: # "w" to open the file in write mode, "utf-8" to support special characters,  "as f" to create a file object
    # General settings
    f.write("==== GLOBAL EXPERIMENT SETTINGS ====\n\n")
    f.write(f"Info type           : {info}\n")
    f.write(f"Activation function : {activation_function}\n")
    f.write(f"Epochs              : {epochs}\n")
    f.write(f"Patience            : {patience}\n")
    f.write(f"Training samples    : {X_train.shape[0]}\n")
    f.write(f"Test samples        : {X_test.shape[0]}\n\n")

    # Model results
    for res in results_summary: # we iterate over the list of dictionaries
        f.write("--------------------------------------------------\n")
        f.write(f"Target parameters : {res['TargetParams']}\n")
        f.write(f"Actual parameters : {res['ParamsActual']}\n")
        f.write(f"Num layers        : {res['NumLayers']}\n")
        f.write(f"Hidden layers     : {res['HiddenLayers']}\n\n")

        # R2 for each property
        f.write("R² per property (mean ± std)\n")
        f.write("----------------------------\n")
        for pname, r2m, r2s in zip(
            res["PropertyNames"], res["R2_mean_all"], res["R2_std_all"]
        ):
            f.write(f"{pname:15s}: {r2m:.4f} ± {r2s:.4f}\n")
        # Global metrics
        f.write("\nGlobal Metrics on Test Set\n")
        f.write("==========================\n")
        f.write(f"MSE (scaled, all properties)   = {res['MSE_global_scaled_mean']:.6f} ± {res['MSE_global_scaled_std']:.6f}\n")
        f.write(f"MSE (original, all properties) = {res['MSE_global_original_mean']:.6f} ± {res['MSE_global_original_std']:.6f}\n")

        # Training time
        f.write(f"\nAverage training time: {res['Time_mean']:.2f}s ± {res['Time_std']:.2f}s\n")
        f.write("--------------------------------------------------\n\n")

print(f"[INFO] Global summary saved to {summary_file}")
