import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import os
import json
import time
import psutil
import threading
from Utils import extract_history, save_experiment_info
from sklearn.metrics import r2_score
from Plots import plot_training_curves, plot_parity_plots

# ---------------- Training Setup --------------------
epochs = 20
patience = int(0.2 * epochs)    # 20 % of total epochs
activation_function = "tanh"    # "relu", "selu", "tanh"
info = "FR"                     # "atom", "FG", "FR"

hidden_layers = [54]                   # [h1, h2 ...] list of hidden layer sizes
Load_model = False                     # whether to load a pre-trained model or train from scratch

# ---------------- Memory Monitoring -----------------
mem_log = []
stop_flag = False

def monitor_memory(interval=1):
    process = psutil.Process(os.getpid())
    while not stop_flag:
        mem_mb = process.memory_info().rss / (1024**2)  # RSS = RAM used in MB
        mem_log.append((time.time(), mem_mb))
        time.sleep(interval)
# To start monitor RAM usage
monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
monitor_thread.start()

# ------------ Path Management (model saving/loading) ---------------
num_layers = len(hidden_layers)
num_neurons = "_".join(map(str, hidden_layers))         # create a descriptive string from the list of hidden layers hidden_layers --> ex. [h1, h2, h3] -> "h1_h2_h3"
subfolder = f"{num_layers}layers_{num_neurons}neurons"  # ex. "3layers_50_50_50neurons" (name of the subfolder to save plots and model)

base_dir = "saved_models_NN"
model_dir = os.path.join(base_dir, info, activation_function, subfolder)
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model.keras")             # where the model is saved/loaded
history_path = os.path.join(model_dir, "training_history.json") # where the training history is saved/loaded (JSON file)

Results_dir = os.path.join("Results", "NN", "RF", info, activation_function, subfolder)
os.makedirs(Results_dir, exist_ok=True)

# ------------ Data Loading and Preprocessing ---------------
X = np.load("X_features_filtered.npy") # Load preprocessed data
Y_labels = np.load("Y_labels.npy")     # Load labels

num_atoms = 5
num_func_groups = 17  # after filtering, 17 functional groups remain

# input selection
if info == "atom":   
    X = X[:, :num_atoms]                   # Select atomic features
elif info == "FG":
    X = X[:, :num_atoms + num_func_groups] # Select atomic and functional group features
elif info == "FR":
    X = X                                  # Select all features
else:
    raise ValueError("info must be 'atom', 'FG', or 'FR'")

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_labels, test_size=0.2, random_state=42) # Split the data into training and testing sets

# Scale inputs
scaler_X = MinMaxScaler()  # fitting over X_train to avoid data leakage
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scale outputs 
scaler_Y = MinMaxScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

# ------------ Model Definition, Training, and Evaluation ---------------
if Load_model and os.path.exists(model_path):
    print(f"------- [INFO] Loading pre-trained model from {model_path} -------")
    net = tf.keras.models.load_model(model_path)

    if os.path.exists(history_path):        # Load training history
        with open(history_path, "r") as f:  # "r" = reading mode
            history_dict = json.load(f)
            MSE_training_history, MSE_val_history = extract_history(history_dict) # "extract_history" from Utils.py --> extract training and validation MSE from history dictionary
else:
    print("------- [INFO] Training model from scratch -------")
    # Define network
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
    for units in hidden_layers:
        net.add(tf.keras.layers.Dense(units, activation=activation_function))
    net.add(tf.keras.layers.Dense(Y_labels.shape[1], activation="linear"))

    # Compile
    net.compile(optimizer='adam', loss='mse', metrics=['mse'])

    # Early stopping callback
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',           # monitor validation loss
        patience = patience,            # stop if no improvement after patience % of total epochs
        restore_best_weights = True     # restore weights from best epoch
    )

    # Train
    start_time = time.time()                 # start time measurement
    history = net.fit(X_train_scaled, Y_train_scaled, validation_split = 0.2, epochs=epochs, batch_size=32, verbose=1, callbacks=[early_stop])
    training_time = time.time() - start_time # end time measurement

    # Extract and save training history to JSON
    MSE_training_history, MSE_val_history = extract_history(history.history)
    with open(history_path, "w") as f:
        json.dump(history.history, f) # save history as JSON

    # Save model after training
    print("------- [INFO] Trained model saved -------")
    net.save(model_path)

# ---------------- Stop Memory Monitoring ----------------
stop_flag = True
monitor_thread.join()

# Save memory log to CSV
mem_df = pd.DataFrame(mem_log, columns=["timestamp", "memory_MB"])
mem_df["time_s"] = mem_df["timestamp"] - mem_df["timestamp"].iloc[0]
mem_csv_path = os.path.join(model_dir, "memory_usage.csv")
mem_df.to_csv(mem_csv_path, index=False)

# Plot memory usage curve
plt.figure(figsize=(8,5))
plt.plot(mem_df["time_s"], mem_df["memory_MB"], label="RAM used (MB)")
plt.xlabel("Time [s]")
plt.ylabel("Memory [MB]")
plt.title("Memory usage during training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "memory_usage.png"))
plt.close()

# ---------------- Integral of memory usage ----------------
# Compute the area under the RAM usage curve (MB*s)
memory_integral = np.trapz(mem_df["memory_MB"], mem_df["time_s"]) # integral by using trapz function
print(f"Integral of RAM usage curve: {memory_integral:.2f} MB*s")

# Save result to a text file
integral_path = os.path.join(model_dir, "memory_integral.txt")
with open(integral_path, "w") as f:
    f.write(f"Integral of RAM usage curve: {memory_integral:.2f} MB*s\n")


# ---------------- Evaluation --------------------------
Y_pred_scaled = net.predict(X_test_scaled)
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

r2_scores = [r2_score(Y_test[:, i], Y_pred[:, i]) for i in range(Y_test.shape[1])] # Y_test/pred[:, i] corresponds to each property, Y_test.shape[1]=12 (tot num of properties)

# Property names + u.m.
property_names = ['μ (D)', 'α (a₀³)', 'ε_HOMO (Ha)', 'ε_LUMO (Ha)', 'ε_gap (Ha)', '⟨r²⟩ (a₀²)', 'zpve (Ha)', 'U₀ (Ha)', 'U (Ha)', 'H (Ha)', 'G (Ha)', 'Cᵥ (cal/mol·K)']

# ------------- Plot Section -------------
plots_base_dir = "Plots"
plots_dir = os.path.join(plots_base_dir, "NN", info, activation_function, subfolder)
os.makedirs(plots_dir, exist_ok=True)

plot_training_curves(MSE_training_history, MSE_val_history, plots_dir) # Plot training curves
plot_parity_plots(Y_test, Y_pred, r2_scores, property_names, plots_dir) # Parity plot 4x3 grid for all the molecules

# ---------------- Save experiment info and metrics -------------------------
save_experiment_info(Results_dir, info, activation_function, hidden_layers, epochs, patience, X_train, X_test, property_names, r2_scores, Y_test, Y_pred, Y_test_scaled, Y_pred_scaled, net, training_time)
