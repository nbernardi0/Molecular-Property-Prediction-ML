# DSCE_Bernardi_Liberatore

## Code Files

- **ms_catalog.py**  
  Example of RDKit fragment catalog usage: build a catalog from a few test molecules, generate fingerprints, and print detected fragments.

- **RDKit_basics.py**  
  Introductory RDKit script: load functional groups, count atoms, identify groups in a single molecule (e.g., benzene), and visualize the molecule.

- **Data_parsing.py**  
  Parse `.xyz` files from QM9, extract thermodynamic properties and SMILES, and save the dataset as `qm9_preprocessed.pkl`.

- **main_input.py**  
  Build the input feature matrix (atom counts, functional groups, fragments), apply frequency filtering, save `X_features.npy` and `Y_labels.npy`, and generate dataset visualizations (frequency plots, correlation heatmap).

- **NN_training.py**  
  Train a single neural network (Keras/TensorFlow), optionally load a saved model, save training history, learning curves, parity plots, and R2 scores for each property.

- **NN_training_different_architectures.py**  
  Systematic study of different NN architectures (layers/neurons). Runs multiple seeds, computes mean/std metrics, and saves detailed result summaries.

- **RandomForest.py**  
  Random Forest regression with k-fold cross-validation. Computes R² per property, generates parity plots, and plots R2 vs training time.

- **Regression.py**  
  Linear, Ridge, or Lasso regression with k-fold cross-validation. Computes MSE and R² per property, produces parity plots and coefficient plots, and saves results.

- **Plots.py**  
  Collection of plotting utilities: frequency plots, functional group visualization, correlation heatmaps, training curves, parity plots, Random Forest plots, and regression coefficient plots.

- **Utils.py**  
  Helper functions: build fragment catalogs, count atoms and functional groups, generate binary fingerprints, handle training history, save plots, and save experiment results (including parameters, metrics, and performance).
