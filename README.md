# DSCE_Bernardi_Liberatore

## Descritpion
Molecular Properties Prediction - Comparative ML Modeling

Project Goal: To establish high-accuracy predictive models for molecular properties by applying the Benson-Buss theory: number of atoms and functional groups are the features used to learn to predcit thermodynamic properties of a set 20k molecules taken from the qm9 dataset (https://figshare.com/articles/dataset/Readme_file_Data_description_for_Quantum_chemistry_structures_and_properties_of_134_kilo_molecules_/1057641?backTo=%2Fcollections%2F_%2F978904&file=3195392). After pre-processing the dataset, the project focuses on applying and comparing various supervised Machine Learning algorithms and Neural Networks, thereby accelerating the simulation and analysis process in chemical engineering.

### Contribution and Collaboration Statement
This project was developed as part of a Master’s level coursework (Data Science in Chemical Engineering) in collaboration with a colleague. This repository is maintained by Nicola Bernardi and serves as a portfolio to specifically showcase my **Machine Learning evaluation and analytical expertise**.

The data pre-processing and the initial setup of the Neural Network architecture were handled by my collaborator.

**My explicit technical contributions were focused on:**

1.  **Random Forest Modeling:** Full implementation, training, and optimization of the **Random Forest** model using **scikit-learn**.
2.  **In-Depth Comparative Analysis:** Execution of an **extensive performance evaluation** across all models, including the Random Forest and the collaborative Neural Network. My work focused on **statistical analysis, rigorous cross-validation**, and **interpretation of results** to determine the optimal model for error minimization.
3.  **Core Modeling & Evaluation Files:** This repository contains the code defining the Random Forest architecture and all the scripts used for the comprehensive evaluation methodology.

---

### Key Results & Technologies

| Result | Achievement |
| :--- | :--- |
| **Model Performance** | Achieved a **[R-squared values in the range of 0.79-0.99]** on the holdout test set, significantly outperforming initial linear regression benchmarks. |
| **Analytical Insight** | Provided detailed **comparative insights** on the bias-variance trade-off between the ensemble (Random Forest) and deep learning (NN) methods, guiding the final model selection. |

**Technologies Used:**

* **Python (Core):** `scikit-learn`, `NumPy`, `Pandas`, `Matplotlib`
* **Evaluation Context:** `TensorFlow`, `Keras` (as models analyzed)

---


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
