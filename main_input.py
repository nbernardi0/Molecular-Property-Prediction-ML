from rdkit import Chem
from rdkit.Chem import FragmentCatalog
from rdkit import RDConfig
from collections import defaultdict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utils import build_fragment_catalog, atom_count_from_smiles, count_functional_groups,  binary_fingerprint_from_smiles, save_plot
from Plots import plot_frequency, plot_functional_groups, plot_correlation_heatmap

# Load dataset 
df = pd.read_pickle("qm9_preprocessed.pkl")
df = df.head(20000)                          # select only 20k molecules for faster testing 

smiles_list = df['smiles'].tolist()          # to transform a dataframe column in a python list

# Build fragment catalog 
fcat, fparams = build_fragment_catalog(smiles_list)  # generation of the fragment catalog 
num_fr = fcat.GetNumEntries()                        # to obtain the number of fragments in the catalog
print(f"\nNumber of fragments in catalog: {num_fr}")

# Input Matrix Construction
features = []

for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)                              # Convert SMILES to RDKit Mol objects
    atom_counts = atom_count_from_smiles(mol)                  # Count atoms (C, H, O, N, F)
    func_group_counts = count_functional_groups(mol, fparams)  # Count functional groups
    fragment_fp = binary_fingerprint_from_smiles(smi, fcat)    # Generate binary fragment fingerprint

    input_vector = atom_counts + func_group_counts + fragment_fp # Concatenate all feature types into a single input vector
    features.append(input_vector)                                # Append to feature list

X = np.array(features) # Convert feature matrix to numpy array

# Labels Matrix Construction
target_cols = [col for col in df.columns if col not in  ['smiles', 'tag', 'index']] # considering all columns except 'smiles', 'tag', 'index' 
Y_labels = df[target_cols].to_numpy(dtype=np.float32)                               # Convert to numpy array

# Save features and labels
np.save("X_features.npy", X)
np.save("Y_labels.npy", Y_labels)

# ------------- Frequency analysis and filtering ------------------------------

num_atoms = 5                                  # C, H, O, N, F
num_func_groups = fparams.GetNumFuncGroups()   # total number of functional groups in the catalog
num_fragments = fcat.GetNumEntries()           # total number of fragments in the catalog

# Slice functional groups and fragments
X_func_groups = X[:, num_atoms : num_atoms + num_func_groups] # (num_molecules, num_functional_groups)
X_fragments = X[:, num_atoms + num_func_groups:]              # (num_molecules, num_fragments)

# Absolute frequency of appearance of each functional group and fragment
freq_func_groups = X_func_groups.sum(axis=0)  # (num_functional_groups,) <-- axis=0 to sum over rows (molecules)
freq_fragments = X_fragments.sum(axis=0)      # (num_fragments,)

threshold_FG = 50;   # treshold for filtering functional groups
threshold_FR = 100;  # treshold for filtering fragments

# Filter functional groups index and fragments based on frequency threshold
keep_func_ids = [i for i, f in enumerate(freq_func_groups) if f > threshold_FG]      # enumerate() to get both index (i) and element (f)
keep_frag_ids = [i for i, f in enumerate(freq_fragments) if f > threshold_FR]

# Count before/after filtering
print(f"Functional groups: initial = {num_func_groups}, after filtering = {len(keep_func_ids)}")
print(f"Fragments: initial = {num_fragments}, after filtering = {len(keep_frag_ids)}")

# X_filtered construction
X_filtered = np.concatenate([
    X[:, :num_atoms],                   # Atoms
    X_func_groups[:, keep_func_ids],    # Functional groups filtered
    X_fragments[:, keep_frag_ids]       # Fragments filtered
], axis=1)                              # to concatenate along columns

print(f"Original X shape: {X.shape}")
print(f"Filtered X shape: {X_filtered.shape}")

# Save the filtered version
np.save("X_features_filtered.npy", X_filtered)

# ------------- Plots ---------------------
dataset_visualization_dir = os.path.join("Dataset_visualization") # Directory to save plots
os.makedirs(dataset_visualization_dir, exist_ok=True)

# Plot before/after filtering
FG_labels = [f"#{i+1}" for i in range(num_func_groups)] # Labels for all functional groups
FG_labels_filtered = [f"#{i+1}" for i in keep_func_ids] # Labels for functional groups that remain after filtering

# Unfiltered plots
plot_frequency(freq_func_groups, "Functional Group", "FG bar plot (not filtered)", threshold_FG, save_dir = dataset_visualization_dir, x_labels=FG_labels)   # Plot functional groups before filtering
plot_frequency(freq_fragments, "Fragment", "Fragment bar plot (not filtered)", threshold_FR, save_dir = dataset_visualization_dir, x_labels=None)            # Plot fragments before filtering
# Filtered plots
plot_frequency(X_func_groups[:, keep_func_ids].sum(axis=0), "Functional Group (filtered)", "FG bar plot (filtered)", threshold_FG, save_dir = dataset_visualization_dir, x_labels = FG_labels_filtered)   # Plot functional groups after filtering
plot_frequency(X_fragments[:, keep_frag_ids].sum(axis=0), "Fragment (filtered)", "Fragment bar plot (filtered)", threshold_FR, save_dir = dataset_visualization_dir, x_labels = None)                     # Plot fragments after filtering

# Plot all functional groups
plot_functional_groups(fparams,list(range(fparams.GetNumFuncGroups())), save_path=os.path.join(dataset_visualization_dir, "functional_groups_all"))

# Plot only filtered functional groups, sorted by frequency
plot_functional_groups(fparams, keep_func_ids, freq_func_groups=freq_func_groups, sort_by_freq=True, save_path=os.path.join(dataset_visualization_dir, "functional_groups_filtered"))

plot_correlation_heatmap(Y_labels, target_cols, save_dir = dataset_visualization_dir) # Plot correlation heatmap of the target properties

