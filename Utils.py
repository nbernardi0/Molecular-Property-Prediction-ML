import pandas as pd
from rdkit import Chem
from rdkit.Chem import FragmentCatalog
from rdkit import RDConfig
from collections import defaultdict
import os
import pandas as pd
import numpy as np

def build_fragment_catalog(smiles_list):
    ms = [Chem.MolFromSmiles(smi) for smi in smiles_list] # Convert SMILES to RDKit Mol objects
    # Create fragment catalog parameters and objects
    fg_file_path = os.path.join(os.path.dirname(__file__), 'CustomFunctionalGroups.txt') # os.path.dirname(__file__) to get the directory of the current script
    fparams = FragmentCatalog.FragCatParams(1, 1, fg_file_path)                          # Create the parameters for a fragment catalog that uses the file fName as the definition of functional groups
    fcat = FragmentCatalog.FragCatalog(fparams)                                          # Create the fragment catalog object (empty at the beginning)
    fcgen = FragmentCatalog.FragCatGenerator()                                           # Create the fragment catalog generator object
    for m in ms:
        fcgen.AddFragsFromMol(m, fcat)  # Add fragments from a molecule to the catalog object

    return fcat, fparams

def atom_count_from_smiles(mol):
    mol_with_H = Chem.AddHs(mol)                  # the representation does not include hydrogens
    atom_counts = defaultdict(int)                # to initialize missing keys --> int by default is 0 --> atom_counts = {'C':0, 'H':0, 'O':0, 'N':0, 'F':0}
    for atom in mol_with_H.GetAtoms():            # iterate over all atoms in the molecule
        symbol = atom.GetSymbol()                 # get the atomic symbol (e.g., 'C', 'H', 'O', 'N', 'F')
        if symbol in {'C', 'H', 'O', 'N', 'F'}:
            atom_counts[symbol] += 1              # if atom type is one of the five, increment its count

    return [atom_counts['C'], atom_counts['H'], atom_counts['O'], atom_counts['N'], atom_counts['F']] # return a python list with the counts of each atom type in a fixed order

def count_functional_groups(mol, fparams):
    
    fg_counts = defaultdict(int)
    seen_matches = set()

    for fid in range(fparams.GetNumFuncGroups()): # cycle over all functional groups in the catalog (fid = functional group ID)
        fg = fparams.GetFuncGroup(fid)            # get the functional group corresponding to the current fid
        matches = mol.GetSubstructMatches(fg)     # Example: for mol = "CCO" (ethanol) and fg = "O", matches = ((2,),)
                                                  #          atom index 2 in the molecule corresponds to the oxygen
        for match in matches:
            atom_set = frozenset(match)           # convert match to immutable set
            key = (fid, atom_set)
            if key not in seen_matches:
                seen_matches.add(key)
                fg_counts[fid] += 1

    # Convert to ordered list (vector)
    return [fg_counts[i] for i in range(fparams.GetNumFuncGroups())]

def binary_fingerprint_from_smiles(smiles, fcat):
    mol = Chem.MolFromSmiles(smiles)                          # Convert SMILES string into an RDKit Mol object
    fpgen = FragmentCatalog.FragFPGenerator()                 # Initialize a fragment fingerprint generator
    fp = fpgen.GetFPForMol(mol, fcat)                         # Generate the fingerprint for the molecule using the fragment catalog as reference
    n_bits = fp.GetNumBits()                                  # Get the total number of bits in the fingerprint -> total number of fragments in the catalog
    return [1 if fp.GetBit(i) else 0 for i in range(n_bits)]  # Convert fingerprint into a binary list (1 = fragment present, 0 = absent)


# NN training functions

def extract_history(history_dict):
    mse_train = history_dict.get('loss', [])
    mse_val = history_dict.get('val_loss', [])
    return mse_train, mse_val


def save_plot(fig, filename_base, save_dir):
    """Save a matplotlib figure in both PNG and PDF format."""
    os.makedirs(save_dir, exist_ok=True) # to create directory if it doesn't exist
    fig.savefig(os.path.join(save_dir, filename_base + ".png"))
    fig.savefig(os.path.join(save_dir, filename_base + ".pdf"))

def save_experiment_info(plots_dir, info, activation_function, hidden_layers, epochs, patience, X_train, X_test, property_names, r2_scores,Y_test, Y_pred, Y_test_scaled, Y_pred_scaled,net, training_time):
    info_file = os.path.join(plots_dir, "experiment_info.txt")
    with open(info_file, "w", encoding="utf-8") as f:   
        f.write("Experiment Information\n")
        f.write("======================\n")
        f.write(f"Info type: {info}\n")
        f.write(f"Activation function: {activation_function}\n")
        f.write(f"Hidden layers: {hidden_layers}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Patience: {patience}\n")
        f.write(f"Training samples: {X_train.shape[0]}\n")
        f.write(f"Test samples: {X_test.shape[0]}\n")
        f.write(f"Total parameters: {net.count_params()}\n")
        f.write(f"Training time: {training_time:.2f} seconds\n\n")
        
        f.write("Performance Metrics\n")
        f.write("===================\n")
        for name, r2 in zip(property_names, r2_scores):
            f.write(f"{name}: R2 = {r2:.4f}\n")
        
        mse_test = np.mean((Y_test - Y_pred) ** 2)
        mse_test_scaled = np.mean((Y_test_scaled - Y_pred_scaled) ** 2) # with scaled data

        f.write(f"\nTest MSE (real scale, avg over properties): {mse_test:.6e}\n")
        f.write(f"Test MSE (normalized scale, avg over properties): {mse_test_scaled:.6e}\n")

    print(f"[INFO] Experiment details saved to {info_file}")

    # Print results in console
    print("\n------- Test Performance -------")
    print(f"Total parameters: {net.count_params()}")
    print(f"Training time: {training_time:.2f} seconds")
    for name, r2 in zip(property_names, r2_scores):
        print(f"{name}: R2 = {r2:.4f}")
    print(f"Test MSE (real scale): {mse_test:.6e}")
    print(f"Test MSE (normalized scale): {mse_test_scaled:.6e}")

