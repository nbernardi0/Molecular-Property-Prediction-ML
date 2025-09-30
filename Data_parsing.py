import os
import pandas as pd
from rdkit import Chem

# DISCLAIMER: The folder 'dsgdb9nsd.xyz' has not been included in the github repo because of its size. 
#             As a matter of fact it causes the  machine to slow down significantly.


# Path to the folder containing .xyz files
xyz_folder = 'dsgdb9nsd.xyz'

# List of property names with units (from the README)
property_names = ["tag", "index", "A [GHz]", "B [GHz]","C [GHz]","mu [Debye]","alpha [Bohr^3]","homo [Hartree]","lumo [Hartree]","gap [Hartree]","r2 [Bohr^2]","zpve [Hartree]","U0 [Hartree]","U [Hartree]","H [Hartree]","G [Hartree]","Cv [cal/(mol K)]"]

# List of dictionaries to be converted into a DataFrame
data = []

# Loop through the .xyz files
for idx, filename in enumerate(os.listdir(xyz_folder)): # os.listdir() generates a list of all files names in the directory, enumerate adds an index
    file_path = os.path.join(xyz_folder, filename) # to join the specific molecule file 
 
    with open(file_path, 'r') as f:  # to open the file in reading mode
        lines = f.readlines()        # to read all lines in the file and put in lines list (lines[0], lines[1], ...)
    # Line 0: number of atoms
    num_atoms = int(lines[0].strip())  # 1st line: number of atoms (.strip() to remove whitespace)
    # Line 1: list of 17 properties
    props = lines[1].strip().split()   # .split() to separate properties
    
    # Line containing SMILES strings (GDB9 and optimized); we take the first
    smiles_line = lines[2 + num_atoms + 1].strip()  
    smiles_gdb9 = smiles_line.split()[0]   # .split() to separate  [0] to take the first SMILES string
    mol = Chem.MolFromSmiles(smiles_gdb9)  # Convert SMILES to RDKit molecule
    smiles_RDKit = Chem.MolToSmiles(mol)   # Convert back to univocal SMILES (gbd9 may have different valid SMILES for the same molecule)
    # Collect relevant properties, excluding A, B, C
    selected_props = {"smiles": smiles_RDKit,} 
    for i, key in enumerate(property_names):
        if i in (2, 3, 4):  # Skip rotational constants A, B, C
            continue
        try:
            selected_props[key] = float(props[i]) # We are adding a new key to the dictionary selected_props corresponding to the property name (ex. "mu [Debye]") and assigning it the value converted to float
        except ValueError:
            selected_props[key] = props[i]  # Keep as string if not convertible (to avoid errors)
    data.append(selected_props)

# From a list to a DataFrame
df = pd.DataFrame(data)

# Save as a Pickle
df.to_pickle("qm9_preprocessed.pkl")
print("Extraction complete. Data saved to qm9_preprocessed.pkl.")
