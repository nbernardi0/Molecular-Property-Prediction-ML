from rdkit import Chem
from rdkit.Chem import FragmentCatalog
import os

# 1. Define list of SMILES
smiles_list = ['CC(O)CO', 'c1ccccc1', 'CN(C)C=O']
ms = [Chem.MolFromSmiles(smi) for smi in smiles_list]

# 2. Create fragment parameters and initialize catalog
fg_file = os.path.join(os.path.dirname(__file__), 'CustomFunctionalGroups.txt')
fparams = FragmentCatalog.FragCatParams(1, 1, fg_file)
fcat = FragmentCatalog.FragCatalog(fparams)
fcgen = FragmentCatalog.FragCatGenerator()

# 3. Add fragments from all molecules to the catalog
for m in ms:
    nAdded = fcgen.AddFragsFromMol(m, fcat)

# 4. Check number of fragments
print(f"Number of fragments in the catalog: {fcat.GetNumEntries()}")  # Total number of unique fragments

# 5. Generate fingerprints
fpgen = FragmentCatalog.FragFPGenerator()
fp = fpgen.GetFPForMol(ms[0], fcat)
print(f"Bit vector for molecule 1 (SMILES = {smiles_list[0]}): {fp}")        # RDKit fingerprint bitvector object in molecule 1
print(f"Number of active fragment bits in molecule 1: {fp.GetNumOnBits()}")  # How many fragments matched in molecule 1

# 6. Print fingerprints for all molecules
for i, m in enumerate(ms):
    fp = fpgen.GetFPForMol(m, fcat) # Generate the fragment fingerprint (bit vector) for molecule m using the catalog fcat
    on_bits = list(fp.GetOnBits())  # Get the indices of bits that are set to 1 (fragments present in the molecule)
    print(f"Molecule {i+1}: {smiles_list[i]} Fragments present: {on_bits}")  # Print indices of fragments found
    



