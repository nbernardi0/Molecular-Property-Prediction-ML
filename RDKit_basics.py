from rdkit import Chem
from rdkit.Chem import FragmentCatalog, RDConfig, Draw
from collections import defaultdict
import os

# The following code is designed to test RDKit's tools 
# on a single molecule defined by its SMILES representation

# 1. Load RDKit functional groups definition file
fName = os.path.join(os.path.dirname(__file__), 'CustomFunctionalGroups.txt')

# 2. Create fragment parameters (fragments between 1 and 6 atoms)
fparams = FragmentCatalog.FragCatParams(1, 1, fName)
print(f"Number of functional groups in the catalog: {fparams.GetNumFuncGroups()}")

# 3. Initialize fragment catalog and generator
fcat = FragmentCatalog.FragCatalog(fparams)
fcgen = FragmentCatalog.FragCatGenerator()

# 4. Define the molecule via SMILES
smiles = 'c1ccccc1'
mol = Chem.MolFromSmiles(smiles)

# 5. Count atoms C, H, O, N, F in the whole molecule
# Add explicit hydrogens
mol_with_H = Chem.AddHs(mol)

# Count atoms C, H, O, N, F
atom_counts = defaultdict(int)
for atom in mol_with_H.GetAtoms():
    symbol = atom.GetSymbol()
    if symbol in {'C', 'H', 'O', 'N', 'F'}:
        atom_counts[symbol] += 1

# Print result
print("\nAtom counts in the full molecule (with explicit H):")
for symbol in ['C', 'H', 'O', 'N', 'F']:
    print(f"  {symbol}: {atom_counts[symbol]}")

# 6. Add fragments to the catalog
num_entries = fcgen.AddFragsFromMol(mol, fcat)
print(f"\nNumber of fragments identified: {num_entries}")

# 7. Count and display functional groups with name, SMILES and SMARTS
print("\nFunctional groups present in the molecule:")
fg_counts = defaultdict(int)
seen_matches = set()

for fid in range(fparams.GetNumFuncGroups()):
    fg = fparams.GetFuncGroup(fid)
    name = fg.GetProp('_Name')  
    smarts = Chem.MolToSmarts(fg)
    smarts_mol = Chem.MolFromSmarts(smarts)

    matches = mol.GetSubstructMatches(smarts_mol)

    for match in matches:
        atom_set = frozenset(match)
        key = (fid, atom_set)
        if key not in seen_matches:
            seen_matches.add(key)
            fg_counts[fid] += 1

# Print found functional groups
for fid in sorted(fg_counts):
    count = fg_counts[fid]
    fg = fparams.GetFuncGroup(fid)
    name = fg.GetProp('_Name')
    smarts = Chem.MolToSmarts(fg)
    smiles_frag = Chem.MolToSmiles(fg)
    print(f"ID{fid + 1}: {count} occurrence(s) — Name: {name} — SMILES: {smiles_frag} — SMARTS: {smarts}")

# 8. Show the molecule
img = Draw.MolToImage(mol, size=(300, 300))
img.show()
