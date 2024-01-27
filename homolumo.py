from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, scf
from scipy.constants import physical_constants

# Constants
HARTREE_TO_EV_FACTOR = dict(physical_constants)['Hartree energy in eV'][0]

def find_homo_lumo(mf):
    lumo = float("inf")
    lumo_idx = None
    homo = -float("inf")
    homo_idx = None
    for i, (energy, occ) in enumerate(zip(mf.mo_energy, mf.mo_occ)):
        if occ > 0 and energy > homo:
            homo = energy
            homo_idx = i
        if occ == 0 and energy < lumo:
            lumo = energy
            lumo_idx = i

    return homo, lumo

def calculate_energy_gap(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        # Embed molecule only if it has conformers
        AllChem.EmbedMolecule(mol, maxAttempts=5000, useRandomCoords=True)
        Chem.MolToXYZFile(mol, "initial.xyz")
        mol = gto.M(atom="initial.xyz", basis="6-31g*")
        mol.build()
        mf = scf.RHF(mol).run()

        homo_hartree, lumo_hartree = find_homo_lumo(mf)

        homo_eV = homo_hartree * HARTREE_TO_EV_FACTOR
        lumo_eV = lumo_hartree * HARTREE_TO_EV_FACTOR

        energy_gap = abs(homo_eV - lumo_eV)

        return round(energy_gap, 6)

    except FileNotFoundError:
        print("XYZ file not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"Error occurred while processing XYZ file: {e}")
        return None
