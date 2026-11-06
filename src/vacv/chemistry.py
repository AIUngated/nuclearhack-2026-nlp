from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "HBD", "HBA", "RotatableBonds", "RingCount",
    "FractionCSP3", "HeavyAtomCount", "NHOHCount", "NOCount", "AromaticRings",
    "AliphaticRings", "FormalCharge", "BertzCT", "MolMR",
]


@lru_cache(maxsize=50000)
def molecular_record(smiles: str) -> dict:
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string.")
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {smiles[:120]}")
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    parent = rdMolStandardize.FragmentParent(mol)
    parent = rdMolStandardize.Uncharger().uncharge(parent)
    Chem.SanitizeMol(parent)
    parent_smiles = Chem.MolToSmiles(parent, isomericSmiles=True)
    achiral = Chem.Mol(parent)
    Chem.RemoveStereochemistry(achiral)
    Chem.SanitizeMol(achiral)
    group = Chem.MolToSmiles(achiral, isomericSmiles=False)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=achiral, includeChirality=False)
    # Keep all acyclic compounds together: conservative; never invent row-specific scaffolds.
    return {"canonical_smiles": canonical, "parent_smiles": parent_smiles,
            "compound_group": group, "scaffold": scaffold or "__ACYCLIC__"}


def molecular_metadata(smiles: list[str]) -> pd.DataFrame:
    records = []
    for row, value in enumerate(smiles):
        try:
            record = molecular_record(value)
        except (ValueError, RuntimeError) as error:
            raise ValueError(f"Molecule in row {row} cannot be processed: {error}") from error
        records.append(dict(record))
    return pd.DataFrame(records)


def descriptors(mol) -> np.ndarray:
    return np.asarray([
        Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol), Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcFractionCSP3(mol),
        mol.GetNumHeavyAtoms(), Lipinski.NHOHCount(mol), Lipinski.NOCount(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol), rdMolDescriptors.CalcNumAliphaticRings(mol),
        Chem.GetFormalCharge(mol), Descriptors.BertzCT(mol), Crippen.MolMR(mol),
    ], dtype=np.float32)


def make_features(smiles: list[str], config: dict) -> tuple[np.ndarray, pd.DataFrame]:
    metadata = molecular_metadata(smiles)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=config["radius"], fpSize=config["n_bits"],
        includeChirality=config.get("use_chirality", True))
    values = []
    for parent in metadata["parent_smiles"]:
        mol = Chem.MolFromSmiles(parent)
        bits = np.zeros(config["n_bits"], dtype=np.float32)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), bits)
        values.append(np.concatenate([bits, descriptors(mol)]))
    return np.stack(values), metadata


def tanimoto_matrix(x: np.ndarray, y: np.ndarray, n_bits: int) -> np.ndarray:
    a, b = np.asarray(x[:, :n_bits], dtype=np.float64), np.asarray(y[:, :n_bits], dtype=np.float64)
    intersection = a @ b.T
    union = a.sum(axis=1)[:, None] + b.sum(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def max_train_similarity(x: np.ndarray, reference: np.ndarray, n_bits: int) -> np.ndarray:
    # Chunk queries to avoid allocating an unbounded query x training matrix in the app.
    return np.concatenate([tanimoto_matrix(x[i:i + 256], reference, n_bits).max(axis=1)
                           for i in range(0, len(x), 256)])
