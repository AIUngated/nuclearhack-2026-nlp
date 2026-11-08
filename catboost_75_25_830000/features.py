"""Deterministic features: only molecular SMILES, no assay inputs at inference."""
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
import core

ROOT=Path(__file__).resolve().parent
NAMES=tuple(json.loads((ROOT/'reference/config.json').read_text())['descriptor_names'])


@lru_cache(maxsize=20000)
def row(smiles,kind='count3'):
    rec=core.canonical_record(smiles)
    if rec is None: raise ValueError('Invalid SMILES')
    mol=Chem.MolFromSmiles(rec['parent'])
    if kind=='achiral3': Chem.RemoveStereochemistry(mol)
    elif kind=='tautomer3': mol=rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    elif kind not in ['count3','binary3','density3']: raise ValueError(kind)
    parent=Chem.MolToSmiles(mol,isomericSmiles=True)
    desc=core.feature_row(parent,False,NAMES)[2048:]
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=4096,includeChirality=True)
    fp=np.zeros(4096,dtype=np.float32)
    for k,v in generator.GetCountFingerprint(mol).GetNonzeroElements().items(): fp[k]=v
    if kind=='binary3': fp=(fp>0).astype(np.float32)
    elif kind=='density3': fp=fp/max(1,mol.GetNumHeavyAtoms())
    return np.r_[fp,desc]


def matrix(smiles,kind='count3'):return np.stack([row(s,kind) for s in smiles])
