#!/usr/bin/env python3
"""Shared SMILES, feature and CSV helpers retained from the original project.
The current command-line entry point is pipeline.py; deployment is fixed to 75/25.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
import argparse,json,hashlib,time,importlib.metadata
import numpy as np
import pandas as pd
import joblib
from tqdm.auto import tqdm
from rdkit import Chem,DataStructs
from rdkit.Chem import Descriptors,Crippen,Lipinski,rdMolDescriptors,rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
ROOT=Path(__file__).resolve().parent
TARGET='SI VACV';SMILES='SMILES';ID='index';EPS=1e-8
DESCRIPTOR_NAMES=tuple(name for name,_ in Descriptors._descList)
@dataclass
class Config:
    folds:int=5
    dedup:str='median_log'
    clip_min:float=.01
    clip_max:float=10000.

def json_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@lru_cache(maxsize=20000)
def canonical_record(smiles):
    if not isinstance(smiles,str) or not smiles.strip():return None
    mol=Chem.MolFromSmiles(smiles.strip())
    if mol is None or mol.GetNumHeavyAtoms()==0:return None
    canonical=Chem.MolToSmiles(mol,isomericSmiles=True)
    parent=rdMolStandardize.Uncharger().uncharge(rdMolStandardize.FragmentParent(mol))
    Chem.SanitizeMol(parent)
    achiral=Chem.Mol(parent);Chem.RemoveStereochemistry(achiral)
    Chem.GetSymmSSSR(achiral)
    return dict(canonical=canonical,parent=Chem.MolToSmiles(parent,isomericSmiles=True),
                compound=Chem.MolToSmiles(achiral,isomericSmiles=False),
                scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=achiral,includeChirality=False) or '__ACYCLIC__')

def prepare_train(path,cfg,output):
    raw=pd.read_csv(path)
    required=[SMILES,'CC50_uM','IC50_uM_VACV']
    if not set(required)<=set(raw):raise ValueError('Train must contain '+str(required))
    raw=raw.copy();raw['source_row']=np.arange(len(raw))
    cc=pd.to_numeric(raw.CC50_uM,errors='coerce');ic=pd.to_numeric(raw.IC50_uM_VACV,errors='coerce')
    keep=np.isfinite(cc)&np.isfinite(ic)&(cc>0)&(ic>0)
    excluded=raw.loc[~keep].copy();excluded['exclusion_reason']='missing_nonfinite_or_nonpositive_measurement'
    records=raw.loc[keep].copy();records['raw_si']=(cc[keep]/ic[keep]).to_numpy()
    valid=np.isfinite(records.raw_si)&(records.raw_si>0)
    extra=records.loc[~valid].copy();extra['exclusion_reason']='invalid_ratio';excluded=pd.concat([excluded,extra],ignore_index=True)
    records=records.loc[valid].copy()
    molecular=[canonical_record(s) for s in tqdm(records.SMILES,desc='Canonicalizing train')]
    bad=np.array([m is None for m in molecular]);extra=records.loc[bad].copy();extra['exclusion_reason']='invalid_smiles'
    excluded=pd.concat([excluded,extra],ignore_index=True)
    records=records.loc[~bad].copy().reset_index(drop=True)
    meta=pd.DataFrame([m for m in molecular if m is not None]);records=pd.concat([records,meta],axis=1)
    records['log_si']=np.log10(records.raw_si+EPS)
    grouped=[]
    for canonical,part in records.groupby('canonical',sort=True):
        log_target=float(part.log_si.median()) if cfg.dedup=='median_log' else float(np.log10(part.raw_si.median()+EPS))
        grouped.append(dict(canonical=canonical,parent=part.parent.iloc[0],compound=part.compound.iloc[0],scaffold=part.scaffold.iloc[0],
                            log_si=log_target,aggregate_si=float(10**log_target-EPS),measurement_count=len(part),
                            source_rows=';'.join(map(str,part.source_row.tolist()))))
    data=pd.DataFrame(grouped)
    if len(data)<max(20,cfg.folds*3):raise ValueError('Too few valid unique molecules for the requested CV')
    index={s:i for i,s in enumerate(data.canonical)};records['group_id']=records.canonical.map(index).astype(int)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    records.to_csv(out/'training_measurements.csv',index=False);data.to_csv(out/'training_deduplicated.csv',index=False)
    excluded.to_csv(out/'excluded_training_rows.csv',index=False)
    lower_bound=float(np.mean(np.square(records.raw_si-np.clip(records.raw_si,cfg.clip_min,cfg.clip_max))))
    audit=dict(source_rows=len(raw),complete_valid_measurements=len(records),excluded_rows=len(excluded),unique_molecules=len(data),
               duplicate_measurements=len(records)-len(data),invalid_smiles=int(bad.sum()),dedup=cfg.dedup,
               target_definition='log10(CC50_uM/IC50_uM_VACV + 1e-8)',maximum_si=float(records.raw_si.max()),
               minimum_possible_MSE_from_prediction_clipping_on_measurements=lower_bound,source_sha256=digest(path))
    json_write(out/'data_audit.json',audit);return data,records,audit

def read_test(path):
    # Do not load test concentrations or even their missingness.
    frame=pd.read_csv(path,usecols=[ID,SMILES])
    if frame.empty or frame[ID].isna().any() or frame[ID].duplicated().any():raise ValueError('Invalid or repeated test identifiers')
    values=[canonical_record(s) for s in tqdm(frame.SMILES,desc='Canonicalizing test')]
    invalid=[i for i,r in enumerate(values) if r is None]
    if invalid:raise ValueError('Invalid test SMILES at rows '+str(invalid[:20])+'; rows are not silently dropped')
    frame=frame.copy();frame['canonical']=[r['canonical'] for r in values];return frame

def write_submission(path,test,predictions,template_path=None):
    p=np.asarray(predictions,float).reshape(-1)
    if len(p)!=len(test) or not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid predictions')
    ids=test[ID].to_numpy()
    if template_path:
        template=pd.read_csv(template_path)
        if list(template.columns)!=[ID,TARGET] or template[ID].isna().any() or template[ID].duplicated().any():raise ValueError('Invalid submission template')
        if len(template)!=len(test) or set(template[ID])!=set(ids):raise ValueError('Template and test IDs differ')
        lookup=pd.Series(p,index=ids);result=pd.DataFrame({ID:template[ID],TARGET:lookup.loc[template[ID]].to_numpy()})
    else:result=pd.DataFrame({ID:ids,TARGET:p})
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);result.to_csv(path,index=False);return result

def legacy_descriptors(mol):
    return np.asarray([Descriptors.MolWt(mol),Crippen.MolLogP(mol),rdMolDescriptors.CalcTPSA(mol),Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),Lipinski.NumRotatableBonds(mol),rdMolDescriptors.CalcNumRings(mol),rdMolDescriptors.CalcFractionCSP3(mol),
        mol.GetNumHeavyAtoms(),Lipinski.NHOHCount(mol),Lipinski.NOCount(mol),rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),Chem.GetFormalCharge(mol),Descriptors.BertzCT(mol),Crippen.MolMR(mol)],dtype=np.float32)

@lru_cache(maxsize=20000)
def feature_row(smiles,legacy=False,names=DESCRIPTOR_NAMES):
    rec=canonical_record(smiles)
    if rec is None:raise ValueError('Invalid SMILES')
    mol=Chem.MolFromSmiles(rec['parent'] if legacy else rec['canonical'])
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    bits=np.zeros(2048,np.float32);DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol),bits)
    if legacy:desc=legacy_descriptors(mol)
    else:
        values=[]
        for name in names:
            try:v=float(getattr(Descriptors,name)(mol))
            except (ValueError,RuntimeError,OverflowError,ZeroDivisionError):v=np.nan
            values.append(v if np.isfinite(v) else np.nan)
        with np.errstate(over='ignore',invalid='ignore'):desc=np.asarray(values,np.float32)
        desc[~np.isfinite(desc)]=np.nan
    return np.r_[bits,desc]

def features(smiles,legacy=False,names=DESCRIPTOR_NAMES,progress=False):
    rows=tqdm(smiles,desc='Morgan + descriptors',leave=False) if progress else smiles
    return np.stack([feature_row(s,legacy,tuple(names)) for s in rows])

