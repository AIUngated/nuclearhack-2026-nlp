#!/usr/bin/env python3
"""Check grouping, source-label qualification, ensemble arithmetic and reloads."""
from pathlib import Path
import argparse,json,subprocess,sys,time
import numpy as np
import pandas as pd
import joblib
import pipeline as p

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,default=ROOT/'validation');parser.add_argument('--work',type=Path,default=Path('.verification'))
    args=parser.parse_args();args.work.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'models/config.json').read_text());d,r,e=p.load_data()
    raw=pd.read_csv(ROOT/'data/train_v2.csv');keep=(raw.CC50_uM>0)&(raw.IC50_uM_VACV>0)&np.isfinite(raw.CC50_uM)&np.isfinite(raw.IC50_uM_VACV)
    assert len(r)==237 and len(d)==230 and r.raw_si.max()>36000
    assert np.array_equal(raw.index[keep],r.source_row)
    np.testing.assert_allclose(r.raw_si,raw.loc[keep,'CC50_uM']/raw.loc[keep,'IC50_uM_VACV'],rtol=1e-14)
    test=pd.read_csv(ROOT/'data/test.csv');assert list(test)==['index','SMILES']
    blocked={p.identity(s) for s in raw.SMILES.tolist()+test.SMILES.tolist()}
    assert not set(e.verified_identity)&blocked
    measurements=pd.read_csv(ROOT/'data/external/selectivity_measurements.csv')
    kind=measurements.original_type.fillna('').str.lower().str.replace(r'\s+','',regex=True)
    ok=measurements.virus.eq('VACV')&(kind.isin(['ratiocc50/ic50','cc50/ic50'])|measurements.derivation.eq('paired_CC50_ic50_same_document_cell_duration'))
    expected=10.**measurements.loc[ok].groupby('canonical').log_value.median()
    np.testing.assert_allclose(e.canonical.map(expected),e.strict_si,equal_nan=True,rtol=1e-14)
    group_checks=0;external_scaffold_checks=0
    for split in p.SPLITS:
        f=np.load(args.cache/f'folds_{split}.npy');groups=d.scaffold if split.startswith('scaffold') else d.compound
        for fold in range(5):
            assert not set(groups[f==fold])&set(groups[f!=fold]);group_checks+=1
            if split.startswith('scaffold'):
                held=set(d.scaffold[f==fold]);ids=np.isfinite(e.strict_si)&~e.scaffold_core.isin(held)
                assert not set(e.loc[ids,'scaffold_core'])&held;external_scaffold_checks+=1
    table=pd.read_csv(ROOT/'reports/validation.csv');checks=0
    def oof(name,split,memo):
        if name in memo:return memo[name]
        if name in config['components']:value=np.load(ROOT/f'reports/cases/{name}/{split}.npy')
        else:
            formula=config['formulas'][name];values={n:oof(n,split,memo) for n in formula['members']};value=p.evaluate_formula(formula,values)
        memo[name]=value;return value
    for _,row in table.iterrows():
        pred=oof(row['name'],row['split'],{});y=r.raw_si.to_numpy();error=(y-pred[r.group_id.to_numpy(int)])**2
        for key,mask in [('mse',np.ones(len(y),bool)),('mse_low',y<10000),('mse_ordinary',y<1000),('mse_tail',(y>=1000)&(y<10000))]:
            assert np.isclose(error[mask].mean(),row[key],rtol=1e-12);checks+=1
    template=pd.read_csv(ROOT/'data/sample_submit.csv');files=[]
    for file in ROOT.glob('submission*.csv'):
        a=pd.read_csv(file);assert list(a)==['index','SI VACV'] and len(a)==597 and a['index'].equals(template['index'])
        assert np.isfinite(a['SI VACV']).all() and a['SI VACV'].ge(.01).all();files.append(file.name)
    baseline=pd.read_csv(ROOT/f'submission_{p.REF}.csv');old=pd.read_csv(ROOT/'reference/submission_833000.csv')
    reference_delta=float(abs(baseline['SI VACV']-old['SI VACV']).max());assert reference_delta<1e-8
    names=list(dict.fromkeys([config['selected'],p.REF]+[n for n in ['seed_mean','binary3','ordered'] if n in config['formulas']]))
    reloads=[]
    for name in names:
        out=args.work/f'{name}.csv';start=time.perf_counter()
        subprocess.run([sys.executable,str(ROOT/'pipeline.py'),'predict','--model',name,'--template',str(ROOT/'data/sample_submit.csv'),'--output',str(out)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        a=pd.read_csv(out);b=pd.read_csv(ROOT/f'submission_{name}.csv');delta=float(abs(a['SI VACV']-b['SI VACV']).max());assert delta<1e-8
        reloads.append(dict(model=name,fresh_process=True,max_delta=delta,seconds=time.perf_counter()-start))
    native=[]
    for name in config['components']:
        folder=ROOT/'models'/name;info=json.loads((folder/'config.json').read_text());spec=info['spec']
        model=p.CatBoostRegressor();model.load_model(str(folder/'model.cbm'));params=model.get_all_params()
        pp=joblib.load(folder/'preprocessor.joblib');assert len(model.feature_names_)==int(pp.mask_.sum())
        if spec['kind']=='multi':assert params['loss_function']=='MultiRMSE'
        else:assert params['loss_function']==f"Tweedie:variance_power={spec['power']}"
        for key,value in spec.get('params',{}).items():
            if key not in ['iterations']:assert params[key]==value
        native.append(dict(name=name,loss=params['loss_function'],boosting_type=params.get('boosting_type'),random_seed=params['random_seed'],features=len(model.feature_names_)))
    poison=test.copy();poison['CC50_uM']=1e25;poison['IC50_uM_VACV']=1e-25;poison['SI VACV']=-1e30;poison['Culture']='unused'
    input=args.work/'poison.csv';poison.to_csv(input,index=False);out=args.work/'poison_predictions.csv'
    subprocess.run([sys.executable,str(ROOT/'pipeline.py'),'predict','--test',str(input),'--template',str(ROOT/'data/sample_submit.csv'),'--output',str(out)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    delta=float(abs(pd.read_csv(out)['SI VACV']-pd.read_csv(ROOT/'submission.csv')['SI VACV']).max());assert delta<1e-8
    info=dict(status='passed',unchanged_validation_measurements=237,group_checks=group_checks,external_identity_overlaps=0,
      strict_ic50_external_structures=int(e.strict_si.notna().sum()),external_scaffold_checks=external_scaffold_checks,
      independent_metric_checks=checks,reference_833000_delta=reference_delta,csvs_checked=files,reloads=reloads,
      native_model_checks=native,forbidden_columns_prediction_delta=delta,test_input_columns=list(test),new_kaggle_scores=None)
    p.core.json_write(ROOT/'reports/verification.json',info);print(json.dumps(info,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
