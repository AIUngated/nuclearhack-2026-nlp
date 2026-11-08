#!/usr/bin/env python3
"""Fixed 75/25 CatBoost delivery; Kaggle MSE 830000 reported by the user."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,time,hashlib,traceback,multiprocessing,shutil
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import core,features as fe
from preprocess import FoldPreprocessor

ROOT=Path(__file__).resolve().parent
REF='reference_833000'
PRIMARY='canonical2042'
SPLITS=[PRIMARY,'scaffold2042','canonical42','scaffold42']
BASE=dict(iterations=1000,learning_rate=.03,depth=5,l2_leaf_reg=20,random_seed=42,
          thread_count=2,verbose=False,allow_writing_files=False)
SEEDS=[REF,'seed202','seed2042']


def positive(a):
    a=np.asarray(a,float)
    if not np.isfinite(a).all():raise ValueError('Non-finite prediction')
    return np.maximum(a,.01)


def specs():
    def c(name,**kw):
        s=dict(name=name,kind='scalar',feature='count3',cap=20000.,power=1.35,category='model');s.update(kw);return s
    return [c(REF),c('seed202',params=dict(random_seed=202),category='seed'),c('seed2042',params=dict(random_seed=2042),category='seed'),
      c('ordered',params=dict(boosting_type='Ordered')),
      c('depthwise',params=dict(grow_policy='Depthwise',min_data_in_leaf=8,depth=6)),
      c('lossguide',params=dict(grow_policy='Lossguide',max_leaves=16,min_data_in_leaf=8,depth=6)),
      c('l2_50',params=dict(l2_leaf_reg=50)),c('iterations600',params=dict(iterations=600)),
      c('iterations1500',params=dict(iterations=1500,learning_rate=.02)),
      c('tweedie125',power=1.25),c('tweedie150',power=1.5),
      c('binary3',feature='binary3',category='features'),c('descriptor_corr',correlation=.98,category='features'),
      c('density3',feature='density3',category='features'),
      c('multitask03',kind='multi',aux_weight=.3,category='multitask'),c('multitask10',kind='multi',aux_weight=1.,category='multitask'),
      c('external010',external_weight=.1,category='external'),
      c('cap15000',cap=15000.,category='target'),c('cap25000',cap=25000.,category='target'),
      c('smooth20000',smooth=True,category='target')]


def load_data():
    return pd.read_csv(ROOT/'data/train_unique.csv'),pd.read_csv(ROOT/'data/train_measurements.csv'),pd.read_csv(ROOT/'data/external/prepared_si.csv')


def identity(smiles):
    rec=core.canonical_record(smiles)
    if rec is None:raise ValueError('Invalid molecular identity')
    mol=Chem.MolFromSmiles(rec['parent']);Chem.RemoveStereochemistry(mol)
    mol=rdMolStandardize.TautomerEnumerator().Canonicalize(mol);Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol,isomericSmiles=False)


def score(pred,d,r):
    y=r.raw_si.to_numpy();p=np.asarray(pred)[r.group_id.to_numpy(int)];e=(y-p)**2
    return dict(mse=float(e.mean()),mae=float(abs(y-p).mean()),mse_low=float(e[y<10000].mean()),
      mse_ordinary=float(e[y<1000].mean()),mse_tail=float(e[(y>=1000)&(y<10000)].mean()))


def target(y,s):
    return s['cap']*np.tanh(y/s['cap']) if s.get('smooth') else np.minimum(y,s['cap'])


def qualify_external(external):
    measurements=pd.read_csv(ROOT/'data/external/selectivity_measurements.csv')
    kind=measurements.original_type.fillna('').str.lower().str.replace(r'\s+','',regex=True)
    direct=kind.isin(['ratiocc50/ic50','cc50/ic50'])
    paired=measurements.derivation.eq('paired_CC50_ic50_same_document_cell_duration')
    retained=measurements.loc[measurements.virus.eq('VACV')&(direct|paired)&np.isfinite(measurements.log_value)].copy()
    labels=retained.groupby('canonical').log_value.median()
    out=external.copy();out['strict_si']=out.canonical.map(10.**labels)
    out['strict_measurements']=out.canonical.map(retained.groupby('canonical').size()).fillna(0).astype(int)
    retained.to_csv(ROOT/'data/external/strict_ic50_measurements.csv',index=False)
    return out,dict(retained_measurements=len(retained),retained_unique_structures=int(out.strict_si.notna().sum()),
        original_unique_structures=len(external),policy='Explicit CC50/IC50 ratios or matched-document/cell/duration CC50/IC50 pairs only; EC50 and unspecified SI excluded')


def prepare(cache):
    cache.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(ROOT/'data/train_unique.csv');r=pd.read_csv(ROOT/'data/train_measurements.csv');t=core.read_test(ROOT/'data/test.csv')
    raw=pd.read_csv(ROOT/'data/train_v2.csv');keep=(raw.CC50_uM>0)&(raw.IC50_uM_VACV>0)&np.isfinite(raw.CC50_uM)&np.isfinite(raw.IC50_uM_VACV)
    assert len(d)==230 and len(r)==237 and np.array_equal(raw.index[keep],r.source_row)
    np.testing.assert_allclose(r.raw_si,raw.loc[keep,'CC50_uM']/raw.loc[keep,'IC50_uM_VACV'],rtol=1e-14)
    grouped=r.groupby('canonical')
    for endpoint,name in [('CC50_uM','log_cc'),('IC50_uM_VACV','log_ic')]:
        d[name]=grouped[endpoint].apply(lambda v:np.log10(v).mean()).reindex(d.canonical).to_numpy()
    d.to_csv(ROOT/'data/train_unique.csv',index=False)
    external=pd.read_csv(ROOT/'data/external/task_si_vacv.csv')
    blocked={identity(s) for s in raw.SMILES.tolist()+t.SMILES.tolist()}
    ids=[identity(s) for s in external.canonical]
    if set(ids)&blocked:raise ValueError('External/local molecular identity overlap')
    external['verified_identity']=ids
    external['scaffold_core']=[core.canonical_record(s)['scaffold'] for s in external.canonical]
    external['raw_si']=10.**external.log_y.to_numpy()
    if not (np.isfinite(external.raw_si).all() and (external.raw_si>0).all()):raise ValueError('Invalid external target')
    external,qualification=qualify_external(external)
    core.json_write(ROOT/'reports/external_qualification.json',qualification)
    external.to_csv(ROOT/'data/external/prepared_si.csv',index=False)
    for kind in ['count3','binary3','density3']:
        n=len(d);nt=len(t);x=fe.matrix(d.canonical.tolist()+t.canonical.tolist()+external.canonical.tolist(),kind)
        np.savez_compressed(cache/f'{kind}.npz',train=x[:n],test=x[n:n+nt],external=x[n+nt:])
    for split in SPLITS:
        if split in [PRIMARY,'scaffold2042']:
            groups=d.scaffold if split.startswith('scaffold') else d.compound;fold=np.full(len(d),-1)
            for f,(_,va) in enumerate(GroupKFold(5,shuffle=True,random_state=2042).split(d,groups=groups)):fold[va]=f
        else:fold=np.load(ROOT/f'reference/folds_{split}.npy')
        np.save(cache/f'folds_{split}.npy',fold)
    protocol=dict(cases=specs(),derived_ensembles={'seed_mean':SEEDS,'seed_median':SEEDS},primary=PRIMARY,confirmation=SPLITS[1:],
      reference_kaggle_mse_reported_by_user=833000,new_kaggle_scores=None,
      source_sha256=core.digest(ROOT/'data/train_v2.csv'),external_sha256=core.digest(ROOT/'data/external/task_si_vacv.csv'),
      source_files_sha256={n:core.digest(ROOT/n) for n in ['pipeline.py','core.py','features.py','preprocess.py']},
      test_features=['SMILES'],output_policy='same as confirmed 833000: positive floor0.01, no arbitrary output ceiling',
      validation_truth='all 237 original complete measured targets; no filtering, clipping or weighting of held-out labels',
      external_policy='exclude every raw train and test neutral achiral canonical-tautomer identity; also purge held-out local scaffolds in scaffold CV',
      auxiliary_targets='CC50/IC50 from TRAIN only; train-fold mean/std; no test concentrations or missingness are inputs',
      shortlist='primary only: two best non-seed candidates under robustness bounds, best ensemble, best diverse candidate; at most four',
      selection='mean relative raw MSE over four splits; mean low MSE<1; mean ordinary<=1.02; worst full<=1.02; worst low<=1.10',
      validation_limit='all these partitions are tuning CV on the same molecules, not an independent final test')
    core.json_write(ROOT/'reports/protocol.json',protocol)
    core.json_write(ROOT/'reports/external_audit.json',dict(rows=len(external),identity_overlaps=0,blocked_train_and_test_identities=len(blocked),scaffold_field='same core.canonical_record as local data'))
    # Reuse the known best full model, and verify preprocessing against its CSV.
    folder=ROOT/'models'/REF;folder.mkdir(exist_ok=True);x=np.load(cache/'count3.npz')
    pp=FoldPreprocessor().fit(x['train']);old_imp=joblib.load(ROOT/'reference/imputer.joblib')
    np.testing.assert_array_equal(pp.transform(x['train']),old_imp.transform(x['train']))
    shutil.copy2(ROOT/'reference/model_42.cbm',folder/'model.cbm');joblib.dump(pp,folder/'preprocessor.joblib');core.json_write(folder/'config.json',dict(spec=specs()[0]))
    pred=load_predict(REF,t.canonical);expected=pd.read_csv(ROOT/'reference/submission_833000.csv').set_index('index')['SI VACV'].reindex(t['index']).to_numpy()
    delta=float(np.max(abs(pred-expected)))
    if delta>1e-8:raise RuntimeError(f'Reference 833000 mismatch: {delta}')
    core.json_write(ROOT/'reports/reference_check.json',dict(max_delta=delta))


def fit_query(s,d,e,x,ext,train_ids,query,blocked_scaffolds=None,save=None):
    part=d.iloc[train_ids];xx=x[train_ids];y=target(part.aggregate_si.to_numpy(),s);weights=np.ones(len(y));aux_info=None
    if s.get('external_weight'):
        ids=np.flatnonzero(np.isfinite(e.strict_si.to_numpy()))
        if blocked_scaffolds is not None:ids=ids[~e.scaffold_core.isin(blocked_scaffolds).to_numpy()]
        xx=np.vstack([xx,ext[ids]]);y=np.r_[y,target(e.strict_si.to_numpy()[ids],s)]
        weights=np.r_[weights,np.full(len(ids),s['external_weight'])]
    pp=FoldPreprocessor(correlation=s.get('correlation'));xf=pp.fit_transform(xx);xq=pp.transform(query)
    if s['kind']=='multi':
        aux=part[['log_cc','log_ic']].to_numpy();mu=aux.mean(0);sd=np.maximum(aux.std(0),1e-8)
        labels=np.column_stack([y/1000.,s['aux_weight']*(aux-mu)/sd]);loss='MultiRMSE'
        aux_info=dict(mean=mu.tolist(),std=sd.tolist(),weight=s['aux_weight'])
    else:labels=y/1000.;loss=f"Tweedie:variance_power={s['power']}"
    m=CatBoostRegressor(loss_function=loss,**dict(BASE,**s.get('params',{})))
    m.fit(xf,labels,sample_weight=weights)
    pred=predict_head(m,xq,s)
    if save is not None:
        save.mkdir(parents=True,exist_ok=True);m.save_model(str(save/'model.cbm'));joblib.dump(pp,save/'preprocessor.joblib')
        core.json_write(save/'config.json',dict(spec=s,fit_rows=len(y),local_unique_rows=len(train_ids),features=int(pp.mask_.sum()),auxiliary_normalizer=aux_info))
    return pred


def predict_head(m,x,s):
    values=m.predict(x)[:,0] if s['kind']=='multi' else m.predict(x,prediction_type='Exponent')
    return positive(values*1000.)


def cv_job(task):
    s,split,cachepath=task;cache=Path(cachepath);d,r,e=load_data();folder=ROOT/'reports/cases'/s['name'];folder.mkdir(parents=True,exist_ok=True)
    file=folder/f'{split}.json';predfile=folder/f'{split}.npy'
    external_stamp=core.digest(ROOT/'data/external/prepared_si.csv') if s.get('external_weight') else ''
    stamp=hashlib.sha256((core.digest(ROOT/'reports/protocol.json')+json.dumps(s,sort_keys=True)+split+external_stamp).encode()).hexdigest()
    if file.exists() and predfile.exists():
        old=json.loads(file.read_text())
        if old.get('fingerprint')==stamp:return old
    start=time.perf_counter();matrix=np.load(cache/f"{s['feature']}.npz");folds=np.load(cache/f'folds_{split}.npy');oof=np.full(len(d),np.nan)
    groups=d.scaffold if split.startswith('scaffold') else d.compound
    for fold in range(5):
        tr=np.flatnonzero(folds!=fold);va=np.flatnonzero(folds==fold)
        if set(groups.iloc[tr])&set(groups.iloc[va]):raise ValueError('Group leakage')
        blocked=set(d.scaffold.iloc[va]) if split.startswith('scaffold') else None
        oof[va]=fit_query(s,d,e,matrix['train'],matrix['external'],tr,matrix['train'][va],blocked)
    if not np.isfinite(oof).all():raise ValueError('Incomplete OOF')
    np.save(predfile,oof);info=dict(name=s['name'],split=split,spec=s,fingerprint=stamp,seconds=time.perf_counter()-start,**score(oof,d,r))
    core.json_write(file,info);print('DONE',split,s['name'],round(info['mse'],2),round(info['mse_low'],2),round(info['seconds'],1),flush=True);return info


def run(tasks,workers):
    rows=[];errors=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={pool.submit(cv_job,t):t for t in tasks}
        for future in as_completed(pending):
            try:rows.append(future.result())
            except Exception:errors.append(dict(task=pending[future][:2],traceback=traceback.format_exc()));print('FAILED',errors[-1],flush=True)
    if errors:core.json_write(ROOT/'reports/failures.json',errors);raise RuntimeError('Failed training jobs')
    return rows


def derived(split):
    d,r,e=load_data();values=np.stack([np.load(ROOT/f'reports/cases/{name}/{split}.npy') for name in SEEDS]);out=[]
    for name,pred in [('seed_mean',values.mean(0)),('seed_median',np.median(values,axis=0))]:
        folder=ROOT/'reports/cases'/name;folder.mkdir(parents=True,exist_ok=True);np.save(folder/f'{split}.npy',pred)
        info=dict(name=name,split=split,seconds=0.,shared_fit=SEEDS,**score(pred,d,r));core.json_write(folder/f'{split}.json',info);out.append(info)
    return out


def primary(args):
    cache=Path(args.cache).resolve();prepare(cache)
    rows=run([(s,PRIMARY,str(cache)) for s in specs()],args.workers)+derived(PRIMARY)
    pd.DataFrame([{k:v for k,v in a.items() if k!='spec'} for a in rows]).sort_values('mse').to_csv(ROOT/'reports/primary.csv',index=False)


def confirm(args):
    cache=Path(args.cache).resolve();registry={s['name']:s for s in specs()};table=pd.read_csv(ROOT/'reports/primary.csv');d,r,e=load_data()
    reference=np.load(ROOT/f'reports/cases/{REF}/{PRIMARY}.npy');base=score(reference,d,r);screen=[]
    # Screen the same fixed 25/50% blends that will be evaluated at confirmation.
    # This avoids discarding a useful component solely on its standalone error.
    for name in table.name:
        if name==REF:continue
        pred=np.load(ROOT/f'reports/cases/{name}/{PRIMARY}.npy')
        for alpha in [1.,.25,.5]:screen.append(dict(name=name,alpha=alpha,**score(positive((1-alpha)*reference+alpha*pred),d,r)))
    screen=pd.DataFrame(screen);screen.to_csv(ROOT/'reports/primary_blend_screen.csv',index=False)
    bounded=screen[(screen.mse<=base['mse']*1.01)&(screen.mse_low<=base['mse_low']*1.05)&(screen.mse_ordinary<=base['mse_ordinary']*1.05)]
    nonseed=bounded[~bounded.name.isin(SEEDS+['seed_mean','seed_median'])].sort_values('mse').drop_duplicates('name')
    names=nonseed.nsmallest(2,'mse').name.tolist()
    names+=bounded[bounded.name.isin(['seed_mean','seed_median'])].nsmallest(1,'mse').name.tolist()
    diverse=bounded[bounded.name.map(lambda n:n in registry and registry[n]['category'] in ['features','multitask','external'])]
    names+=diverse.nsmallest(1,'mse_ordinary').name.tolist();names=list(dict.fromkeys(names))
    if len(names)<4:names+=bounded[~bounded.name.isin(names)].sort_values('mse_low').drop_duplicates('name').head(4-len(names)).name.tolist()
    if not names:names=table[table.name!=REF].nsmallest(2,'mse').name.tolist()
    names=names[:4];needed={REF}
    for name in names:needed.update(SEEDS if name.startswith('seed_') and name in ['seed_mean','seed_median'] else [name])
    core.json_write(ROOT/'reports/frozen_shortlist.json',dict(names=names,components=sorted(needed),chosen_before_confirmation=True,
        screen_blends=[.25,.5,1.],screen_expanded_after_primary_before_confirmation=True,diverse_choice='ordinary-SI MSE',created=time.time()))
    seconds=table.set_index('name').seconds.to_dict()
    tasks=[(registry[n],split,str(cache)) for split in SPLITS[1:] for n in sorted(needed)]
    tasks.sort(key=lambda t:-seconds.get(t[0]['name'],0))
    run(tasks,args.workers)
    if any(n in ['seed_mean','seed_median'] for n in names):
        for split in SPLITS[1:]:derived(split)
    finalize(names,needed,cache)


def component_formula(name):
    return dict(members=SEEDS,weights=[1/3]*3,reducer='median' if name=='seed_median' else 'mean') if name in ['seed_mean','seed_median'] else dict(members=[name],weights=[1.],reducer='mean')


def evaluate_formula(f,values):
    a=np.stack([values[n] for n in f['members']])
    return positive(np.median(a,axis=0) if f.get('reducer')=='median' else np.sum(a*np.asarray(f['weights'])[:,None],axis=0))


def finalize(names,needed,cache):
    d,r,e=load_data();formulas={REF:component_formula(REF)}
    for name in names:
        formulas[name]=component_formula(name)
        # A separate node allows correct fixed blending of a median ensemble.
        for alpha in [.25,.5]:formulas[name+f'_blend{int(alpha*100)}']=dict(members=[REF,name],weights=[1-alpha,alpha],reducer='mean')
    rows=[]
    for split in SPLITS:
        values={n:np.load(ROOT/f'reports/cases/{n}/{split}.npy') for n in needed}
        for name in names:
            if name not in values:values[name]=evaluate_formula(component_formula(name),values)
        base=score(values[REF],d,r)
        for name,f in formulas.items():
            metrics=score(evaluate_formula(f,values),d,r)
            rows.append(dict(name=name,split=split,relative_mse=metrics['mse']/base['mse'],relative_low=metrics['mse_low']/base['mse_low'],relative_ordinary=metrics['mse_ordinary']/base['mse_ordinary'],**metrics))
    table=pd.DataFrame(rows);table.to_csv(ROOT/'reports/validation.csv',index=False)
    ranking=table.groupby('name',sort=False).agg(mean_relative_mse=('relative_mse','mean'),worst_relative_mse=('relative_mse','max'),mean_relative_low=('relative_low','mean'),worst_relative_low=('relative_low','max'),mean_relative_ordinary=('relative_ordinary','mean'),mean_mse=('mse','mean')).reset_index().sort_values('mean_relative_mse')
    ranking['eligible']=(ranking.mean_relative_mse<1)&(ranking.worst_relative_mse<=1.02)&(ranking.mean_relative_low<1)&(ranking.mean_relative_ordinary<=1.02)&(ranking.worst_relative_low<=1.10)
    ranking.to_csv(ROOT/'reports/selection.csv',index=False);ok=ranking[ranking.eligible]
    selected_by_train_cv=ok.iloc[0]['name'] if len(ok) else REF
    # This delivery is pinned to the ensemble requested after the user's Kaggle evaluation.
    selected='l2_50_blend25'
    if selected not in formulas:raise RuntimeError('Pinned delivery ensemble is missing from confirmation formulas')
    config=dict(selected=selected,formulas=formulas,components=sorted(needed),source_specs={s['name']:s for s in specs()},reference_kaggle_mse_reported_by_user=833000,new_kaggle_scores=None,test_features=['SMILES'],output_policy='same as 833000: floor0.01, no fixed ceiling')
    config.update(selected_by_train_cv=selected_by_train_cv,selection_basis='Fixed 75/25 delivery; train-CV winner recorded separately')
    core.json_write(ROOT/'models/config.json',config);registry={s['name']:s for s in specs()}
    for name in sorted(needed):
        if name==REF:continue
        s=registry[name];a=np.load(cache/f"{s['feature']}.npz");start=time.perf_counter()
        fit_query(s,d,e,a['train'],a['external'],np.arange(len(d)),a['test'],save=ROOT/'models'/name)
        core.json_write(ROOT/f'models/{name}/fit_time.json',dict(seconds=time.perf_counter()-start))
    test=core.read_test(ROOT/'data/test.csv');pred={n:load_predict(n,test.canonical) for n in needed}
    for name in names:
        if name not in pred:pred[name]=evaluate_formula(component_formula(name),pred)
    for name,f in formulas.items():core.write_submission(ROOT/f'submission_{name}.csv',test,evaluate_formula(f,pred),ROOT/'data/sample_submit.csv')
    core.write_submission(ROOT/'submission.csv',test,evaluate_formula(formulas[selected],pred),ROOT/'data/sample_submit.csv')
    print('SELECTED',selected,'\n'+ranking.to_string(index=False),flush=True)


def load_predict(name,smiles):
    folder=ROOT/'models'/name;info=json.loads((folder/'config.json').read_text());s=info['spec']
    x=joblib.load(folder/'preprocessor.joblib').transform(fe.matrix(smiles,s['feature']));m=CatBoostRegressor();m.load_model(str(folder/'model.cbm'))
    return predict_head(m,x,s)


def predict_named(config,name,test,memo=None):
    memo={} if memo is None else memo
    if name in memo:return memo[name]
    if name in config['components']:value=load_predict(name,test.canonical)
    else:
        f=config['formulas'][name];values={n:predict_named(config,n,test,memo) for n in f['members']};value=evaluate_formula(f,values)
    memo[name]=value;return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ['primary','confirm']:
        p=sub.add_parser(command);p.add_argument('--cache',default='.cache');p.add_argument('--workers',type=int,default=4)
    p=sub.add_parser('predict');p.add_argument('--model',default='selected');p.add_argument('--test',type=Path,default=ROOT/'data/test.csv');p.add_argument('--template',type=Path,default=None);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='primary':primary(args)
    elif args.command=='confirm':confirm(args)
    else:
        config=json.loads((ROOT/'models/config.json').read_text());name=config['selected'] if args.model=='selected' else args.model
        test=core.read_test(args.test);core.write_submission(args.output,test,predict_named(config,name,test),args.template)


if __name__=='__main__':main()
