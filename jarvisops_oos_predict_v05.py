from __future__ import annotations
import json, math, os, time, gc
from pathlib import Path
from hashlib import sha256
import numpy as np
from scipy.special import expit
from flybrain import FlyBrain
from flybrain.reservoir import Trace, bases_for, project, fit_logistic

BASE=Path(__file__).resolve().parent
DEV=json.loads((BASE/'dev_frozen_v05.json').read_text())
OOS=json.loads((BASE/'oos_features_v05.json').read_text())
FLY_DATA=Path(os.environ.get('FLY_DATA',str(Path.home()/'fly-data'))).expanduser()
SEED=6407; STEPS=40; STIM_START=8; STIM_END=28
REWIRE_SEEDS=list(range(1307,1357))
ENCODERS=['E1_CONTROL_SEMANTIC','E3_AUTHORITY_CONTROL','E4_RUNTIME_RECOVERY']
CHANNEL_MAP={
 'CONTROL_BREACH':(['LC4','LPLC2'],'L'),
 'SCHEDULER_DEPENDENCY':(['LC4','LPLC2'],'R'),
 'STATE_DRIFT':(['LC10a'],'L'),
 'OBSERVABILITY_CONFLICT':(['LC10a'],'R'),
 'GATE_RECEIPT':(['LPLC1'],'L'),
}
FIDX={n:i for i,n in enumerate(DEV['feature_names'])}
assert DEV['feature_names']==OOS['feature_names']
assert OOS.get('label_fields_included') is False
assert len(DEV['rows'])==24 and len(OOS['rows'])==6

def file_sha(path): return sha256(Path(path).read_bytes()).hexdigest()
def f(row,name): return float(row['structured_vector'][FIDX[name]]) if name in FIDX else 0.0
def anyf(row,names): return max([f(row,n) for n in names]+[0.0])

def channels(row,enc):
    if enc=='E1_CONTROL_SEMANTIC': return dict(row['e1_fly_channels'])
    if enc=='E3_AUTHORITY_CONTROL':
        return {
          'CONTROL_BREACH':max(anyf(row,['AREA::NETWORK_EGRESS','AREA::WRITER_CONTROL','AREA::PRODUCTION_CONTROL']),.7*f(row,'AREA::AUTHORITY')),
          'SCHEDULER_DEPENDENCY':max(f(row,'AREA::SCHEDULER'),.7*f(row,'AREA::DEPENDENCY')),
          'STATE_DRIFT':anyf(row,['AREA::CANONICAL_STATE','AREA::POINTER']),
          'OBSERVABILITY_CONFLICT':max(f(row,'AREA::OBSERVABILITY'),f(row,'EVIDENCE::CONFLICTING_EVIDENCE')),
          'GATE_RECEIPT':anyf(row,['AREA::GATE','AREA::RECEIPT'])}
    if enc=='E4_RUNTIME_RECOVERY':
        return {
          'CONTROL_BREACH':max(f(row,'REGRESSION::FAIL_OR_REOPENED'),.5*f(row,'REGRESSION::PARTIAL')),
          'SCHEDULER_DEPENDENCY':f(row,'RUNTIME::NATURAL_RUNTIME'),
          'STATE_DRIFT':max(f(row,'RUNTIME::RECOVERY_RUNTIME'),.7*f(row,'RUNTIME::DRY_RUN')),
          'OBSERVABILITY_CONFLICT':max(f(row,'EVIDENCE::CONFLICTING_EVIDENCE'),.5*f(row,'EVIDENCE::UNVERIFIED')),
          'GATE_RECEIPT':max(f(row,'REPAIR::VERIFIED'),.6*f(row,'REPAIR::PARTIAL'))}
    raise KeyError(enc)

def sigmoid(x):
    if x>=0:
        z=math.exp(-min(x,60.0)); return 1/(1+z)
    z=math.exp(max(x,-60.0)); return z/(1+z)

def standardize_fit(train):
    X=np.asarray(train,dtype=np.float64)
    mean=X.mean(axis=0); sd=np.sqrt(((X-mean)**2).mean(axis=0)); sd=np.maximum(sd,1e-6)
    return mean,sd

def fit_py(X,y,l2=.5,lr=.08,epochs=700):
    X=[list(map(float,r)) for r in X]; y=list(map(int,y)); d=len(X[0]); w=[0.0]*d; b=0.0; n=len(X)
    for _ in range(epochs):
        gw=[0.0]*d; gb=0.0
        for x,t in zip(X,y):
            p=sigmoid(sum(a*z for a,z in zip(w,x))+b); e=p-t
            for j in range(d): gw[j]+=e*x[j]
            gb+=e
        for j in range(d):
            gw[j]=gw[j]/n+l2*w[j]; w[j]-=lr*gw[j]
        b-=lr*gb/n
    return w,b

def jarvisops_predict():
    Xtr=np.asarray([r['structured_vector'] for r in DEV['rows']],np.float64)
    y=[int(r['false_pass']) for r in DEV['rows']]
    Xte=np.asarray([r['structured_vector'] for r in OOS['rows']],np.float64)
    mean,sd=standardize_fit(Xtr); tr=(Xtr-mean)/sd; te=(Xte-mean)/sd
    w,b=fit_py(tr.tolist(),y)
    return [sigmoid(float(np.dot(w,row)+b)) for row in te]

def encoder_cells(brain):
    out={}
    for name,(types,side) in CHANNEL_MAP.items():
        idx=brain.cells(types,side=side)
        if len(idx)==0: raise RuntimeError('unresolved encoder '+name)
        out[name]=idx
    return out

def run_features(brain,cellmap,rows,enc):
    trace=Trace(brain,types=['descending_neuron'],tau=.1); out=[]
    for row in rows:
        brain.reset(SEED+sum(map(ord,row['incident_id']))); trace.reset(); feat=None; ch=channels(row,enc)
        for t in range(STEPS):
            inject=[]
            if STIM_START<=t<STIM_END:
                for name,val in ch.items():
                    if val>0: inject.append((cellmap[name],np.float32(.80*val)))
            feat=trace.observe(brain.step(inject=inject))
        out.append(np.asarray(feat,np.float32).ravel())
    return np.stack(out)

def reservoir_train_predict(Xtr,y,Xte,k=10,lam=.5):
    Xtr=np.asarray(Xtr,np.float32); Xte=np.asarray(Xte,np.float32); y=np.asarray(y,np.float64)
    kk=min(k,len(Xtr),Xtr.shape[1]); basis=bases_for(Xtr,[kk])[kk]
    Z=project(basis,Xtr); w,b=fit_logistic(Z,y,lam)
    return [float(x) for x in expit(project(basis,Xte)@w+b)]

def rewire(brain,orig_i,orig_w,seed):
    rng=np.random.default_rng(seed); brain.indices=rng.permutation(orig_i).astype(orig_i.dtype); w=orig_w.copy()
    if not brain.sensory_input and brain.superclass is not None:
        sensory=np.char.find(brain.superclass.astype(str),'sensory')>=0; w=np.where(sensory[brain.indices],0,w)
    brain.weights=w.astype(orig_w.dtype)

def fusions(j,f):
    return {
      'ENSEMBLE_75J_25F':[.75*a+.25*b for a,b in zip(j,f)],
      'ENSEMBLE_50J_50F':[.50*a+.50*b for a,b in zip(j,f)],
      'ENSEMBLE_RISK_OR':[max(a,b) for a,b in zip(j,f)],
    }

def main():
    started=time.time(); ids=[r['incident_id'] for r in OOS['rows']]; ydev=[int(r['false_pass']) for r in DEV['rows']]
    j=jarvisops_predict()
    result={
      'receipt_type':'PRELABEL_PREDICTION_RECEIPT','research_phase':'v0.5','benchmark_id':'JARVISOPS-OOS-P1P2-V0.5-PRELABEL',
      'oos_labels_loaded':False,'oos_incident_count':len(ids),'development_incident_count':len(DEV['rows']),
      'primary_candidate':{'encoder':'E3_AUTHORITY_CONTROL','fusion':'ENSEMBLE_75J_25F','threshold':.5,'endpoint':'FALSE_PASS_FALSE_NEGATIVE_RECOVERY_AT_FIXED_FALSE_POSITIVE_BUDGET'},
      'frozen_contract':{'encoders':['E1_CONTROL_SEMANTIC','E3_AUTHORITY_CONTROL','E4_RUNTIME_RECOVERY'],'e2_status':'UNAVAILABLE_NO_ISOMORPHIC_OOS_REPAIR_ARTIFACT_INPUT','fusion_policies':['ENSEMBLE_75J_25F','ENSEMBLE_50J_50F','ENSEMBLE_RISK_OR'],'threshold':.5,'rewire_seeds':REWIRE_SEEDS},
      'input_sha256':{'dev_frozen_v05.json':file_sha(BASE/'dev_frozen_v05.json'),'oos_features_v05.json':file_sha(BASE/'oos_features_v05.json'),'runner':file_sha(BASE/'jarvisops_oos_predict_v05.py')},
      'jarvisops_incident_only_scores':dict(zip(ids,j)),'real_fly':{},'rewired_controls':{},
    }
    brain=FlyBrain(data=FLY_DATA,seed=SEED,device='cpu',sensory_input=False); oi=brain.indices.copy(); ow=brain.weights.copy(); cm=encoder_cells(brain)
    result['brain_sha256']={'brain.npz':file_sha(FLY_DATA/'brain.npz'),'weights.npz':file_sha(FLY_DATA/'weights.npz')}
    for enc in ENCODERS:
        brain.indices=oi.copy(); brain.weights=ow.copy()
        Xtr=run_features(brain,cm,DEV['rows'],enc); Xte=run_features(brain,cm,OOS['rows'],enc)
        fs=reservoir_train_predict(Xtr,ydev,Xte)
        result['real_fly'][enc]={'scores':dict(zip(ids,fs)),'ensembles':{n:dict(zip(ids,s)) for n,s in fusions(j,fs).items()},'feature_count':int(Xtr.shape[1])}
    for enc in ENCODERS: result['rewired_controls'][enc]=[]
    for seed in REWIRE_SEEDS:
        rewire(brain,oi,ow,seed)
        for enc in ENCODERS:
            Xtr=run_features(brain,cm,DEV['rows'],enc); Xte=run_features(brain,cm,OOS['rows'],enc)
            fs=reservoir_train_predict(Xtr,ydev,Xte)
            result['rewired_controls'][enc].append({'seed':seed,'fly_scores':dict(zip(ids,fs)),'ensembles':{n:dict(zip(ids,s)) for n,s in fusions(j,fs).items()}})
        gc.collect()
    result['runtime']={'python':os.sys.version,'flybrain_version':'0.1.0','numba_threads':os.environ.get('NUMBA_NUM_THREADS'),'total_runtime_s':time.time()-started}
    raw=json.dumps(result,sort_keys=True,separators=(',',':')).encode(); result['preseal_sha256']=sha256(raw).hexdigest()
    Path('oos_predictions_prelabel_v05.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'status':'PRELABEL_PREDICTIONS_COMPLETE','oos_labels_loaded':False,'preseal_sha256':result['preseal_sha256'],'runtime':result['runtime']},indent=2))
if __name__=='__main__': main()
