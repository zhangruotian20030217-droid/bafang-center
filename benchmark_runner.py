from __future__ import annotations
import gc, json, os, time
from pathlib import Path
from hashlib import sha256
import numpy as np
from scipy.special import expit
from flybrain import FlyBrain
from flybrain.reservoir import Trace, bases_for, project, fit_logistic

BASE=Path(__file__).resolve().parent
DATA=json.loads((BASE/'benchmark_data.json').read_text())
FLY_DATA=Path(os.environ.get('FLY_DATA', str(Path.home()/'fly-data'))).expanduser()
SEED=6407; STEPS=40; STIM_START=8; STIM_END=28; REWIRE_SEED=1307
CHANNEL_MAP={
 'CONTROL_BREACH':(['LC4','LPLC2'],'L'),
 'SCHEDULER_DEPENDENCY':(['LC4','LPLC2'],'R'),
 'STATE_DRIFT':(['LC10a'],'L'),
 'OBSERVABILITY_CONFLICT':(['LC10a'],'R'),
 'GATE_RECEIPT':(['LPLC1'],'L'),
}

def reservoir_loocv(X,y,k=10,lam=.5):
    X=np.asarray(X,np.float32); y=np.asarray(y,np.float64); out=[]
    for i in range(len(X)):
        mask=np.ones(len(X),bool); mask[i]=False
        Xtr,ytr=X[mask],y[mask]; kk=min(k,len(Xtr),Xtr.shape[1])
        basis=bases_for(Xtr,[kk])[kk]
        w,b=fit_logistic(project(basis,Xtr),ytr,lam)
        out.append(float(expit(project(basis,X[i:i+1])@w+b)[0]))
    return out

def metrics(y,scores):
    p=[1 if s>=.5 else 0 for s in scores]
    tp=sum(a==1 and b==1 for a,b in zip(y,p)); tn=sum(a==0 and b==0 for a,b in zip(y,p))
    fp=sum(a==0 and b==1 for a,b in zip(y,p)); fn=sum(a==1 and b==0 for a,b in zip(y,p))
    prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0
    pos=[s for a,s in zip(y,scores) if a]; neg=[s for a,s in zip(y,scores) if not a]
    auc=sum(1 if a>b else .5 if a==b else 0 for a in pos for b in neg)/(len(pos)*len(neg))
    return {'accuracy':(tp+tn)/len(y),'precision':prec,'recall':rec,'f1':f1,'roc_auc':auc,'tp':tp,'tn':tn,'fp':fp,'fn':fn,'threshold':.5}

def encoder_cells(brain):
    out={}
    for name,(types,side) in CHANNEL_MAP.items():
        idx=brain.cells(types,side=side)
        if len(idx)==0: raise RuntimeError(f'unresolved encoder mapping: {name}')
        out[name]=idx
    return out

def rewire(brain,seed):
    rng=np.random.default_rng(seed)
    brain.indices=rng.permutation(brain.indices).astype(brain.indices.dtype)
    if not brain.sensory_input and brain.superclass is not None:
        sensory=np.char.find(brain.superclass.astype(str),'sensory')>=0
        brain.weights=np.where(sensory[brain.indices],0,brain.weights).astype(brain.weights.dtype)

def run_features(rewired=False):
    t0=time.time(); brain=FlyBrain(data=FLY_DATA,seed=SEED,device='cpu',sensory_input=False)
    if rewired: rewire(brain,REWIRE_SEED)
    trace=Trace(brain,types=['descending_neuron'],tau=.1); cells=encoder_cells(brain); features=[]; times=[]
    for r in DATA['rows']:
        s=time.perf_counter(); brain.reset(SEED+sum(map(ord,r['incident_id']))); trace.reset(); feat=None
        for t in range(STEPS):
            inj=[]
            if STIM_START<=t<STIM_END:
                for name,val in r['fly_channels'].items():
                    if val>0: inj.append((cells[name],np.float32(.80*val)))
            feat=trace.observe(brain.step(inject=inj))
        features.append(np.asarray(feat,np.float32).ravel()); times.append((time.perf_counter()-s)*1000)
    meta={'feature_count':int(len(features[0])),'mean_incident_runtime_ms':float(np.mean(times)),'total_runtime_s':time.time()-t0}
    del trace,cells,brain; gc.collect()
    return np.stack(features),meta

def main():
    y=[int(r['false_pass']) for r in DATA['rows']]; ids=[r['incident_id'] for r in DATA['rows']]
    result={
      'benchmark_id':'JARVISOPS-FLY-P0-FALSEPASS-V0.3-LIVE-CANONICAL',
      'canonical_dataset_sha256':DATA['canonical_dataset_sha256'],
      'incident_count':len(y),'positive_count':sum(y),'negative_count':len(y)-sum(y),
      'runtime':{'python':os.sys.version,'flybrain_version':'0.1.0','fly_data':str(FLY_DATA),'numba_threads':os.environ.get('NUMBA_NUM_THREADS')},
      'results':{
        'JARVISOPS_STRUCTURED_BASELINE_REFERENCE':{'metrics':DATA['structured_baseline_reference'],'source':'canonical JarvisOps v0.3 benchmark'},
        'MAJORITY_BASELINE':{'metrics':metrics(y,[sum(y)/len(y)]*len(y))}
      }
    }
    real,rmeta=run_features(False); rs=reservoir_loocv(real,y)
    result['results']['FLY_CONNECTOME_LOOCV']={'metrics':metrics(y,rs),'runtime':rmeta,'scores':dict(zip(ids,rs))}
    rew,wmeta=run_features(True); ws=reservoir_loocv(rew,y)
    result['results']['DEGREE_PRESERVING_REWIRED_FLY_LOOCV']={'metrics':metrics(y,ws),'runtime':wmeta,'rewire_seed':REWIRE_SEED,'scores':dict(zip(ids,ws))}
    fm=result['results']['FLY_CONNECTOME_LOOCV']['metrics']; wm=result['results']['DEGREE_PRESERVING_REWIRED_FLY_LOOCV']['metrics']; bm=DATA['structured_baseline_reference']
    result['comparison']={
      'fly_auc_minus_rewired_auc':fm['roc_auc']-wm['roc_auc'],
      'fly_accuracy_minus_rewired_accuracy':fm['accuracy']-wm['accuracy'],
      'fly_auc_minus_jarvisops_baseline_auc':fm['roc_auc']-bm['roc_auc'],
      'fly_accuracy_minus_jarvisops_baseline_accuracy':fm['accuracy']-bm['accuracy'],
      'evidence_status':'EXPERIMENTAL_MODEL_SIGNAL',
      'production_authority':False
    }
    raw=json.dumps(result,sort_keys=True,separators=(',',':')).encode(); result['report_sha256']=sha256(raw).hexdigest()
    Path('benchmark_result.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
