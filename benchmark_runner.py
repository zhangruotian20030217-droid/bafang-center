from __future__ import annotations
import gc, json, math, os, time
from pathlib import Path
from hashlib import sha256
import numpy as np
from scipy.special import expit
from flybrain import FlyBrain
from flybrain.reservoir import Trace, bases_for, project, fit_logistic

BASE=Path(__file__).resolve().parent
DATA=json.loads((BASE/'benchmark_data.json').read_text())
FLY_DATA=Path(os.environ.get('FLY_DATA', str(Path.home()/'fly-data'))).expanduser()
SEED=6407
STEPS=40
STIM_START=8
STIM_END=28
REWIRE_SEED=1307
CHANNEL_MAP={
 'CONTROL_BREACH':(['LC4','LPLC2'],'L'),
 'SCHEDULER_DEPENDENCY':(['LC4','LPLC2'],'R'),
 'STATE_DRIFT':(['LC10a'],'L'),
 'OBSERVABILITY_CONFLICT':(['LC10a'],'R'),
 'GATE_RECEIPT':(['LPLC1'],'L'),
}

def sig(x):
    if x>=0:
        z=math.exp(-min(x,60.0)); return 1/(1+z)
    z=math.exp(max(x,-60.0)); return z/(1+z)

def std(train,test):
    n=len(train); d=len(train[0]); m=[sum(r[j] for r in train)/n for j in range(d)]
    v=[sum((r[j]-m[j])**2 for r in train)/n for j in range(d)]
    s=[max(x**0.5,1e-6) for x in v]
    return [[(r[j]-m[j])/s[j] for j in range(d)] for r in train],[(test[j]-m[j])/s[j] for j in range(d)]

def fit_py(X,y,l2=.5,lr=.08,epochs=700):
    d=len(X[0]); w=[0.0]*d; b=0.0; n=len(X)
    for _ in range(epochs):
        gw=[0.0]*d; gb=0.0
        for x,t in zip(X,y):
            p=sig(sum(a*z for a,z in zip(w,x))+b); e=p-t
            for j in range(d): gw[j]+=e*x[j]
            gb+=e
        for j in range(d):
            gw[j]=gw[j]/n+l2*w[j]; w[j]-=lr*gw[j]
        b-=lr*gb/n
    return w,b

def structured_loocv(X,y):
    out=[]
    for i in range(len(X)):
        tr=[r for k,r in enumerate(X) if k!=i]; ty=[t for k,t in enumerate(y) if k!=i]
        tr,z=std(tr,X[i]); w,b=fit_py(tr,ty)
        out.append(sig(sum(a*q for a,q in zip(w,z))+b))
    return out

def reservoir_loocv(X,y,k=10,lam=.5):
    X=np.asarray(X,np.float32); y=np.asarray(y,np.float64); out=[]
    for i in range(len(X)):
        mask=np.ones(len(X),bool); mask[i]=False
        Xtr,ytr=X[mask],y[mask]
        kk=min(k,len(Xtr),Xtr.shape[1])
        basis=bases_for(Xtr,[kk])[kk]
        Z=project(basis,Xtr)
        w,b=fit_logistic(Z,ytr,lam)
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
    return {'accuracy':(tp+tn)/len(y),'precision':prec,'recall':rec,'f1':f1,'roc_auc':auc,'tp':tp,'tn':tn,'fp':fp,'fn':fn}

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
        arr=np.asarray(feat,np.float32).ravel(); features.append(arr); times.append((time.perf_counter()-s)*1000)
    meta={'feature_count':int(len(features[0])),'mean_incident_runtime_ms':float(np.mean(times)),'total_runtime_s':time.time()-t0}
    del trace,cells,brain; gc.collect()
    return np.stack(features),meta

def main():
    y=[int(r['false_pass']) for r in DATA['rows']]
    sx=[r['structured_vector'] for r in DATA['rows']]
    result={
      'benchmark_id':'JARVISOPS-FLY-P0-FALSEPASS-V0.3-LIVE',
      'incident_count':len(y),'positive_count':sum(y),'negative_count':len(y)-sum(y),
      'runtime':{'python':os.sys.version,'fly_data':str(FLY_DATA),'numba_threads':os.environ.get('NUMBA_NUM_THREADS')},
      'results':{}
    }
    result['results']['MAJORITY_BASELINE']=metrics(y,[sum(y)/len(y)]*len(y))
    result['results']['STRUCTURED_LINEAR_LOOCV']=metrics(y,structured_loocv(sx,y))
    real,rmeta=run_features(False); result['results']['FLY_CONNECTOME_LOOCV']={'metrics':metrics(y,reservoir_loocv(real,y)),'runtime':rmeta}
    rew,wmeta=run_features(True); result['results']['DEGREE_PRESERVING_REWIRED_FLY_LOOCV']={'metrics':metrics(y,reservoir_loocv(rew,y)),'runtime':wmeta,'rewire_seed':REWIRE_SEED}
    result['comparison']={
      'fly_auc_minus_rewired_auc':result['results']['FLY_CONNECTOME_LOOCV']['metrics']['roc_auc']-result['results']['DEGREE_PRESERVING_REWIRED_FLY_LOOCV']['metrics']['roc_auc'],
      'fly_accuracy_minus_rewired_accuracy':result['results']['FLY_CONNECTOME_LOOCV']['metrics']['accuracy']-result['results']['DEGREE_PRESERVING_REWIRED_FLY_LOOCV']['metrics']['accuracy'],
      'evidence_status':'EXPERIMENTAL_MODEL_SIGNAL'
    }
    raw=json.dumps(result,sort_keys=True,separators=(',',':')).encode(); result['report_sha256']=sha256(raw).hexdigest()
    Path('benchmark_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
