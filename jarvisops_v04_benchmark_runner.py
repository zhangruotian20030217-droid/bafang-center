from __future__ import annotations
import argparse, gc, json, math, os, statistics, time
from pathlib import Path
from hashlib import sha256
import numpy as np
from scipy.special import expit
from flybrain import FlyBrain
from flybrain.reservoir import Trace, bases_for, project, fit_logistic

BASE=Path(__file__).resolve().parent
INC=json.loads((BASE/'benchmark_data.json').read_text())
ART=json.loads((BASE/'artifact_aggregate_v04.json').read_text())
FLY_DATA=Path(os.environ.get('FLY_DATA',str(Path.home()/'fly-data'))).expanduser()
SEED=6407; STEPS=40; STIM_START=8; STIM_END=28
REWIRE_SEEDS=list(range(1307,1357))
CHANNEL_MAP={'CONTROL_BREACH':(['LC4','LPLC2'],'L'),'SCHEDULER_DEPENDENCY':(['LC4','LPLC2'],'R'),'STATE_DRIFT':(['LC10a'],'L'),'OBSERVABILITY_CONFLICT':(['LC10a'],'R'),'GATE_RECEIPT':(['LPLC1'],'L')}
FEATURE_INDEX={n:i for i,n in enumerate(INC['feature_names'])}
ART_BY_ID={r['incident_id']:r for r in ART['rows']}
ENCODER_NAMES=['E1_CONTROL_SEMANTIC','E2_EVIDENCE_LIFECYCLE','E3_AUTHORITY_CONTROL','E4_RUNTIME_RECOVERY']

def f(row,name): return float(row['structured_vector'][FEATURE_INDEX[name]]) if name in FEATURE_INDEX else 0.0
def anyf(row,names): return max([f(row,n) for n in names]+[0.0])
def encoders(row):
    e1=dict(row['fly_channels'])
    e2=dict(ART_BY_ID[row['incident_id']]['e2'])
    e3={
      'CONTROL_BREACH':max(anyf(row,['AREA::NETWORK_EGRESS','AREA::WRITER_CONTROL','AREA::PRODUCTION_CONTROL']),.7*f(row,'AREA::AUTHORITY')),
      'SCHEDULER_DEPENDENCY':max(f(row,'AREA::SCHEDULER'),.7*f(row,'AREA::DEPENDENCY')),
      'STATE_DRIFT':anyf(row,['AREA::CANONICAL_STATE','AREA::POINTER']),
      'OBSERVABILITY_CONFLICT':max(f(row,'AREA::OBSERVABILITY'),f(row,'EVIDENCE::CONFLICTING_EVIDENCE')),
      'GATE_RECEIPT':anyf(row,['AREA::GATE','AREA::RECEIPT'])}
    e4={
      'CONTROL_BREACH':max(f(row,'REGRESSION::FAIL_OR_REOPENED'),.5*f(row,'REGRESSION::PARTIAL')),
      'SCHEDULER_DEPENDENCY':f(row,'RUNTIME::NATURAL_RUNTIME'),
      'STATE_DRIFT':max(f(row,'RUNTIME::RECOVERY_RUNTIME'),.7*f(row,'RUNTIME::DRY_RUN')),
      'OBSERVABILITY_CONFLICT':max(f(row,'EVIDENCE::CONFLICTING_EVIDENCE'),.5*f(row,'EVIDENCE::UNVERIFIED')),
      'GATE_RECEIPT':max(f(row,'REPAIR::VERIFIED'),.6*f(row,'REPAIR::PARTIAL'))}
    return {'E1_CONTROL_SEMANTIC':e1,'E2_EVIDENCE_LIFECYCLE':e2,'E3_AUTHORITY_CONTROL':e3,'E4_RUNTIME_RECOVERY':e4}

def sig(x):
    if x>=0: z=math.exp(-min(x,60.0)); return 1/(1+z)
    z=math.exp(max(x,-60.0)); return z/(1+z)
def std(train,test):
    n=len(train); d=len(train[0]); m=[sum(r[j] for r in train)/n for j in range(d)]; v=[sum((r[j]-m[j])**2 for r in train)/n for j in range(d)]; s=[max(x**.5,1e-6) for x in v]
    return [[(r[j]-m[j])/s[j] for j in range(d)] for r in train],[(test[j]-m[j])/s[j] for j in range(d)]
def fit_py(X,y,l2=.5,lr=.08,epochs=700):
    d=len(X[0]); w=[0.0]*d; b=0.0; n=len(X)
    for _ in range(epochs):
        gw=[0.0]*d; gb=0.0
        for x,t in zip(X,y):
            p=sig(sum(a*z for a,z in zip(w,x))+b); e=p-t
            for j in range(d): gw[j]+=e*x[j]
            gb+=e
        for j in range(d): gw[j]=gw[j]/n+l2*w[j]; w[j]-=lr*gw[j]
        b-=lr*gb/n
    return w,b
def linear_loocv(X,y):
    out=[]
    for i in range(len(X)):
        tr=[r for k,r in enumerate(X) if k!=i]; ty=[t for k,t in enumerate(y) if k!=i]; tr,z=std(tr,X[i]); w,b=fit_py(tr,ty); out.append(sig(sum(a*q for a,q in zip(w,z))+b))
    return out
def reservoir_loocv(X,y,k=10,lam=.5):
    X=np.asarray(X,np.float32); y=np.asarray(y,np.float64); out=[]
    for i in range(len(X)):
        mask=np.ones(len(X),bool); mask[i]=False; Xtr,ytr=X[mask],y[mask]; kk=min(k,len(Xtr),Xtr.shape[1]); basis=bases_for(Xtr,[kk])[kk]; Z=project(basis,Xtr); w,b=fit_logistic(Z,ytr,lam); out.append(float(expit(project(basis,X[i:i+1])@w+b)[0]))
    return out
def auc(y,s):
    p=[v for t,v in zip(y,s) if t]; n=[v for t,v in zip(y,s) if not t]; return sum(1 if a>b else .5 if a==b else 0 for a in p for b in n)/(len(p)*len(n))
def metrics(y,s):
    p=[int(v>=.5) for v in s]; tp=sum(a==b==1 for a,b in zip(y,p)); tn=sum(a==b==0 for a,b in zip(y,p)); fp=sum(a==0 and b==1 for a,b in zip(y,p)); fn=sum(a==1 and b==0 for a,b in zip(y,p)); pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0; f1=2*pr*rc/(pr+rc) if pr+rc else 0
    return {'accuracy':(tp+tn)/len(y),'precision':pr,'recall':rc,'f1':f1,'roc_auc':auc(y,s),'tp':tp,'tn':tn,'fp':fp,'fn':fn,'threshold':.5}
def dist(xs):
    xs=sorted(xs); q=lambda p: xs[round((len(xs)-1)*p)]
    return {'n':len(xs),'mean':statistics.fmean(xs),'median':statistics.median(xs),'min':min(xs),'max':max(xs),'p05':q(.05),'p95':q(.95)}
def empirical_p(real,controls): return (1+sum(x>=real for x in controls))/(1+len(controls))
def lift(y,b,n,ids):
    bm=metrics(y,b); nm=metrics(y,n); rec=[iid for iid,t,x,z in zip(ids,y,b,n) if t==1 and x<.5 and z>=.5]; fp=[iid for iid,t,x,z in zip(ids,y,b,n) if t==0 and x<.5 and z>=.5]
    return {'accuracy_delta':nm['accuracy']-bm['accuracy'],'recall_delta':nm['recall']-bm['recall'],'f1_delta':nm['f1']-bm['f1'],'roc_auc_delta':nm['roc_auc']-bm['roc_auc'],'baseline_false_negative_count':bm['fn'],'recovered_false_negative_count':len(rec),'recovered_false_negative_ids':rec,'new_false_positive_count':len(fp),'new_false_positive_ids':fp}
def fusions(b,f): return {'ENSEMBLE_75J_25F':[.75*x+.25*z for x,z in zip(b,f)],'ENSEMBLE_50J_50F':[.5*x+.5*z for x,z in zip(b,f)],'ENSEMBLE_RISK_OR':[max(x,z) for x,z in zip(b,f)]}
def cells(brain):
    out={}
    for name,(types,side) in CHANNEL_MAP.items():
        idx=brain.cells(types,side=side)
        if len(idx)==0: raise RuntimeError('unresolved encoder '+name)
        out[name]=idx
    return out
def run_features(brain,cellmap,enc):
    tr=Trace(brain,types=['descending_neuron'],tau=.1); out=[]; times=[]
    for row in INC['rows']:
        t0=time.perf_counter(); brain.reset(SEED+sum(map(ord,row['incident_id']))); tr.reset(); feat=None; ch=encoders(row)[enc]
        for t in range(STEPS):
            inj=[]
            if STIM_START<=t<STIM_END:
                for name,val in ch.items():
                    if val>0: inj.append((cellmap[name],np.float32(.80*val)))
            feat=tr.observe(brain.step(inject=inj))
        out.append(np.asarray(feat,np.float32).ravel()); times.append((time.perf_counter()-t0)*1000)
    return np.stack(out),{'feature_count':int(len(out[0])),'mean_incident_runtime_ms':float(np.mean(times))}
def rewire(brain,orig_i,orig_w,seed):
    rng=np.random.default_rng(seed); brain.indices=rng.permutation(orig_i).astype(orig_i.dtype); w=orig_w.copy()
    if not brain.sensory_input and brain.superclass is not None:
        sensory=np.char.find(brain.superclass.astype(str),'sensory')>=0; w=np.where(sensory[brain.indices],0,w)
    brain.weights=w.astype(orig_w.dtype)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--rewires',type=int,default=50); args=ap.parse_args(); seeds=REWIRE_SEEDS[:args.rewires]
    ids=[r['incident_id'] for r in INC['rows']]; y=[int(r['false_pass']) for r in INC['rows']]
    art=[ART_BY_ID[i]['artifact_vector'] for i in ids]; incident=[r['structured_vector'] for r in INC['rows']]; full=[a+b for a,b in zip(incident,art)]
    b0=linear_loocv(incident,y); b1=linear_loocv(full,y)
    result={'benchmark_id':'JARVISOPS-FLY-FULL-CORPUS-V0.4-LIVE','incident_count':len(ids),'repair_artifact_count':ART['repair_artifact_count'],'encoder_count':4,'rewire_control_count':len(seeds),'target_policy':'Only formal P0 incident labels; artifacts never create/override False-PASS labels.','results':{'JARVISOPS_INCIDENT_ONLY':{'metrics':metrics(y,b0),'scores':dict(zip(ids,b0))},'JARVISOPS_INCIDENT_PLUS_ARTIFACT':{'metrics':metrics(y,b1),'scores':dict(zip(ids,b1))}},'encoders':{},'runtime':{'python':os.sys.version,'flybrain_version':'0.1.0','fly_data':str(FLY_DATA),'numba_threads':os.environ.get('NUMBA_NUM_THREADS')}}
    brain=FlyBrain(data=FLY_DATA,seed=SEED,device='cpu',sensory_input=False); oi=brain.indices.copy(); ow=brain.weights.copy(); cm=cells(brain); tstart=time.time()
    real={}
    for enc in ENCODER_NAMES:
        brain.indices=oi.copy(); brain.weights=ow.copy(); X,meta=run_features(brain,cm,enc); sc=reservoir_loocv(X,y); real[enc]=sc; result['encoders'][enc]={'real_fly':{'metrics':metrics(y,sc),'runtime':meta,'scores':dict(zip(ids,sc))},'ensembles':{},'rewired_controls':{}}
        for name,es in fusions(b1,sc).items(): result['encoders'][enc]['ensembles'][name]={'metrics':metrics(y,es),'lift_vs_jarvisops_full':lift(y,b1,es,ids),'scores':dict(zip(ids,es))}
    ctrl={e:{'fly_auc':[],'fly_acc':[],'ensemble':{n:{'auc':[],'recall':[],'f1':[]} for n in ['ENSEMBLE_75J_25F','ENSEMBLE_50J_50F','ENSEMBLE_RISK_OR']}} for e in ENCODER_NAMES}
    for seed in seeds:
        rewire(brain,oi,ow,seed)
        for enc in ENCODER_NAMES:
            X,_=run_features(brain,cm,enc); sc=reservoir_loocv(X,y); m=metrics(y,sc); ctrl[enc]['fly_auc'].append(m['roc_auc']); ctrl[enc]['fly_acc'].append(m['accuracy'])
            for name,es in fusions(b1,sc).items():
                mm=metrics(y,es); ctrl[enc]['ensemble'][name]['auc'].append(mm['roc_auc']); ctrl[enc]['ensemble'][name]['recall'].append(mm['recall']); ctrl[enc]['ensemble'][name]['f1'].append(mm['f1'])
    for enc in ENCODER_NAMES:
        e=result['encoders'][enc]; c=ctrl[enc]; rm=e['real_fly']['metrics']; e['rewired_controls']={'seeds':seeds,'fly_auc_distribution':dist(c['fly_auc']),'fly_accuracy_distribution':dist(c['fly_acc']),'real_fly_auc_empirical_p_ge':empirical_p(rm['roc_auc'],c['fly_auc']),'real_fly_auc_percentile':sum(x<rm['roc_auc'] for x in c['fly_auc'])/len(c['fly_auc']),'ensemble_distributions':{}}
        for name,z in c['ensemble'].items():
            rr=e['ensembles'][name]['metrics']; e['rewired_controls']['ensemble_distributions'][name]={'auc':dist(z['auc']),'recall':dist(z['recall']),'f1':dist(z['f1']),'real_ensemble_auc_empirical_p_ge':empirical_p(rr['roc_auc'],z['auc'])}
    cand=[]
    for enc,e in result['encoders'].items():
        for name,z in e['ensembles'].items(): cand.append((z['metrics']['f1'],z['metrics']['recall'],z['metrics']['roc_auc'],enc,name,z))
    cand.sort(reverse=True); z=cand[0]; result['best_observed_exploratory']={'encoder':z[3],'ensemble':z[4],'metrics':z[5]['metrics'],'lift_vs_jarvisops_full':z[5]['lift_vs_jarvisops_full'],'selection_warning':'Selected on same 24-incident corpus; exploratory only.'}
    result['runtime']['total_runtime_s']=time.time()-tstart; result['evidence_status']='EXPERIMENTAL_MODEL_SIGNAL'; result['production_authority']=False
    raw=json.dumps(result,sort_keys=True,separators=(',',':')).encode(); result['report_sha256']=sha256(raw).hexdigest(); Path('benchmark_result_v04.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
