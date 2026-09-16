from __future__ import annotations
import argparse, gc, json, os, statistics, time
from pathlib import Path
from hashlib import sha256
import numpy as np
from scipy.special import expit
from flybrain import FlyBrain
from flybrain.reservoir import Trace, bases_for, project, fit_logistic

BASE = Path(__file__).resolve().parent
INC = json.loads((BASE / 'benchmark_data.json').read_text())
ART = json.loads((BASE / 'artifact_aggregate_v04.json').read_text())
REF = json.loads((BASE / 'baseline_reference_v04.json').read_text())
FLY_DATA = Path(os.environ.get('FLY_DATA', str(Path.home() / 'fly-data'))).expanduser()
SEED = 6407
STEPS = 40
STIM_START = 8
STIM_END = 28
REWIRE_SEEDS = list(range(1307, 1357))
CHANNEL_MAP = {
    'CONTROL_BREACH': (['LC4', 'LPLC2'], 'L'),
    'SCHEDULER_DEPENDENCY': (['LC4', 'LPLC2'], 'R'),
    'STATE_DRIFT': (['LC10a'], 'L'),
    'OBSERVABILITY_CONFLICT': (['LC10a'], 'R'),
    'GATE_RECEIPT': (['LPLC1'], 'L'),
}
ENCODER_NAMES = [
    'E1_CONTROL_SEMANTIC',
    'E2_EVIDENCE_LIFECYCLE',
    'E3_AUTHORITY_CONTROL',
    'E4_RUNTIME_RECOVERY',
]
ART_BY_ID = {r['incident_id']: r for r in ART['rows']}
REF_BY_ID = {r['incident_id']: r for r in REF['rows']}


def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def encoder_channels(row):
    iid = row['incident_id']
    return {
        'E1_CONTROL_SEMANTIC': dict(row['fly_channels']),
        'E2_EVIDENCE_LIFECYCLE': dict(ART_BY_ID[iid]['e2']),
        'E3_AUTHORITY_CONTROL': dict(REF_BY_ID[iid]['e3']),
        'E4_RUNTIME_RECOVERY': dict(REF_BY_ID[iid]['e4']),
    }


def reservoir_loocv(X, y, k=10, lam=.5):
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.float64)
    out = []
    for i in range(len(X)):
        mask = np.ones(len(X), bool)
        mask[i] = False
        Xtr, ytr = X[mask], y[mask]
        kk = min(k, len(Xtr), Xtr.shape[1])
        basis = bases_for(Xtr, [kk])[kk]
        Z = project(basis, Xtr)
        w, b = fit_logistic(Z, ytr, lam)
        out.append(float(expit(project(basis, X[i:i + 1]) @ w + b)[0]))
    return out


def auc(y, scores):
    pos = [s for t, s in zip(y, scores) if t]
    neg = [s for t, s in zip(y, scores) if not t]
    return sum(1 if a > b else .5 if a == b else 0 for a in pos for b in neg) / (len(pos) * len(neg))


def metrics(y, scores):
    pred = [int(v >= .5) for v in scores]
    tp = sum(a == b == 1 for a, b in zip(y, pred))
    tn = sum(a == b == 0 for a, b in zip(y, pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, pred))
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {
        'accuracy': (tp + tn) / len(y),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': auc(y, scores),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'threshold': .5,
    }


def metric_match(actual, expected, tol=1e-12):
    for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        if abs(float(actual[key]) - float(expected[key])) > tol:
            return False
    for key in ['tp', 'tn', 'fp', 'fn']:
        if int(actual[key]) != int(expected[key]):
            return False
    return True


def dist(values):
    xs = sorted(values)
    q = lambda p: xs[round((len(xs) - 1) * p)]
    return {
        'n': len(xs), 'mean': statistics.fmean(xs), 'median': statistics.median(xs),
        'min': min(xs), 'max': max(xs), 'p05': q(.05), 'p95': q(.95),
    }


def empirical_p(real, controls):
    return (1 + sum(x >= real for x in controls)) / (1 + len(controls))


def lift(y, baseline, candidate, ids):
    bm = metrics(y, baseline)
    cm = metrics(y, candidate)
    recovered = [iid for iid, t, b, c in zip(ids, y, baseline, candidate) if t == 1 and b < .5 and c >= .5]
    new_fp = [iid for iid, t, b, c in zip(ids, y, baseline, candidate) if t == 0 and b < .5 and c >= .5]
    lost_tp = [iid for iid, t, b, c in zip(ids, y, baseline, candidate) if t == 1 and b >= .5 and c < .5]
    removed_fp = [iid for iid, t, b, c in zip(ids, y, baseline, candidate) if t == 0 and b >= .5 and c < .5]
    return {
        'accuracy_delta': cm['accuracy'] - bm['accuracy'],
        'recall_delta': cm['recall'] - bm['recall'],
        'f1_delta': cm['f1'] - bm['f1'],
        'roc_auc_delta': cm['roc_auc'] - bm['roc_auc'],
        'baseline_false_negative_count': bm['fn'],
        'recovered_false_negative_count': len(recovered),
        'recovered_false_negative_ids': recovered,
        'lost_true_positive_count': len(lost_tp),
        'lost_true_positive_ids': lost_tp,
        'new_false_positive_count': len(new_fp),
        'new_false_positive_ids': new_fp,
        'removed_false_positive_count': len(removed_fp),
        'removed_false_positive_ids': removed_fp,
    }


def fusions(jarvisops, fly):
    return {
        'ENSEMBLE_75J_25F': [.75 * j + .25 * f for j, f in zip(jarvisops, fly)],
        'ENSEMBLE_50J_50F': [.50 * j + .50 * f for j, f in zip(jarvisops, fly)],
        'ENSEMBLE_RISK_OR': [max(j, f) for j, f in zip(jarvisops, fly)],
    }


def encoder_cells(brain):
    out = {}
    for name, (types, side) in CHANNEL_MAP.items():
        idx = brain.cells(types, side=side)
        if len(idx) == 0:
            raise RuntimeError(f'unresolved encoder mapping: {name}')
        out[name] = idx
    return out


def run_features(brain, cellmap, encoder_name):
    trace = Trace(brain, types=['descending_neuron'], tau=.1)
    output, runtimes = [], []
    for row in INC['rows']:
        t0 = time.perf_counter()
        brain.reset(SEED + sum(map(ord, row['incident_id'])))
        trace.reset()
        feat = None
        channels = encoder_channels(row)[encoder_name]
        for t in range(STEPS):
            inject = []
            if STIM_START <= t < STIM_END:
                for name, value in channels.items():
                    if value > 0:
                        inject.append((cellmap[name], np.float32(.80 * value)))
            feat = trace.observe(brain.step(inject=inject))
        output.append(np.asarray(feat, np.float32).ravel())
        runtimes.append((time.perf_counter() - t0) * 1000)
    return np.stack(output), {
        'feature_count': int(len(output[0])),
        'mean_incident_runtime_ms': float(np.mean(runtimes)),
    }


def rewire(brain, original_indices, original_weights, seed):
    rng = np.random.default_rng(seed)
    brain.indices = rng.permutation(original_indices).astype(original_indices.dtype)
    weights = original_weights.copy()
    if not brain.sensory_input and brain.superclass is not None:
        sensory = np.char.find(brain.superclass.astype(str), 'sensory') >= 0
        weights = np.where(sensory[brain.indices], 0, weights)
    brain.weights = weights.astype(original_weights.dtype)


def validate_inputs(ids, y, baseline_incident, baseline_full):
    assert len(ids) == INC['incident_count'] == REF['incident_count'] == 24
    assert sum(y) == INC['positive_count'] == 11
    assert ART['repair_artifact_count'] == REF['repair_artifact_count'] == 39
    assert set(ids) == set(ART_BY_ID) == set(REF_BY_ID)
    assert REF['formal_corpus_sha256'] == '97a9e0f7ed9a18edd1e7e0917d9b3081647bcf39659e912d88cc6488217b5450'
    mi = metrics(y, baseline_incident)
    mf = metrics(y, baseline_full)
    assert metric_match(mi, REF['incident_only_metrics'])
    assert metric_match(mf, REF['incident_plus_artifact_metrics'])
    return mi, mf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rewires', type=int, default=50)
    args = parser.parse_args()
    seeds = REWIRE_SEEDS[:args.rewires]
    ids = [r['incident_id'] for r in INC['rows']]
    y = [int(r['false_pass']) for r in INC['rows']]
    baseline_incident = [float(REF['incident_only_scores'][iid]) for iid in ids]
    baseline_full = [float(REF['incident_plus_artifact_scores'][iid]) for iid in ids]
    metrics_incident, metrics_full = validate_inputs(ids, y, baseline_incident, baseline_full)

    result = {
        'benchmark_id': 'JARVISOPS-FLY-FORMAL-P0-CORPUS-V0.4-LIVE',
        'scope': 'CURRENT_FULL_FORMAL_P0_SUPERVISED_CORPUS',
        'formal_corpus_sha256': REF['formal_corpus_sha256'],
        'incident_count': len(ids),
        'repair_artifact_count': ART['repair_artifact_count'],
        'encoder_count': len(ENCODER_NAMES),
        'rewire_control_count': len(seeds),
        'target_policy': 'False-PASS labels come only from formal P0 Incident Register; repair artifacts never create or override labels.',
        'baseline_provenance': 'Canonical locally-computed out-of-fold reference scores, SHA-bound and revalidated against labels in this run.',
        'input_sha256': {
            'benchmark_data.json': file_sha(BASE / 'benchmark_data.json'),
            'artifact_aggregate_v04.json': file_sha(BASE / 'artifact_aggregate_v04.json'),
            'baseline_reference_v04.json': file_sha(BASE / 'baseline_reference_v04.json'),
            'jarvisops_v04_benchmark_runner.py': file_sha(BASE / 'jarvisops_v04_benchmark_runner.py'),
        },
        'results': {
            'JARVISOPS_INCIDENT_ONLY': {'metrics': metrics_incident, 'scores': dict(zip(ids, baseline_incident))},
            'JARVISOPS_INCIDENT_PLUS_ARTIFACT': {'metrics': metrics_full, 'scores': dict(zip(ids, baseline_full))},
        },
        'encoders': {},
        'runtime': {
            'python': os.sys.version,
            'flybrain_version': '0.1.0',
            'fly_data': str(FLY_DATA),
            'numba_threads': os.environ.get('NUMBA_NUM_THREADS'),
        },
    }

    brain = FlyBrain(data=FLY_DATA, seed=SEED, device='cpu', sensory_input=False)
    original_indices = brain.indices.copy()
    original_weights = brain.weights.copy()
    cellmap = encoder_cells(brain)
    started = time.time()

    for encoder_name in ENCODER_NAMES:
        brain.indices = original_indices.copy()
        brain.weights = original_weights.copy()
        X, runtime_meta = run_features(brain, cellmap, encoder_name)
        fly_scores = reservoir_loocv(X, y)
        result['encoders'][encoder_name] = {
            'real_fly': {'metrics': metrics(y, fly_scores), 'runtime': runtime_meta, 'scores': dict(zip(ids, fly_scores))},
            'ensembles': {}, 'rewired_controls': {},
        }
        for ensemble_name, scores in fusions(baseline_full, fly_scores).items():
            result['encoders'][encoder_name]['ensembles'][ensemble_name] = {
                'metrics': metrics(y, scores),
                'lift_vs_jarvisops_full': lift(y, baseline_full, scores, ids),
                'scores': dict(zip(ids, scores)),
            }

    controls = {
        e: {
            'fly_auc': [], 'fly_accuracy': [],
            'ensemble': {n: {'auc': [], 'recall': [], 'f1': []} for n in ['ENSEMBLE_75J_25F', 'ENSEMBLE_50J_50F', 'ENSEMBLE_RISK_OR']},
        } for e in ENCODER_NAMES
    }

    for seed in seeds:
        rewire(brain, original_indices, original_weights, seed)
        for encoder_name in ENCODER_NAMES:
            X, _ = run_features(brain, cellmap, encoder_name)
            fly_scores = reservoir_loocv(X, y)
            fm = metrics(y, fly_scores)
            controls[encoder_name]['fly_auc'].append(fm['roc_auc'])
            controls[encoder_name]['fly_accuracy'].append(fm['accuracy'])
            for ensemble_name, scores in fusions(baseline_full, fly_scores).items():
                em = metrics(y, scores)
                controls[encoder_name]['ensemble'][ensemble_name]['auc'].append(em['roc_auc'])
                controls[encoder_name]['ensemble'][ensemble_name]['recall'].append(em['recall'])
                controls[encoder_name]['ensemble'][ensemble_name]['f1'].append(em['f1'])

    for encoder_name in ENCODER_NAMES:
        entry = result['encoders'][encoder_name]
        control = controls[encoder_name]
        real_metrics = entry['real_fly']['metrics']
        entry['rewired_controls'] = {
            'seeds': seeds,
            'fly_auc_distribution': dist(control['fly_auc']),
            'fly_accuracy_distribution': dist(control['fly_accuracy']),
            'real_fly_auc_empirical_p_ge': empirical_p(real_metrics['roc_auc'], control['fly_auc']),
            'real_fly_auc_percentile': sum(x < real_metrics['roc_auc'] for x in control['fly_auc']) / len(control['fly_auc']),
            'ensemble_distributions': {},
        }
        for ensemble_name, values in control['ensemble'].items():
            real_ensemble = entry['ensembles'][ensemble_name]['metrics']
            entry['rewired_controls']['ensemble_distributions'][ensemble_name] = {
                'auc': dist(values['auc']),
                'recall': dist(values['recall']),
                'f1': dist(values['f1']),
                'real_ensemble_auc_empirical_p_ge': empirical_p(real_ensemble['roc_auc'], values['auc']),
            }

    candidates = []
    for encoder_name, entry in result['encoders'].items():
        for ensemble_name, e in entry['ensembles'].items():
            candidates.append((e['metrics']['f1'], e['metrics']['recall'], e['metrics']['roc_auc'], encoder_name, ensemble_name, e))
    candidates.sort(reverse=True)
    best = candidates[0]
    result['best_observed_exploratory'] = {
        'encoder': best[3], 'ensemble': best[4],
        'metrics': best[5]['metrics'],
        'lift_vs_jarvisops_full': best[5]['lift_vs_jarvisops_full'],
        'selection_warning': 'Selected on the same 24-incident corpus. Exploratory only; not an independent validation result.',
    }
    result['runtime']['total_runtime_s'] = time.time() - started
    result['evidence_status'] = 'EXPERIMENTAL_MODEL_SIGNAL'
    result['production_authority'] = False
    del brain
    gc.collect()
    raw = json.dumps(result, sort_keys=True, separators=(',', ':')).encode()
    result['report_sha256'] = sha256(raw).hexdigest()
    Path('benchmark_result_v04.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
