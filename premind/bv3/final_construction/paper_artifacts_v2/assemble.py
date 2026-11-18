"""Read-only CPU assembly; no generation backend is imported or invoked."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
FC = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(FC))
from eval_v1 import release_data as D, online_retrieval as OR, metrics_v4 as M4, acute_rule_v4 as AR
from trainer_v2 import reward_v5 as R5, metrics_v9 as MV

spec = importlib.util.spec_from_file_location('artifact_io_v1', FC / 'paper_artifacts_v1/io.py')
IO = importlib.util.module_from_spec(spec)
spec.loader.exec_module(IO)

@lru_cache(maxsize=None)
def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def jl(path):
    return IO.read_jsonl(Path(path), final_holdout=True)

def release_rows(release, mode):
    path = Path(release) / ('' if mode == 'retrieved' else mode) / 'holdout.jsonl'
    return {r['sample_id']: r for r in jl(path)}, path

def fingerprint(row, release, k=8, online_set=None):
    """Media contents, exact strings and ordered context; labels deliberately excluded."""
    root = Path(release).resolve().parent
    payload = {'query_sha256': sha(str((root / row['query']).resolve())),
               'request': row['query_text'], 'k': k,
               'context': [{'sha256': sha(str((root / p).resolve())), 'text': t}
                           for p, t in zip(row['contexts'], row['context_texts'], strict=True)],
               'online_retrieved_set': online_set}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

class InputView:
    """Transform using frozen online evaluator, including relabel/exclusion rules."""
    def __init__(self, release, mode, k, online, query_emb=None):
        self.release, self.mode, self.k, self.online = Path(release), mode, k, online
        self.rows, self.path = release_rows(release, mode)
        self.partners = OR.partner_index(list(self.rows.values())) if online else {}
        self.retr = OR.OnlineRetriever(Path(query_emb) if query_emb else None, release=self.release) if online else None
        self.cache = {}
    def get(self, rid):
        if rid not in self.cache:
            row = self.rows[rid]
            if self.online:
                row, info = OR.online_row(row, self.k, self.retr, self.partners)
            else:
                info = {}
            if row is None:
                self.cache[rid] = (None, None, info, None)
            else:
                fp = fingerprint(row, self.release, self.k, info.get('online_topk'))
                self.cache[rid] = (row, fp, info, None)
        return self.cache[rid]
    def item(self, rid, arm):
        row = self.get(rid)[0]
        return D.convert(row, self.release.resolve().parent, (arm,), final_holdout=True)[0]


def verified_source(meta, mode):
    """Only an exact source-file hash can vouch for source row inputs."""
    block = meta['data'][mode]
    original = Path(meta['release'])
    candidates = [original, *sorted(original.parent.glob(original.name + '*'))]
    for release in candidates:
        path = release / ('' if mode == 'retrieved' else mode) / 'holdout.jsonl'
        if path.is_file() and sha(str(path.resolve())) == block['sha256']:
            return release
    raise ValueError(f"unverifiable_source_release_hash:{original}:{mode}")


def choose_rows(target_ids, candidates, target_fps, dropped=()):
    """Pure fixture-testable precedence + fingerprint gate. Never fallback past stale newest."""
    chosen, stale, missing = {}, {}, []
    dropped = set(dropped)
    for rid in target_ids:
        if rid in dropped:
            continue
        options = candidates.get(rid, [])
        if not options:
            missing.append(rid); continue
        newest = max(options, key=lambda c: c['priority'])
        if newest.get('fingerprint') != target_fps.get(rid):
            stale[rid] = newest
        else:
            chosen[rid] = newest
    return chosen, stale, missing


def join_target_id(source_row, target_by_evaluator):
    """Conditions/releases may use different sample_id aliases for one evaluator row."""
    return target_by_evaluator.get(source_row['evaluator']['row_id'])


def rescore(item, text, original, meta, mode):
    """Re-score every reused answer, not just explicit rescore lists, via frozen scorer."""
    preset = meta.get('reward_preset', 'main')
    flags = R5.PRESETS[preset]
    if flags.vef or flags.withhold_match:
        raise ValueError('VEF/Match preset requires supported scorer inputs; refuse instead of silent fallback')
    count = text.get('n_stage2_tokens') or original.get('stage2_gen_tokens') or 0
    if item.arm == 'passive':
        ledger = R5.reward_passive(item.target, text.get('stage2') or '', count, flags=flags)
    else:
        ledger = R5.reward_call(item.target, text.get('stage1') or '', text.get('stage2') or '', count,
                                r=meta.get('r', .1), flags=flags)
    rec = {**original, **MV.episode_record(item, ledger, mode=mode, kind='greedy', sample_idx=0),
           'family_id': item.row['family_id'], 'action_pred': (ledger.get('parse') or {}).get('action_pred')}
    return rec


def assemble(setting, registry, views):
    mode, arm, k = setting['mode'], setting['arm'], setting['k']
    online = setting['online']
    key = (registry['release'], mode, k, online, registry.get('query_emb'))
    if key not in views: views[key] = InputView(*key[:4], query_emb=key[4])
    target = views[key]
    fps, excluded = {}, {}
    target_by_evaluator = {r['evaluator']['row_id']:rid for rid,r in target.rows.items()}
    if len(target_by_evaluator) != len(target.rows): raise ValueError('duplicate evaluator.row_id in target mode')
    for rid in target.rows:
        row, fp, info, _ = target.get(rid)
        if row is None: excluded[rid] = info.get('topk_exclusion', 'not_evaluable')
        else: fps[rid] = fp
    candidates, issues, sources = {}, [], []
    identity = None
    for priority, runname in enumerate(setting.get('runs', [])):
        run = FC / runname
        if not (run / 'eval_report.json').is_file():
            issues.append({'run': str(run), 'reason': 'pending_run'}); continue
        meta = IO.read_json(run / 'eval_report.json', final_holdout=True)['meta']
        block = meta['data'].get(mode)
        if block is None: raise ValueError(f'{run}: requested mode absent')
        run_k = meta.get('online_k') or block.get('online', {}).get('online_k') or meta.get('context_k') or 8
        if run_k != k or bool(block.get('online')) != online:
            raise ValueError(f'{run}: setting K/online mismatch')
        pin = meta.get('model_pin', [meta.get('model')])
        current_identity = (json.dumps(pin, sort_keys=True), json.dumps(meta.get('lora'), sort_keys=True))
        if identity is not None and identity != current_identity:
            raise ValueError(f'{run}: model/checkpoint identity differs across deltas')
        identity = current_identity
        if setting.get('zero_shot') and meta.get('lora') is not None:
            raise ValueError(f'{run}: checkpoint cannot populate a zero-shot slot')
        if setting.get('expected_model') and meta.get('model') != setting['expected_model']:
            raise ValueError(f'{run}: wrong model {meta.get("model")}')
        release = verified_source(meta, mode)
        qe = (block.get('online', {}).get('fresh') or {}).get('dir')
        skey = (str(release), mode, k, online, qe)
        if skey not in views: views[skey] = InputView(*skey[:4], query_emb=qe)
        src = views[skey]
        recs = [r for r in jl(run / 'records.jsonl') if (r['arm'], r['mode'], r['kind'], r.get('sample_idx', 0)) == (arm,mode,'greedy',0)]
        texts = {r['row_id']:r for r in jl(run / 'texts.jsonl') if (r['arm'],r['mode'],r['kind'],r.get('sample_idx',0)) == (arm,mode,'greedy',0)}
        if len({r['row_id'] for r in recs}) != len(recs): raise ValueError(f'{run}: duplicate records')
        seen_online = block.get('online', {}).get('rows', {})
        for rec in recs:
            source_id = rec['row_id']
            if source_id not in src.rows: raise ValueError(f'{run}: source row absent {source_id}')
            rid = join_target_id(src.rows[source_id], target_by_evaluator)
            if rid is None: continue
            if source_id not in texts: raise ValueError(f'{run}: missing answer {source_id}')
            sr, fp, info, _ = src.get(source_id)
            if sr is None: raise ValueError(f'{run}: excluded source record {rid}')
            if online and info.get('online_topk') != seen_online.get(source_id, {}).get('online_topk'):
                raise ValueError(f'{run}: recomputed source ranking differs from run manifest {rid}')
            candidates.setdefault(rid, []).append({'priority':priority, 'run':str(run), 'source_release':str(release),
                'fingerprint':fp, 'model_pin':pin, 'source_sample_id':source_id, 'target_sample_id':rid, 'record':rec, 'text':{**texts[source_id], 'row_id':rid}, 'meta':meta})
        sources += [run / 'eval_report.json', run / 'records.jsonl', run / 'texts.jsonl', src.path]
    chosen, stale, missing = choose_rows(target.rows, candidates, fps)
    records, texts, targets, provenance = [], [], {}, {}
    for rid, c in chosen.items():
        if rid in excluded: continue
        item = target.item(rid, arm)
        r = rescore(item, c['text'], c['record'], c['meta'], mode)
        r.update(target.get(rid)[2])
        records.append(r); texts.append(c['text'])
        targets.update(M4.targets_from_items([item]))
        provenance[rid] = {q:c[q] for q in ('run','source_release','fingerprint','model_pin','source_sample_id','target_sample_id')}
        provenance[rid]['rescored_on'] = registry['release']
    records = AR.attach_hazard(M4.attach_risk_pred(records, texts), texts, targets)
    # Evaluator exclusions are explicit, not counted as missing model runs.
    stale = {rid:{k:v for k,v in c.items() if k not in ('record','text','meta')} | {'target_fingerprint':fps.get(rid)}
             for rid,c in stale.items() if rid not in excluded}
    missing = [rid for rid in missing if rid not in excluded]
    status = {'id':setting['id'], 'n_release':len(target.rows), 'n_assembled':len(records),
              'n_evaluable':len(target.rows)-len(excluded), 'complete_evaluable':not stale and not missing,
              'full_population':len(records)==len(target.rows), 'stale':stale, 'missing':missing,
              'evaluator_excluded':excluded, 'run_issues':issues, 'provenance':provenance}
    return status, records, texts, targets, target.rows, sources
