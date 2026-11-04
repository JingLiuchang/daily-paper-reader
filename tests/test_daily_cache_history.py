import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from daily_history import canonical_id, filter_file, published_ids
from daily_model_cache import ModelCache, cached_rerank
from daily_report_state import merge_daily_state, load_daily_state
from filter import compute_embeddings


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_versions_merge_preserves_routes_and_actual_summary():
    old = dict(paper_id='2610.37176v1', route='202611/03/2610.37176v1-old', title='old', section='deep')
    new = dict(paper_id='2610.37176v2', route='202611/03/2610.37176v2-new', title='new', section='quick')
    state = merge_daily_state({}, '20261103', [old], '2026-11-03', True)
    state = merge_daily_state(state, '20261103', [new], '2026-11-04', True)
    assert len(state['papers']) == 1
    paper = state['papers'][0]
    assert paper['route'] == old['route'] and paper['title'] == 'old'
    assert set(paper['version_routes']) == {old['route'], new['route']}


def test_same_day_history_uses_cumulative_state_even_if_archive_overwritten(tmp_path):
    day = tmp_path / 'docs/202611/03'
    day.mkdir(parents=True)
    (day / '_daily_state.json').write_text(json.dumps({'papers': [{'paper_id':'2610.37176v1'}]}))
    raw = tmp_path / 'raw.json'
    raw.write_text(json.dumps([{'id':'2610.37176v2'}, {'id':'2611.12345v1'}]))
    filter_file(raw, tmp_path)
    assert json.loads(raw.read_text()) == [{'id':'2611.12345v1'}]
    ranked = tmp_path / 'ranked.json'
    ranked.write_text(json.dumps({'papers':[{'id':'2610.37176v2'}], 'queries':[{'ranked':[{'paper_id':'2610.37176v2'}]}]}))
    filter_file(ranked, tmp_path)
    assert json.loads(ranked.read_text())['queries'][0]['ranked'] == []
    assert canonical_id('https://arxiv.org/pdf/2610.37176v2.pdf') in published_ids(tmp_path)


def test_vectors_persist_across_instances_and_invalidate_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv('DPR_MODEL_CACHE_DIR', str(tmp_path))
    class Model:
        model_name = 'm'
        endpoint = 'https://example.org/embed'
        calls = []
        def encode(self, texts, **kwargs):
            self.calls.append(texts)
            return np.array([[len(t), 1.] for t in texts])
    def papers(*texts):
        return [SimpleNamespace(text_for_embedding=t) for t in texts]
    model = Model()
    first = compute_embeddings(model, papers('a', 'b'), batch_size=1)
    np.testing.assert_array_equal(compute_embeddings(Model(), papers('a', 'b')), first)
    assert len(model.calls) == 2
    compute_embeddings(model, papers('a', 'changed'))
    assert model.calls[-1] == ['changed']
    model.endpoint = 'https://another.org/embed'
    compute_embeddings(model, papers('a'))
    assert len(model.calls) == 4
    model.model_name = 'new-model'
    compute_embeddings(model, papers('a'))
    assert len(model.calls) == 5


def test_failed_embeddings_are_not_cached(tmp_path, monkeypatch):
    monkeypatch.setenv('DPR_MODEL_CACHE_DIR', str(tmp_path))
    class Model:
        calls = 0
        def encode(self, texts, **kwargs):
            self.calls += 1
            return np.array([[float('nan')]])
    m = Model()
    for _ in range(2):
        with pytest.raises(ValueError):
            compute_embeddings(m, [SimpleNamespace(text_for_embedding='x')])
    assert m.calls == 2


def test_rerank_exact_batches_and_invalid_responses(tmp_path, monkeypatch):
    monkeypatch.setenv('DPR_MODEL_CACHE_DIR', str(tmp_path))
    class Ranker:
        base_url = 'https://example.org/rerank'
        calls = 0
        broken = False
        def rerank(self, **kwargs):
            self.calls += 1
            return {'results': [] if self.broken else [{'index':i,'relevance_score':i/10} for i in range(len(kwargs['documents']))]}
    r = Ranker()
    response = cached_rerank(r, 'q', ['a', 'b'], 'm')
    assert cached_rerank(r, 'q', ['a', 'b'], 'm') == response and r.calls == 1
    cached_rerank(r, 'changed', ['a', 'b'], 'm')
    assert r.calls == 2
    r.broken = True
    for _ in range(2):
        with pytest.raises(ValueError):
            cached_rerank(r, 'q', ['new'], 'm')
    assert r.calls == 4


def test_score_cache_reuses_valid_results_and_invalidates_requirements(tmp_path, monkeypatch):
    monkeypatch.setenv('DPR_MODEL_CACHE_DIR', str(tmp_path))
    mod = load('score_cache_test', 'src/4.llm_refine_papers.py')
    calls = []
    monkeypatch.setattr(mod, '_make_filter_client', lambda *a: None)
    def make_runner(*args, **kwargs):
        def run(docs, attempt, retry):
            calls.extend(d['id'] for d in docs)
            return [dict(id=d['id'], score=0, matched_requirement_index=0) for d in docs]
        return run
    monkeypatch.setattr(mod, '_make_filter_runner', make_runner)
    docs = [{'id':'2610.12345v1', 'content':'abstract'}]
    a = mod._filter_batch(1, docs, 'unused', [], 'model', 500, str(tmp_path))
    assert mod._filter_batch(2, docs, 'unused', [], 'model', 500, str(tmp_path))[2] == a[2]
    assert len(calls) == 1
    mod._filter_batch(1, docs, 'unused', [{'query':'changed'}], 'model', 500, str(tmp_path))
    assert len(calls) == 2
    mod._filter_batch(1, [{'id':'2610.12345v2','content':'new abstract'}], 'unused', [], 'model', 500, str(tmp_path))
    assert len(calls) == 3


def test_ten_day_pipeline_does_not_filter_history(monkeypatch):
    mod = load('main_cache_test', 'src/main.py')
    monkeypatch.setattr(mod, 'run_step', lambda *a, **kw: None)
    monkeypatch.setattr(mod, 'should_skip_fetch', lambda: False)
    monkeypatch.setattr('daily_history.filter_file', lambda *a: pytest.fail('review must not exclude published papers'))
    monkeypatch.setattr(sys, 'argv', ['main.py', '--fetch-days', '10', '--fetch-mode', 'standard'])
    mod.main()


def test_offline_reconciliation_preserves_files_and_run_history(tmp_path):
    docs = tmp_path / 'docs'
    day = docs / '202611/03'
    day.mkdir(parents=True)
    papers = []
    for v in (1, 2):
        route = f'202611/03/2610.12345v{v}-paper'
        papers.append(dict(paper_id=f'2610.12345v{v}', route=route, title='Paper', section='deep'))
        (docs / (route+'.md')).write_text('# Paper\n\n## 论文详细总结（自动生成）\nExisting summary')
        (docs / (route+'.txt')).write_text('Original full text; References')
    (day / '_daily_state.json').write_text(json.dumps(dict(date='20261103', papers=papers,
        run_count=3, generated_at='2026-11-04 01:08:46 UTC', recommend_exists=True)))
    (docs / 'README.md').write_text('2026-11-03 2026-11-04 01:08:46 UTC')
    (docs / '_sidebar.md').write_text('* Daily Papers\n')
    original = {p.relative_to(docs):p.read_bytes() for p in docs.rglob('*') if p.is_file()}
    mod = load('reconcile_test', 'scripts/reconcile_daily_versions.py')
    changes = mod.reconcile(docs)
    assert changes == [{'date':'20261103','before':2,'after':1}]
    assert mod.reconcile(docs) == []
    state = load_daily_state(str(docs / '202611/03/_daily_state.json'))
    assert state['run_count'] == 3 and state['generated_at'] == '2026-11-04 01:08:46 UTC'
    for relative, contents in original.items():
        if relative.name not in {'README.md','_daily_state.json','_sidebar.md','papers.meta.json'}:
            assert (docs / relative).read_bytes() == contents
    assert '共 1 篇' in (docs / 'README.md').read_text()
