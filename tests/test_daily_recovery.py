import copy
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load('daily_recovery_runner', 'scripts/run_daily_recommendations.py')
fetcher = load('daily_recovery_fetcher', 'src/maintain/fetchers/fetch_arxiv.py')


@pytest.fixture
def config():
    return {
        'arxiv_paper_setting': {'days_window': 9, 'prefer_supabase_read': True},
        'supabase': {'enabled': True, 'url': 'https://example.invalid', 'anon_key': 'test'},
        'subscriptions': {'intent_profiles': [{
            'id': 'p1', 'tag': 'test', 'enabled': True, 'paper_sources': ['arxiv'],
            'keywords': [{'keyword': 'retrieval', 'query': 'retrieval models', 'enabled': True}],
        }]},
    }


@pytest.mark.parametrize('status,body,available', [(200, [], True), (503, {}, False)])
def test_database_health_distinguishes_empty_from_outage(config, status, body, available):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(body).encode()
    assert runner.database_available(config, get=Mock(return_value=response)) is available


def test_dns_failure_falls_back_but_auth_error_is_not_hidden(config):
    assert not runner.database_available(config, get=Mock(side_effect=requests.ConnectionError('DNS')))
    response = requests.Response()
    response.status_code = 401
    with pytest.raises(requests.HTTPError):
        runner.database_available(config, get=Mock(return_value=response))


@pytest.mark.parametrize('days', [1, 10])
def test_fallback_preserves_subscriptions_and_applies_window(config, days):
    original = copy.deepcopy(config)
    snapshot = runner.prepare_config(config, days, True)
    assert config == original
    assert snapshot['subscriptions'] == original['subscriptions']
    assert snapshot['arxiv_paper_setting']['days_window'] == days
    assert not snapshot['supabase']['enabled']
    assert not snapshot['source_backends']['arxiv']['enabled']


def test_fallback_refuses_to_drop_other_sources(config):
    config['subscriptions']['intent_profiles'][0]['paper_sources'] = ['arxiv', 'biorxiv']
    with pytest.raises(RuntimeError, match='no sources'):
        runner.prepare_config(config, 10, True)


def test_runner_passes_isolated_config_and_propagates_failure(config, tmp_path, monkeypatch):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    original = path.read_bytes()
    monkeypatch.setenv('DPR_CONFIG_FILE', str(path))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    snapshots = []

    def run(command, env):
        snapshot = Path(env['DPR_CONFIG_FILE'])
        snapshots.append(snapshot)
        assert snapshot != path
        assert yaml.safe_load(snapshot.read_text())['arxiv_paper_setting']['days_window'] == 10
        assert '--no-skip-fetch' in command and '--fetch-ignore-seen' in command
        assert env['DPR_STRICT_RETRIEVAL'] == env['DPR_STRICT_ARXIV_FETCH'] == '1'
        return 7

    monkeypatch.setattr(runner.subprocess, 'call', run)
    assert runner.main(['--retrieval-source', 'arxiv', '--fetch-days', '10']) == 7
    assert path.read_bytes() == original
    assert not snapshots[0].exists()


@pytest.mark.parametrize('lane', ['bm25', 'embedding'])
@pytest.mark.parametrize('message,fails', [('rpc 查询异常: DNS', True), ('rpc 分片查询成功 | partial_failures=1', True), ('rpc 分片查询成功: 0 rows', False)])
def test_retrieval_failure_cannot_become_empty_success(lane, message, fails, monkeypatch):
    filename = '2.1.retrieval_papers_bm25.py' if lane == 'bm25' else '2.2.retrieval_papers_embedding.py'
    module = load('daily_recovery_' + lane, 'src/' + filename)
    monkeypatch.setenv('DPR_STRICT_RETRIEVAL', '1')
    rpc = 'query_supabase_bm25_with_shards' if lane == 'bm25' else 'query_supabase_vector_with_shards'
    monkeypatch.setattr(module, rpc, Mock(return_value=([], message)))
    kwargs = dict(queries=[{'query_text': 'retrieval', 'keyword': 'retrieval', 'query_embedding': [1.0, 0.0]}], top_k=10,
                  supabase_conf={'url': 'https://example.invalid', 'anon_key': 'test'})
    if lane == 'embedding':
        kwargs.update(model=None, rpc_mode='exact')
    if fails:
        with pytest.raises(RuntimeError, match='refusing to publish'):
            module.rank_papers_for_queries_via_supabase(**kwargs)
    else:
        result = module.rank_papers_for_queries_via_supabase(**kwargs)
        assert result['total_hits'] == 0


def test_arxiv_failure_does_not_write_results_or_advance_state(tmp_path, monkeypatch):
    monkeypatch.setenv('DPR_STRICT_ARXIV_FETCH', '1')
    monkeypatch.setattr(fetcher, 'load_config', lambda: {})
    monkeypatch.setattr(fetcher, 'CATEGORIES_TO_FETCH', ['cs'])
    monkeypatch.setattr(fetcher, 'load_seen_state', lambda: (set(), None))
    monkeypatch.setattr(fetcher, 'load_last_crawl_at', lambda: None)
    client = Mock()
    client.results.side_effect = RuntimeError('arXiv unavailable')
    monkeypatch.setattr(fetcher, 'ResilientArxivClient', lambda **kwargs: client)
    save_seen, save_crawl = Mock(), Mock()
    monkeypatch.setattr(fetcher, 'save_seen_state', save_seen)
    monkeypatch.setattr(fetcher, 'save_last_crawl_at', save_crawl)
    output = tmp_path / 'papers.json'
    with pytest.raises(RuntimeError, match='arXiv fetch failed'):
        fetcher.fetch_all_domains_metadata_robust(days=1, output_file=str(output), ignore_seen=True, disable_supabase_read=True)
    assert not output.exists()
    save_seen.assert_not_called()
    save_crawl.assert_not_called()


@pytest.mark.parametrize('token,days,expected_start,expected_end', [
    ('20261103', 1, '2026-11-03', '2026-11-04'),
    ('20261024-20261102', 10, '2026-10-24', '2026-11-03'),
])
def test_daily_and_review_windows(token, days, expected_start, expected_end, monkeypatch):
    monkeypatch.setenv('DPR_RUN_DATE', token)
    start, end, _ = fetcher.resolve_supabase_time_window(end_date=datetime(2026, 11, 3, 2, tzinfo=timezone.utc), days=days)
    assert start.isoformat().startswith(expected_start)
    assert end.isoformat().startswith(expected_end)
