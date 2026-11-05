import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import arxiv
import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import arxiv_fetch_recovery as recovery

spec = importlib.util.spec_from_file_location('fetch_recovery_test', ROOT / 'src/maintain/fetchers/fetch_arxiv.py')
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)
NOW = datetime(2026, 11, 5, 8, tzinfo=timezone.utc)


def response(status, retry_after=None):
    result = requests.Response()
    result.status_code = status
    result._content = b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    if retry_after is not None:
        result.headers['Retry-After'] = retry_after
    return result


def test_real_client_retries_same_page_and_honors_retry_after(monkeypatch):
    send = Mock(side_effect=[response(429, '75'), response(429), response(200)])
    monkeypatch.setattr(requests.adapters.HTTPAdapter, 'send', send)
    sleep = Mock()
    monkeypatch.setattr(recovery.time, 'sleep', sleep)
    client = recovery.ResilientArxivClient(log=lambda text: None)
    client.delay_seconds = 0  # No real waiting in tests.
    client._parse_feed('https://export.arxiv.org/api/query?start=200')
    assert [call.args[0] for call in sleep.call_args_list] == [75, 120]
    assert {call.args[0].url for call in send.call_args_list} == {'https://export.arxiv.org/api/query?start=200'}
    assert all(call.kwargs['timeout'] == (10, 60) for call in send.call_args_list)


@pytest.mark.parametrize('status,retries', [(429, 5), (503, 5), (403, 1)])
def test_retries_are_bounded_and_auth_errors_fail_fast(monkeypatch, status, retries):
    send = Mock(side_effect=lambda *args, **kwargs: response(status))
    monkeypatch.setattr(requests.adapters.HTTPAdapter, 'send', send)
    sleep = Mock()
    monkeypatch.setattr(recovery.time, 'sleep', sleep)
    client = recovery.ResilientArxivClient(log=lambda text: None)
    client.delay_seconds = 0
    with pytest.raises(arxiv.HTTPError):
        client._parse_feed('https://export.arxiv.org/api/query')
    assert send.call_count == retries
    if status == 429:
        assert [call.args[0] for call in sleep.call_args_list] == [60, 120, 240, 480]
        with pytest.raises(arxiv.HTTPError):
            client._parse_feed('https://export.arxiv.org/api/query?next')
        assert sleep.call_count == 4  # Budget is global, not reset per page.


def test_server_wait_exceeding_budget_is_not_shortened(monkeypatch):
    send = Mock(return_value=response(429, '3600'))
    monkeypatch.setattr(requests.adapters.HTTPAdapter, 'send', send)
    sleep = Mock()
    monkeypatch.setattr(recovery.time, 'sleep', sleep)
    client = recovery.ResilientArxivClient(log=lambda text: None)
    with pytest.raises(arxiv.HTTPError):
        client._parse_feed('https://export.arxiv.org/api/query')
    assert send.call_count == 1
    sleep.assert_not_called()


def test_rate_limit_never_splits_the_time_window(monkeypatch):
    client = Mock()
    client.results.side_effect = arxiv.HTTPError('https://example.invalid', 0, 429)
    monkeypatch.setenv('DPR_STRICT_ARXIV_FETCH', '1')
    with pytest.raises(RuntimeError, match='rate limit persisted'):
        fetcher.fetch_category_in_windows(client, 'eess', [(NOW - timedelta(days=9), NOW)], set(), {})
    assert client.results.call_count == 1


def paper(pid):
    return SimpleNamespace(get_short_id=lambda: pid, title='A paper', summary='Abstract',
                           authors=[], primary_category='cs.IR', categories=['cs.IR'],
                           published=NOW - timedelta(hours=1), pdf_url='https://arxiv.org/pdf/' + pid)


def test_failed_fetch_resumes_completed_windows_without_publishing_partial_data(tmp_path, monkeypatch):
    monkeypatch.setenv('DPR_STRICT_ARXIV_FETCH', '1')
    monkeypatch.setenv('DPR_ARXIV_FETCH_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setenv('GITHUB_RUN_ID', 'run-123')
    monkeypatch.setenv('DPR_RUN_DATE', '20261105')
    monkeypatch.setattr(fetcher, 'load_config', lambda: {})
    monkeypatch.setattr(fetcher, 'CATEGORIES_TO_FETCH', ['cs', 'eess'])
    save_seen, save_crawl = Mock(), Mock()
    monkeypatch.setattr(fetcher, 'save_seen_state', save_seen)
    monkeypatch.setattr(fetcher, 'save_last_crawl_at', save_crawl)
    client = Mock()
    # One complete category, then a partial page followed by a limit.
    def incomplete():
        yield paper('2611.00002v1')
        raise arxiv.HTTPError('https://example.invalid', 0, 429)
    client.results.side_effect = [iter([paper('2611.00001v1')]), incomplete()]
    monkeypatch.setattr(fetcher, 'ResilientArxivClient', lambda **kw: client)
    output = tmp_path / 'papers.json'
    kwargs = dict(days=2, output_file=str(output), ignore_seen=True, disable_supabase_read=True)
    with pytest.raises(RuntimeError, match='rate limit persisted'):
        fetcher.fetch_all_domains_metadata_robust(**kwargs)
    assert not output.exists()
    save_seen.assert_not_called()
    save_crawl.assert_not_called()
    state = json.loads(next((tmp_path / 'cache').glob('*.json')).read_text())
    assert len(state['windows']) == 1 and not state['complete']
    saved_end = datetime.fromisoformat(state['end'])
    # A rerun must use the exact original bound and fetch only the failed category.
    client.results.reset_mock(side_effect=True)
    client.results.return_value = iter([paper('2611.00001v1'), paper('2611.00002v1')])
    fetcher.fetch_all_domains_metadata_robust(**kwargs)
    assert client.results.call_count == 1
    assert 'cat:eess*' in client.results.call_args.args[0].query
    assert {p['id'] for p in json.loads(output.read_text())} == {'2611.00001v1', '2611.00002v1'}
    assert save_crawl.call_args.args[0] == saved_end
    assert json.loads(next((tmp_path / 'cache').glob('*.json')).read_text())['complete']


def test_checkpoints_do_not_leak_between_runs_windows_or_completed_fetches(tmp_path):
    start = NOW - timedelta(days=10)
    checkpoint = recovery.FetchCheckpoint(tmp_path, ['run1', 10], start, NOW)
    checkpoint.put('cs', [{'id': 'paper1'}])
    resumed = recovery.FetchCheckpoint(tmp_path, ['run1', 10], start, NOW + timedelta(hours=1))
    assert resumed.bounds == (start, NOW) and resumed.get('cs')
    for scope in (['run2', 10], ['run1', 9]):
        fresh = recovery.FetchCheckpoint(tmp_path, scope, start, NOW + timedelta(hours=1))
        assert fresh.get('cs') is None
    resumed.finish()
    complete = recovery.FetchCheckpoint(tmp_path, ['run1', 10], start, NOW + timedelta(hours=1))
    assert complete.get('cs') is None
    assert complete.bounds[1] == NOW + timedelta(hours=1)


def test_empty_completed_window_is_cached_but_expired_window_is_not(tmp_path, monkeypatch):
    checkpoint = recovery.FetchCheckpoint(tmp_path, ['run1'], NOW, NOW)
    checkpoint.put('empty', [])
    assert recovery.FetchCheckpoint(tmp_path, ['run1'], NOW, NOW).get('empty') == []
    original_time = recovery.time.time()
    monkeypatch.setattr(recovery.time, 'time', lambda: original_time + 86401)
    assert recovery.FetchCheckpoint(tmp_path, ['run1'], NOW, NOW).get('empty') is None
