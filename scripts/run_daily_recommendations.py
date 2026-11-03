#!/usr/bin/env python3
"""Use the existing direct arXiv pipeline when its shared database is down.

Only a per-run configuration is changed. Subscription configuration and published
results remain owned by the normal pipeline. This is for the 1–30 day workflow;
long-range research keeps its existing database-only implementation.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from source_config import get_source_backend, load_config_with_source_migration
from source_backend_router import group_queries_by_source
from subscription_plan import build_pipeline_inputs


def database_available(config, get=requests.get):
    backend = get_source_backend(config, 'arxiv')
    if not backend.get('enabled'):
        return False
    url = str(backend.get('url') or '').rstrip('/')
    key = str(backend.get('anon_key') or '')
    if not url or not key:
        raise RuntimeError('arXiv database URL or public key is missing')
    try:
        response = get(
            f"{url}/rest/v1/{backend['papers_table']}",
            params={'select': 'id', 'limit': 1},
            headers={'apikey': key, 'Authorization': f'Bearer {key}',
                     'Accept-Profile': backend.get('schema', 'public')},
            timeout=(5, 15),
        )
    except (requests.ConnectionError, requests.Timeout):
        return False
    if response.status_code >= 500:
        return False
    # A permission/configuration error must not silently change the data source.
    response.raise_for_status()
    if not isinstance(response.json(), list):
        raise RuntimeError('arXiv database returned an invalid response')
    return True


def prepare_config(config, days, direct):
    result = copy.deepcopy(config)
    if days is not None:
        result.setdefault('arxiv_paper_setting', {})['days_window'] = days
    if direct:
        inputs = build_pipeline_inputs(result)
        sources = set(group_queries_by_source(
            inputs.get('bm25_queries', []) + inputs.get('embedding_queries', [])))
        if not sources or sources != {'arxiv'}:
            raise RuntimeError('Direct recovery requires active arXiv-only subscriptions; no sources will be silently dropped')
        result.setdefault('arxiv_paper_setting', {})['prefer_supabase_read'] = False
        result.setdefault('supabase', {})['enabled'] = False
        result.setdefault('source_backends', {}).setdefault('arxiv', {})['enabled'] = False
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval-source', choices=['auto', 'supabase', 'arxiv'], default='auto')
    parser.add_argument('--check-only', action='store_true', help='Check routing without fetching papers or calling models')
    options, pipeline_args = parser.parse_known_args(argv)
    inputs = argparse.ArgumentParser(add_help=False)
    inputs.add_argument('--fetch-days', type=int)
    inputs.add_argument('--profile-tag', default='')
    known, _ = inputs.parse_known_args(pipeline_args)
    if known.fetch_days is not None and not 1 <= known.fetch_days <= 365:
        parser.error('--fetch-days must be between 1 and 365')
    if known.fetch_days and known.fetch_days > 30:
        if options.retrieval_source == 'arxiv':
            parser.error('Direct recovery supports daily/1–30 day runs only')
        if options.check_only:
            parser.error('Use the research workflow checks for windows over 30 days')
        return subprocess.call([sys.executable, str(ROOT / 'src/main.py'), *pipeline_args])
    if known.profile_tag:
        os.environ['DPR_FILTER_PROFILE_TAG'] = known.profile_tag
    config_path = Path(os.environ.get('DPR_CONFIG_FILE', ROOT / 'config.yaml'))
    config = load_config_with_source_migration(str(config_path), write_back=False)
    direct = options.retrieval_source == 'arxiv'
    if options.retrieval_source != 'arxiv':
        available = database_available(config)
        if not available and options.retrieval_source == 'supabase':
            raise RuntimeError('The shared arXiv database is unavailable')
        direct = not available
    runtime_config = prepare_config(config, known.fetch_days, direct)
    print(json.dumps({'retrieval_source': 'arxiv' if direct else 'supabase',
                      'days_window': runtime_config.get('arxiv_paper_setting', {}).get('days_window'),
                      'check_only': options.check_only}), flush=True)
    if options.check_only:
        return 0
    env = dict(os.environ, DPR_STRICT_RETRIEVAL='1')
    if direct:
        print('::warning::Shared database unavailable or direct mode requested; fetching metadata from arXiv. This can take longer.', flush=True)
        # Preserve the same date window on retries and do not skip raw candidates
        # simply because a previous run fetched (but did not recommend) them.
        pipeline_args += ['--no-skip-fetch', '--fetch-ignore-seen']
        env.update(DPR_ARXIV_ENABLED='0', DPR_ENABLE_MULTI_SOURCE_RPC='0', DPR_STRICT_ARXIV_FETCH='1')
    run_dir = ROOT / '.local-runs'
    run_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='daily-', dir=run_dir) as directory:
        path = Path(directory) / 'config.yaml'
        path.write_text(yaml.safe_dump(runtime_config, allow_unicode=True, sort_keys=False), encoding='utf-8')
        env['DPR_CONFIG_FILE'] = str(path)
        return subprocess.call([sys.executable, str(ROOT / 'src/main.py'), *pipeline_args], env=env)


if __name__ == '__main__':
    sys.exit(main())
