"""Bounded page retries and checkpoints for the direct arXiv metadata fetcher."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import os
from pathlib import Path
import time

import arxiv
import requests


class _TimeoutAdapter(requests.adapters.HTTPAdapter):
    def send(self, request, **kwargs):
        if kwargs.get('timeout') is None:
            kwargs['timeout'] = (10, 60)
        return super().send(request, **kwargs)


class ResilientArxivClient(arxiv.Client):
    """Retry the failed page, without restarting pagination or splitting 429s."""

    def __init__(self, *, log=print, **kwargs):
        # Own the retry loop: arxiv.py otherwise retries every error after 3s.
        super().__init__(page_size=200, delay_seconds=5.0, num_retries=0, **kwargs)
        self.log = log
        self.retry_after = 0.0
        self.retry_budget = 900.0  # Total server backoff per process, not per page.
        self._session.mount('https://', _TimeoutAdapter())
        self._session.hooks['response'].append(self._remember_retry_after)

    def _remember_retry_after(self, response, **kwargs):
        value = response.headers.get('Retry-After', '')
        try:
            delay = float(value)
        except ValueError:
            try:
                delay = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                delay = 0.0
        self.retry_after = max(0.0, delay)

    def _parse_feed(self, url, first_page=True, _try_index=0):
        for attempt in range(5):
            self.retry_after = 0.0
            try:
                return super()._parse_feed(url, first_page, _try_index=0)
            except (arxiv.HTTPError, arxiv.UnexpectedEmptyPageError,
                    requests.ConnectionError, requests.Timeout) as error:
                status = getattr(error, 'status', None)
                if isinstance(error, arxiv.HTTPError) and status not in (429, 500, 502, 503, 504):
                    raise
                delay = max((60 if status == 429 else 10) * 2 ** attempt, self.retry_after)
                if attempt == 4 or delay > self.retry_budget:
                    raise
                self.retry_budget -= delay
                self.log(f'[arxiv] page retry {attempt + 1}/4: HTTP {status or "network/feed"}; waiting {delay:g}s')
                time.sleep(delay)


class FetchCheckpoint:
    """Resume only an unfinished run, preserving its original time bounds.

    Scope includes the workflow run ID and requested window. A new daily run
    always fetches fresh metadata. Completed runs and checkpoints older than
    24 hours are not reused. No recommendation/seen state lives in this cache.
    """

    def __init__(self, directory, scope, start, end):
        self.path = None
        self.state = {'created': time.time(), 'complete': False,
                      'start': start.isoformat(), 'end': end.isoformat(), 'windows': {}}
        if not directory:
            return
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.glob('*.json'):
            if old.stat().st_mtime < time.time() - 86400:
                old.unlink()
        key = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
        self.path = directory / f'{key}.json'
        try:
            state = json.loads(self.path.read_text(encoding='utf-8'))
            if (state.get('complete') is False and time.time() - state['created'] < 86400
                    and isinstance(state['windows'], dict)):
                datetime.fromisoformat(state['start'])
                datetime.fromisoformat(state['end'])
                self.state = state
        except (OSError, ValueError, KeyError, TypeError):
            pass
        self._save()

    @property
    def bounds(self):
        return tuple(datetime.fromisoformat(self.state[k]) for k in ('start', 'end'))

    def get(self, query):
        return self.state['windows'].get(query)

    def put(self, query, papers):
        self.state['windows'][query] = papers
        self._save()

    def finish(self):
        self.state['complete'] = True
        self._save()

    def _save(self):
        if self.path is None:
            return
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.state, ensure_ascii=False), encoding='utf-8')
        os.replace(temporary, self.path)
