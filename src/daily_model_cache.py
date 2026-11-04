"""Optional persistent cache. Keys hash complete inputs; only validated outputs enter."""
import hashlib
import json
import math
import os
import sqlite3
import time
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


_PRUNED = set()


class ModelCache:
    def __init__(self, namespace):
        self.namespace = namespace
        self.db = None
        directory = os.getenv('DPR_MODEL_CACHE_DIR')
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(str(Path(directory) / 'results.sqlite3'), timeout=60)
            self.db.execute('CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, value TEXT NOT NULL, created REAL NOT NULL)')
            if directory not in _PRUNED:
                self.db.execute("DELETE FROM results WHERE created < ?", (time.time()-45*86400,))
                self.db.execute("DELETE FROM results WHERE key IN (SELECT key FROM results ORDER BY created DESC LIMIT -1 OFFSET 100000)")
                _PRUNED.add(directory)
            self.db.commit()

    def get(self, inputs):
        if self.db is None:
            return None
        row = self.db.execute('SELECT value FROM results WHERE key=? AND created>?',
                             (fingerprint([self.namespace, inputs]), time.time()-45*86400)).fetchone()
        try:
            return json.loads(row[0]) if row else None
        except (ValueError, TypeError):
            return None

    def put(self, inputs, result):
        if self.db is not None:
            payload = json.dumps(result, allow_nan=False)
            with self.db:
                self.db.execute('INSERT OR REPLACE INTO results VALUES (?,?,?)',
                                (fingerprint([self.namespace, inputs]), payload, time.time()))

    def close(self):
        if self.db is not None:
            self.db.close()


def cached_rerank(reranker, query, documents, model):
    """Cache exact ordered batches: ranker semantics may depend on other documents."""
    cache = ModelCache('rerank-v1')
    key = [type(reranker).__name__, getattr(reranker, 'base_url', ''), model, query, documents]
    try:
        response = cache.get(key)
        if response is not None:
            print(f'[cache:rerank] hit documents={len(documents)}', flush=True)
            return response
        response = reranker.rerank(query=query, documents=documents, top_n=len(documents), model=model)
        results = response.get('output', response).get('results', [])
        indices = [r.get('index') for r in results]
        if len(indices) != len(documents) or set(indices) != set(range(len(documents))):
            raise ValueError('Incomplete reranker result; not cached')
        if not all(math.isfinite(float(r.get('relevance_score', r.get('score', float('nan'))))) for r in results):
            raise ValueError('Invalid reranker scores; not cached')
        cache.put(key, response)
        print(f'[cache:rerank] miss documents={len(documents)}', flush=True)
        return response
    finally:
        cache.close()
