"""Published recommendations, not fetch history, define already recommended works."""
import json
import re
from pathlib import Path


def canonical_id(value):
    text = str(value or '').strip()
    match = re.search(r'(?:^|/|arxiv:)(\d{4}\.\d{4,5}|[a-zA-Z.-]+/\d{7})(?:v\d+)?(?=$|[-?#/]|\.pdf)', text)
    return match.group(1).lower() if match else text


def published_ids(root):
    root = Path(root)
    seen = set()
    # The daily recommendation JSON is overwritten on a same-day rerun. The
    # cumulative published state is therefore essential, including today.
    for path in (root / 'docs').glob('**/_daily_state.json'):
        if 'long-range' in path.parts:
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        for paper in data.get('papers', []):
            seen.add(canonical_id(paper.get('paper_id') or paper.get('route')))
    for path in (root / 'archive').glob('*/recommend/arxiv_papers_*.json'):
        data = json.loads(path.read_text(encoding='utf-8'))
        for paper in data.get('deep_dive', []) + data.get('quick_skim', []):
            seen.add(canonical_id(paper.get('id') or paper.get('paper_id')))
    return seen - {''}


def filter_file(path, root):
    path = Path(path)
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding='utf-8'))
    papers = data if isinstance(data, list) else data.get('papers', [])
    seen = published_ids(root)
    retained = [p for p in papers if canonical_id(p.get('id') or p.get('paper_id')) not in seen]
    if isinstance(data, list):
        data = retained
    else:
        data['papers'] = retained
        ids = {p.get('id') or p.get('paper_id') for p in retained}
        for query in data.get('queries', []):
            if isinstance(query.get('sim_scores'), dict):
                query['sim_scores'] = {k:v for k,v in query['sim_scores'].items() if k in ids}
            query['ranked'] = [p for p in query.get('ranked', []) if p.get('paper_id') in ids]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    print(f'[history] removed={len(papers)-len(retained)} remaining={len(retained)}', flush=True)
