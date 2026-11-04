"""Rebuild duplicate daily indexes offline; never remove/rename paper or reading files."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from daily_report_state import load_daily_state, save_daily_state, entries_from_state


def reconcile(docs):
    docs = Path(docs)
    spec = importlib.util.spec_from_file_location('generate_daily_docs', ROOT / 'src/6.generate_docs.py')
    generate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate)
    paths = sorted(docs.glob('[0-9][0-9][0-9][0-9][0-9][0-9]/[0-9][0-9]/_daily_state.json'))
    changed = []
    for path in paths:
        old = json.loads(path.read_text(encoding='utf-8'))
        state = load_daily_state(str(path))
        if len(old.get('papers', [])) == len(state['papers']):
            continue
        deep, quick, evidence = entries_from_state(state)
        # Preserve run_count/generated_at: this is index maintenance, not a new recommendation.
        args = dict(docs_dir=str(docs), date_str=state['date'], date_label=state.get('date_label'),
                    deep_entries=deep, quick_entries=quick, recommend_exists=state['recommend_exists'],
                    run_count=state['run_count'], generated_at=state['generated_at'],
                    summary=f'合并同一论文的不同版本后，共 {len(deep)+len(quick)} 篇，精读 {len(deep)} 篇、速读 {len(quick)} 篇。历史文章链接与阅读记录保留。')
        generate.write_day_report_readme(**args)
        generate.update_sidebar(str(docs / '_sidebar.md'), state['date'], deep, quick, evidence,
                                date_label=state.get('date_label'), replace_existing=True)
        generate.write_day_meta_index_json(str(docs), state['date'], state.get('date_label'), [], [], deep, quick)
        if path == paths[-1]:
            # Only update a matching latest daily dashboard; never replace a newer review homepage.
            homepage = (docs / 'README.md').read_text(encoding='utf-8')
            if generate.format_date_str(state['date']) in homepage and state['generated_at'] in homepage:
                generate.sync_home_readme_from_day_report(**args, paper_evidence_by_id=evidence)
        save_daily_state(str(path), state)
        changed.append({'date': state['date'], 'before': len(old['papers']), 'after': len(state['papers'])})
    return changed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docs-dir', default=str(ROOT / 'docs'))
    args = parser.parse_args()
    print(json.dumps(reconcile(args.docs_dir), ensure_ascii=False))
