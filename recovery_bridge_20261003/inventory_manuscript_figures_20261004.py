"""Lightweight source/asset inventory, not rendering or empirical validation."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
OUTPUT = PAPER / 'MANUSCRIPT_FIGURE_INVENTORY_20261004.json'


def asset(path):
    return dict(path=str(path), exists=path.is_file(),
                bytes=path.stat().st_size if path.is_file() else None,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None)


def main():
    assert not OUTPUT.exists(), 'Preserve inventory snapshots'
    sources = ['main.tex', 'migration_primary_body.tex', 'historical_source_free_exploration.tex', 'migration_gate_map.tex']
    figures = []
    for name in sources:
        text = (PAPER/name).read_text(encoding='utf-8')
        for match in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text):
            path = (PAPER/match.group(1)).resolve()
            assert path.is_relative_to(ROOT), path
            item = dict(source=name, reference=match.group(1), pdf=asset(path), png=asset(path.with_suffix('.png')))
            if path.name.startswith('figure'):
                item['generation_source_candidate'] = asset(PAPER/'figures/regenerate_figures.R')
                item['readiness'] = 'Historical asset; full numerical/render/figure-code correspondence revalidation pending. Label-source revision not yet rendered.'
            else:
                item['readiness'] = 'Separately audited release exists; this inventory only verifies file existence/hash, not audit scope or visual quality.'
            figures.append(item)
    result = dict(scope='Static inventory only; no figures rendered, no empirical numbers rescored, no active trial read.',
                  sources={name:asset(PAPER/name) for name in sources}, figure_count=len(figures),
                  all_referenced_pdfs_exist=all(f['pdf']['exists'] for f in figures),
                  all_sibling_pngs_exist=all(f['png']['exists'] for f in figures), figures=figures,
                  inventory_source=asset(Path(__file__)), manuscript_pdf_current=False,
                  limits=['R generation-source presence is not proof that an existing PDF corresponds to its latest code.',
                          'Existence is not empirical validity, sufficient resolution, readability or completed whole-paper verification.'])
    OUTPUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['sources','figures','inventory_source']},indent=2))


if __name__ == '__main__':
    main()
