"""CPU checks of the research snapshot: path mapping, entry-point templates, no machine-specific paths in code."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'premind'


def test_pathmap_remap(tmp_path, monkeypatch):
    import premind.pathmap as PM
    monkeypatch.setattr(PM, '_MAP', [('/orig/root/sub', str(tmp_path / 'local')), ('/orig/root', str(tmp_path))])
    assert PM.remap('/orig/root/sub/a.json') == str(tmp_path / 'local' / 'a.json')
    assert PM.remap('/orig/root/b.json') == str(tmp_path / 'b.json')
    assert PM.remap('/elsewhere/c.json') == '/elsewhere/c.json'
    assert PM.remap(3) == 3


def test_args_templates_expand():
    for phase in ('phase1', 'phase2'):
        args = json.loads((ROOT / 'training/cprime' / phase / 'args_4b.json').read_text())
        assert '--release' in args and '--out' in args
        left = {m for a in args for m in re.findall(r'\$\{([A-Z0-9_]+)\}', a)}
        assert left <= {'FC', 'PKG', 'OUT', 'PHASE1_RUN'}, left


def test_no_machine_paths_in_code():
    bad = []
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix in ('.py', '.json', '.md', '.txt', '.sh'):
            s = p.read_text(errors='ignore')
            if '/' + 'data03/' in s or '.claude/' + 'jobs' in s or re.search('/' + r'home/[a-z0-9_]+/(anaconda3|\.cache|\.claude)', s):
                bad.append(str(p.relative_to(ROOT)))
    assert not bad, bad
