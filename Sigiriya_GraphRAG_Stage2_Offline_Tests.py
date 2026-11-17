# -*- coding: utf-8 -*-
"""Zero-DB unit tests: manifest provenance, route existence, fusion and labels.

Run: python Sigiriya_GraphRAG_Stage2_Offline_Tests.py
These are structural offline checks. They are NOT accuracy results on Neo4j.
"""
import csv
import re
from collections import defaultdict
from pathlib import Path
from Sigiriya_GraphRAG_Stage2 import (
    load_manifest,validate_facts,extract_excerpt,rank_chunks,validate_offline,
)

BASE=Path(__file__).resolve().parent
source=next(iter(BASE.glob('SIGIRIYA*Sinhalatxt*.txt')),None)
assert source is not None,'Include original Sinhala Sigiriya .txt source'
assert validate_offline(source)
manifest=load_manifest()
text=source.read_text(encoding='utf-8')
sections={f'sigiriya_{int(n):03}':{'text':v,'heading':str(n)} for n,v in re.findall(
    r'(?ms)^##\s+(\d+)\.\s+[^\n]+\n(.*?)(?=^##\s+\d+\.|\Z)',text)}
assert len(sections)==22
valid,missing=validate_facts(manifest,sections)
assert not missing and len(valid)==54
assert len({r['fact_id'] for r in valid})==len(valid)
assert len(manifest['entities'])==49
assert len({x['chunk_id'] for x in valid})==22
pairs={(r['subject'],r['object']) for r in valid}
rows=list(csv.DictReader((BASE/'Sigiriya_GraphRAG_Stage2_Candidate_Evaluation.csv').open(
    'r',encoding='utf-8-sig',newline='')))
assert len(rows)==30 and len({r['question'] for r in rows})==30
for row in rows:
    if row['gold_chunks']:
        assert set(row['gold_chunks'].split(';'))<=set(sections)
    path=[p for p in row['gold_path'].split('>') if p]
    if path:
        assert len(path)==3 and all((a,b) in pairs for a,b in zip(path,path[1:])),row['id']
# No gold IDs passed into this fake ranking test.
paths=[{'score':8.,'edges':[{'chunk_id':'sigiriya_020'}]}]
top3,full=rank_chunks(['sigiriya_001','sigiriya_002','sigiriya_003'],
                       ['sigiriya_002','sigiriya_004','sigiriya_003'],paths)
assert len(top3)==3 and len(set(top3))==3
print('PASS: 22/22 sections with sourced facts; 49 entity definitions; 54 exact-source relations.')
print('PASS: 30 annotated candidate questions validated for IDs and graph path structure.')
print('PASS: source extraction + graph fusion smoke tests; database was not accessed.')
