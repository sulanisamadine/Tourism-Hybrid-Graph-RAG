# -*- coding: utf-8 -*-
"""Sigiriya Hybrid GraphRAG Stage 2 Quality Fix: conservative answerability + protected RRF ranking.

Requirements: neo4j, sentence-transformers; keep the original 3 retrieval files
in the same folder. Uses the user's existing Neo4j 22 Chunk nodes and E5 indexes.

Commands:
  python Sigiriya_GraphRAG_Stage2.py --offline-check
  python Sigiriya_GraphRAG_Stage2.py --preview-graph
  python Sigiriya_GraphRAG_Stage2.py --apply-graph   # EXPLICIT DB WRITES
  python Sigiriya_GraphRAG_Stage2.py --inspect
  python Sigiriya_GraphRAG_Stage2.py --question "..."
  python Sigiriya_GraphRAG_Stage2.py --file Sigiriya_Questions_Batch.txt

All added edges are tied to verbatim passages and existing source chunk IDs.
--preview-graph and --offline-check never write to Neo4j; --apply-graph is idempotent.
New nodes/relationships are namespaced and DO NOT delete or rewrite existing
chunks, embeddings, indexes or old attraction relationships.

No LLM answer generator is used. Source-extractive answers remain heuristic.
Graph paths represent sourced statements, not independently verified facts.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from getpass import getpass
from pathlib import Path

BASE = Path(__file__).resolve().parent
MANIFEST_PATH = BASE / 'Sigiriya_KG_Expansion_Manifest.json'
OUTPUT_PATH = BASE / 'Sigiriya_GraphRAG_Stage2_Quality_Fix_Answers.json'
VALID_RELATIONS = frozenset({
    'DEVELOPED','HAS_STATUS','HAS_ATTRACTION','HAS_FEATURE','HAS_ACTIVITY',
    'DEPICTS','HAS_ADVICE','USES','NAMED_FOR','NEARBY','HAS_SERVICE',
    'HAS_EXAMPLE','TRAVEL_VIA','TRAVEL_TO','ACCESSIBLE_VIA','AVAILABLE_AT','HAS_RISK',
})
REL_PATTERN = '|'.join(sorted(VALID_RELATIONS))
# Source is consulted to validate *evidence*, not to fabricate graph facts.
SECTION_RE = re.compile(r'(?ms)^##\s+(\d+)\.\s+[^\n]+\n(.*?)(?=^##\s+\d+\.|\Z)')
TOKEN_RE = re.compile(r'[\u0D80-\u0DFF]+|[A-Za-z]{3,}|\d{4}', re.UNICODE)
STOP = {'සීගිරිය','සීගිරියේ','සීගිරියට','ගැන','කියන්න','කරන්න','විස්තර',
        'කුමක්ද','මොනවාද','දෙන්න','සහ','අතර','එහි','තිබෙන','මෙම','කොහෙද'}


def load_manifest(path=MANIFEST_PATH):
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    ids = [e['id'] for e in doc['entities']]
    if len(ids) != len(set(ids)) or not all(x.startswith('sg:') for x in ids):
        raise ValueError('Duplicate or unnamespaced entity ID')
    for f in doc['facts']:
        if f['relation'] not in VALID_RELATIONS:
            raise ValueError('Unapproved relation: ' + f['relation'])
        if f['subject'] not in ids or f['object'] not in ids:
            raise ValueError('Unknown endpoint in manifest: ' + repr(f))
        if not re.fullmatch(r'sigiriya_\d{3}', f['chunk_id']):
            raise ValueError('Unexpected chunk id')
        if not f['needle'].strip():
            raise ValueError('Missing supporting text fragment')
    return doc


def extract_excerpt(source, needle):
    """Return an actual substring of source text containing the cited fragment."""
    start = source.find(needle)
    if start < 0:
        raise ValueError(f'Evidence fragment absent from DB chunk: {needle!r}')
    # Exact source sentence / line, but do not truncate within the fragment.
    lefts = [source.rfind(sep, 0, start) for sep in ('\n', '。', '.', '!', '?')]
    left = max(lefts) + 1
    ends = [v for sep in ('\n', '。', '.', '!', '?') if (v := source.find(sep, start+len(needle))) >= 0]
    right = min(ends) + 1 if ends else len(source)
    excerpt = source[left:right].strip()
    if needle not in excerpt:
        raise AssertionError('Lost evidence during sentence extraction')
    return excerpt


def validate_facts(doc, chunks):
    checked = []
    failures = []
    for idx, fact in enumerate(doc['facts'], 1):
        cid = fact['chunk_id']
        row = chunks.get(cid)
        if row is None:
            failures.append({'fact':idx, 'chunk':cid, 'error':'Missing Chunk'})
            continue
        try:
            quote = extract_excerpt(row.get('text') or '', fact['needle'])
        except ValueError as exc:
            failures.append({'fact':idx, 'chunk':cid, 'error':str(exc)})
            continue
        identity='|'.join([fact['subject'], fact['relation'], fact['object'],cid,fact['needle']])
        stable_id=hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]
        checked.append({**fact,
                        'fact_id':f'sg:fact:{stable_id}',
                        'quote':quote,
                        'heading':row.get('heading') or ''})
    return checked, failures


def validate_offline(source_path):
    src = Path(source_path).read_text(encoding='utf-8')
    chunks = {f'sigiriya_{int(k):03}':{'text':v,'heading':''}
              for k,v in SECTION_RE.findall(src)}
    doc = load_manifest()
    good, bad = validate_facts(doc, chunks)
    print(f'Offline source validation: {len(good)}/{len(doc["facts"])} sourced links; {len(doc["entities"])} entity definitions.')
    for issue in bad: print('INVALID:', issue)
    return not bad


def get_chunks(session):
    query = """
    MATCH (c:Chunk) WHERE c.destination = 'Sigiriya'
    RETURN c.id AS id,c.heading AS heading,c.text AS text
    """
    return {r['id']:dict(r) for r in session.run(query)}


def db_connection():
    from neo4j import GraphDatabase
    from Sigiriya_Multi_Query_Retrieval import URI, USERNAME, DATABASE
    pw = getpass('Neo4j password: ')
    driver = GraphDatabase.driver(URI, auth=(USERNAME, pw))
    driver.verify_connectivity()
    return driver, DATABASE


def preview(session, doc):
    chunks = get_chunks(session)
    validated, failures = validate_facts(doc, chunks)
    support_counts = {cid:0 for cid in chunks}
    for f in validated: support_counts[f['chunk_id']] += 1
    summary = {
        'source_chunks_found':len(chunks),
        'entities_defined':len(doc['entities']),
        'facts_defined':len(doc['facts']),
        'facts_with_exact_source_passages':len(validated),
        'failures':failures,
        'chunks_with_structured_fact_links':sum(v>0 for v in support_counts.values()),
        'chunks_without_structured_facts':[k for k,v in sorted(support_counts.items()) if not v],
        'approved_relation_types':sorted({f['relation'] for f in validated}),
        'database_modified':False,
    }
    (BASE/'Sigiriya_GraphRAG_Graph_Preview.json').write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return validated, failures


def merge_entity(tx, entity):
    tx.run("""
    MERGE (e:TourismEntity {id:$id})
    SET e.name=$name,e.kind=$kind,e.aliases=$aliases,
        e.scope='Sigiriya',e.source='Curated Sigiriya tourism text'
    """, **{k:entity[k] for k in ('id','name','kind','aliases')}).consume()


def merge_fact(tx, fact):
    rel = fact['relation']
    if rel not in VALID_RELATIONS: raise ValueError('Unsafe relation')
    # Relationship type comes ONLY from a local immutable allowlist, never a user query.
    query = f"""
    MATCH (s:TourismEntity {{id:$subject}})
    MATCH (o:TourismEntity {{id:$object}})
    MATCH (c:Chunk {{id:$chunk_id}})
    WHERE c.destination='Sigiriya'
    MERGE (f:TourismFact {{id:$fact_id}})
    SET f.text=$quote,f.source_chunk=$chunk_id,f.relation=$relation,
        f.scope='Sigiriya'
    MERGE (c)-[:SUPPORTS_FACT]->(f)
    MERGE (f)-[:FACT_SUBJECT]->(s)
    MERGE (f)-[:FACT_OBJECT]->(o)
    MERGE (c)-[:LINKS_TO_ENTITY {{origin:'curated_fact'}}]->(s)
    MERGE (c)-[:LINKS_TO_ENTITY {{origin:'curated_fact'}}]->(o)
    MERGE (s)-[r:{rel} {{fact_id:$fact_id}}]->(o)
    SET r.scope='Sigiriya',r.chunk_id=$chunk_id,r.evidence=$quote
    """
    tx.run(query, **{k:fact[k] for k in ('subject','object','chunk_id','fact_id','quote','relation')}).consume()


def link_existing_attraction(tx, entity):
    name=entity.get('legacy_attraction')
    if not name:return
    tx.run("""
    MATCH (a:Attraction {name:$name})
    MATCH (e:TourismEntity {id:$id})
    MERGE (a)-[:KG_EQUIVALENT {scope:'Sigiriya'}]->(e)
    """,name=name,id=entity['id']).consume()


def _apply_all(tx, doc, validated):
    # Single transaction: error rolls back the entire new stage-2 graph.
    for ent in doc['entities']:
        merge_entity(tx,ent)
    for item in validated:
        merge_fact(tx,item)
    for ent in doc['entities']:
        if ent.get('legacy_attraction'):
            link_existing_attraction(tx,ent)

def apply_graph(session, doc, validated):
    session.execute_write(_apply_all,doc,validated)
    print(f'Applied {len(doc["entities"])} entity MERGEs and {len(validated)} sourced fact MERGEs.')
    print('No existing Chunk text, vectors or vector indexes were updated.')


def graph_inspection(session):
    queries = {
      'entities':"MATCH (e:TourismEntity {scope:'Sigiriya'}) RETURN count(e) AS n",
      'fact_nodes':"MATCH (f:TourismFact {scope:'Sigiriya'}) RETURN count(f) AS n",
      'sourced_relations':f"MATCH ()-[r:{REL_PATTERN}]->() WHERE r.scope='Sigiriya' RETURN count(r) AS n",
      'linked_chunks':"MATCH (c:Chunk)-[:LINKS_TO_ENTITY]->(:TourismEntity) WHERE c.destination='Sigiriya' RETURN count(DISTINCT c) AS n",
      'attraction_links':"MATCH (:Attraction)-[r:KG_EQUIVALENT]->(:TourismEntity) WHERE r.scope='Sigiriya' RETURN count(r) AS n",
    }
    result={k:session.run(q).single()['n'] for k,q in queries.items()}
    print('Actual graph counts:',json.dumps(result,ensure_ascii=False,indent=2))
    return result


def _norm(s):
    return re.sub(r'[\u200b\u200c\u200d\ufeff]', '', (s or '').casefold()).strip()



# === QUALITY FIX (post Stage 2 baseline diagnostics) ========================
# These source-only checks address two observed categories of unsupported
# questions. They do NOT consult evaluation IDs, gold answers, or modify Neo4j.
# Reevaluate on genuinely new questions before claiming general accuracy.

LIVE_BOOKING_TIME = (
    'මේ සවස', 'අද සවස', 'අද', 'හෙට', 'අනිද්දා', 'දැන්',
    'මේ මොහොතේ', 'මේ සතියේ', 'ලබන සතියේ', 'ඉදිරි සතියේ',
    'tonight', 'this evening', 'today', 'tomorrow', 'now',
)
LIVE_BOOKING_REQUEST = (
    'වෙන් කළ හැකි', 'වෙන්කර', 'වෙන් කර', 'වෙන් කළ',
    'වෙන්කළ', 'වෙන් කරගත', 'වෙන් කිරීම', 'ලබාගන්න පුළුවන්',
    'ලබා ගත හැකි', 'ලබාගත හැකි', 'හිස් තිබේ',
    'book', 'booking', 'reserve', 'available', 'availability',
)
LIVE_BOOKING_ITEMS = (
    'ජීප්', 'සෆාරි', 'හෝටල්', 'කාමර', 'නවාතැන්',
    'වාහන', 'මඟපෙන්වන්න', 'ප්‍රවේශපත්‍ර', 'ටිකට්',
    'jeep', 'safari', 'hotel', 'room', 'ticket', 'tour',
)
EXACT_COUNT_REQUEST = re.compile(
    r'(?:සංඛ්‍යාව\s+කීය|ගණන\s+කීය|කීයක්|කීදෙනෙක්|මුළු.{0,45}කීයද|how many|total number of)',
    re.IGNORECASE,
)
COUNT_NUMBER = re.compile(r'(?<!\d)\d{1,3}(?:,\d{3})*(?!\d)|(?<!\d)\d{5,}(?!\d)')
QUANTITY_SUFFIX = re.compile(r'\d[\d,]*\s*(?:ක්|කි|දෙනෙක්|පමණ|කින්)')


def is_live_booking_request(question):
    q = _norm(question)
    return (any(w in q for w in LIVE_BOOKING_TIME)
            and any(w in q for w in LIVE_BOOKING_REQUEST)
            and any(w in q for w in LIVE_BOOKING_ITEMS))


def quality_gate(question):
    from Sigiriya_Retrieval_Improvement_Batch_Reviewed import improved_gate
    original_result = improved_gate(question)
    if original_result['rejected']:
        return original_result
    if is_live_booking_request(question):
        return {'rejected': True,
                'reason': 'Time-specific booking/availability requires a live provider; static tourism documents cannot verify inventory.'}
    return original_result


def count_evidence_supported(question, evidence):
    """Negative evidence check for *explicit totals* (not general numerical facts).

    Requires target words plus an explicit quantity statement. A museum opening
    time such as 8.30 must not be mistaken for the number of exhibits.
    Returning True means 'candidate evidence' only, NOT proven factual accuracy.
    """
    q = (question or "").casefold().strip()
    if not EXACT_COUNT_REQUEST.search(q):
        return True
    t = (evidence or "").casefold().strip()
    if not t:
        return False
    if ('ප්‍රදර්ශන' in q or 'භාණ්ඩ' in q or 'exhibit' in q):
        if not any(w in t for w in ('ප්‍රදර්ශන භාණ්ඩ','ප්‍රදර්ශනය කරන භාණ්ඩ','පුරාවස්තු','exhibit')):
            return False
    if ('පඩි' in q or 'පඩිපෙළ' in q):
        if not any(w in t for w in ('පඩි','පඩිපෙළ','steps','stairs')):
            return False
    # Require an actual count expression; general time and date numerals do not suffice.
    if not QUANTITY_SUFFIX.search(t) and not (
        ('සංඛ්‍යාව' in t or 'මුළු' in t or 'ගණන' in t or 'total' in t)
        and COUNT_NUMBER.search(t)
    ):
        return False
    return True


def quality_answer(question, session, model, sources, headings, debug=False):
    from Sigiriya_Retrieval_Improvement_Batch_Reviewed import (
        answer_question, NO_SOURCE_SI, NO_LIVE_SI,
    )
    gate = quality_gate(question)
    if gate['rejected']:
        return {'question':question,'status':'LIVE_DATA_UNSUPPORTED',
                'answer':NO_LIVE_SI,'sources':[],'reason':gate['reason'],
                'subquestions':[]}
    result = answer_question(question, session, model, sources, headings, debug=debug)
    if result.get('status') in ('SOURCE_GROUNDED_EXTRACTIVE',
                                'SOURCE_GROUNDED_FACT_EXTRACTED',
                                'PARTIAL_SOURCE_EVIDENCE'):
        excerpts=' '.join(r.get('source_quote','') for r in result.get('sources',[]))
        if not count_evidence_supported(question, excerpts):
            result.update(status='INSUFFICIENT_DOCUMENT_EVIDENCE',
                          answer=NO_SOURCE_SI,sources=[],
                          reason='No explicit source-backed total/count for the requested subject.')
    return result

# === END QUALITY FIX ========================================================


def find_anchors(question, entities, max_anchors=6):
    q = _norm(question)
    matches=[]
    for ent in entities:
        candidates=[a for a in ent['aliases'] if len(_norm(a))>=4 and _norm(a) in q]
        if candidates:
            a=max(candidates,key=lambda x:len(_norm(x)))
            matches.append((len(_norm(a)),ent['id']))
    matches.sort(key=lambda x:(-x[0],x[1]))
    names=[eid for _,eid in matches if eid!='sg:dest'][:max_anchors]
    return names or ['sg:dest']


def resolve_anchors(session, question, entities, semantic_ranking):
    """Resolve named entities from BOTH the question and semantically found chunks.

    Linked graph candidates are read from the existing chunk-entity connection,
    so synonym/paraphrase questions need not contain literal entity names.
    No test gold is supplied here.
    """
    explicit=find_anchors(question,entities,max_anchors=6)
    ranked=list(explicit) if explicit!=['sg:dest'] else []
    top=list(dict.fromkeys(semantic_ranking[:5]))
    if top:
        cypher="""
        UNWIND range(0,size($ids)-1) AS pos
        MATCH (c:Chunk {id:$ids[pos]})-[:LINKS_TO_ENTITY]->(e:TourismEntity)
        WHERE c.destination='Sigiriya' AND e.scope='Sigiriya'
        RETURN pos,e.id AS id,e.kind AS kind,e.aliases AS aliases,
               c.heading AS heading
        ORDER BY pos
        LIMIT 100
        """
        candidates=[]
        for rec in session.run(cypher,ids=top):
            entid=rec['id']
            if entid=='sg:dest' or entid in ranked: continue
            aliases=rec['aliases'] or []
            in_heading=max((len(_norm(a)) for a in aliases if len(_norm(a))>=4 and _norm(a) in _norm(rec['heading'])),default=0)
            match_query=max((len(_norm(a)) for a in aliases if len(_norm(a))>=4 and _norm(a) in _norm(question)),default=0)
            kind_prior=1 if rec['kind'] in ('Attraction','NearbyPlace','Service','HistoricFeature','EngineeringFeature') else 0
            value=4*match_query+2*in_heading+kind_prior-3*rec['pos']
            candidates.append((value,entid))
        candidates.sort(key=lambda x:(-x[0],x[1]))
        for _,eid in candidates:
            if eid not in ranked:ranked.append(eid)
            if len(ranked)>=6:break
    return ranked[:6] or ['sg:dest']

def source_terms(s):
    return {w for w in TOKEN_RE.findall(_norm(s)) if len(w)>=3 and w not in STOP}


def overlap(a,b):
    a,b=source_terms(a),source_terms(b)
    if not a:return 0.
    return len(a&b)/len(a)


def graph_paths(session, question, entities, anchors, limit=20):
    """Execute *actual* directed graph traversals up to two edges.

    A two-edge return cannot be faked by a single chunk lookup: it requires
    persisted linked nodes and sourced relationships.
    """
    seeds=list(dict.fromkeys([*anchors,'sg:dest']))[:7]
    query=f"""
    MATCH (s:TourismEntity)
    WHERE s.id IN $seeds AND s.scope='Sigiriya'
    MATCH p=(s)-[:{REL_PATTERN}*1..2]->(t:TourismEntity)
    WHERE ALL(r IN relationships(p) WHERE r.scope='Sigiriya' AND r.fact_id IS NOT NULL)
    RETURN s.id AS seed,
      [n IN nodes(p) | n.id] AS entity_ids,
      [n IN nodes(p) | n.name] AS names,
      [r IN relationships(p) | {{relation:type(r), fact_id:r.fact_id,
                                  chunk_id:r.chunk_id,quote:r.evidence}}] AS edges
    LIMIT 300
    """
    rows=[dict(r) for r in session.run(query,seeds=seeds)]
    byid={e['id']:e for e in entities}
    qnorm=_norm(question)
    for p in rows:
        names=p['names']
        name_matches=sum(any(_norm(alias) in qnorm and len(_norm(alias))>=4
                             for alias in byid[eid]['aliases'])
                         for eid in p['entity_ids'] if eid in byid)
        quoted=' '.join(edge.get('quote') or '' for edge in p['edges'])
        p['score']=round(3.0*name_matches + 2.0*overlap(question, quoted) - 0.20*len(p['edges']),4)
    rows.sort(key=lambda r:(-r['score'],-len(r['edges']),r['entity_ids']))
    seen=set(); result=[]
    for p in rows:
        key=tuple(p['entity_ids'])
        if key not in seen:
            result.append(p);seen.add(key)
        if len(result)>=limit: break
    return result


def rank_chunks(original, multi, paths, topk=3):
    """Conservative graph-enriched retrieval without RRF top-k displacement.

    Preserve the original RRF ranking at @3 (the best-scoring retriever on
    the development set), while still returning audited graph paths separately.
    Graph facts can influence diagnostic scores for candidates OUTSIDE the
    protected window, but cannot displace the original top-3. Therefore any
    retrieval gain claimed for this approach must be tested separately.
    """
    scores=defaultdict(float)
    for ranking, weight in ((original,1.0),(multi,0.7)):
        for i,cid in enumerate(ranking,1):
            scores[cid] += weight / (60+i)
    for path in paths[:15]:
        boost=0.001 * max(0.0, min(1.0, path.get('score',0)/9.0))
        for edge in path['edges']:
            scores[edge['chunk_id']] += boost
    protected=list(dict.fromkeys(original[:topk]))
    for cid in multi:
        if len(protected)>=topk: break
        if cid not in protected: protected.append(cid)
    other=sorted((cid for cid in scores if cid not in protected),
                 key=lambda cid:(-scores[cid],cid))
    ranked=protected+other
    return ranked[:topk], ranked


def startup_retrieval(session):
    from sentence_transformers import SentenceTransformer
    from Sigiriya_Multi_Query_Retrieval import HEADINGS_QUERY, MODEL_NAME
    from Sigiriya_Retrieval_Improvement_Batch_Reviewed import load_sources
    print('Loading multilingual E5 once...')
    model=SentenceTransformer(MODEL_NAME)
    headings={r['chunk_id']:r['heading'] for r in session.run(HEADINGS_QUERY)}
    sources=load_sources(session)
    return model,headings,sources


def process_question(question, session, model, headings, sources, doc):
    from Sigiriya_Multi_Query_Retrieval import fused_search,multi_query_search
    start=time.perf_counter()
    base=quality_answer(question,session,model,sources,headings)
    base['graph_anchors']=[]
    base['graph_paths']=[]
    base['hybrid_top3']=[]
    base['original_top3']=[]
    base['multi_query_top3']=[]
    if quality_gate(question)['rejected']:
        base['latency_ms']=round((time.perf_counter()-start)*1000,2)
        return base
    orig=fused_search(session,model,question)
    _,multi=multi_query_search(session,model,question,headings)
    base['original_top3']=orig[:3];base['multi_query_top3']=multi[:3]
    base['graph_anchors']=resolve_anchors(session,question,doc['entities'],orig)
    paths=graph_paths(session,question,doc['entities'],base['graph_anchors'])
    hybrid,_=rank_chunks(orig,multi,paths)
    base['graph_paths']=paths[:8]
    base['hybrid_top3']=hybrid
    # Existing source-extractive answer remains conservative; graph paths are
    # independently traceable evidence, not automatic proof of answerability.
    base['latency_ms']=round((time.perf_counter()-start)*1000,2)
    return base


def format_result(res,index=None,total=None):
    from Sigiriya_Retrieval_Improvement_Batch_Reviewed import format_answer
    displayed=format_answer(res,index,total)
    extra=['', 'GRAPH RELATIONSHIP EVIDENCE (source-traceable 1–2 hop paths):']
    if res.get('graph_paths'):
        for p in res['graph_paths'][:4]:
            chain=p['names'][0]
            for step,name in zip(p['edges'],p['names'][1:]):
                chain+=f" -[{step['relation']}]-> {name}"
            extra.append('  '+chain)
            extra.extend(f"    {edge['chunk_id']}: {edge['quote']}" for edge in p['edges'])
    else:
        extra.append('  No applicable sourced graph traversal found; or live-data gate applied.')
    extra.append('Evidence top3 (Original / Multi-query / Graph-Hybrid):')
    extra.append('  '+str(res.get('original_top3',[]))+' / '+str(res.get('multi_query_top3',[]))+' / '+str(res.get('hybrid_top3',[])))
    return displayed+'\n'+'\n'.join(extra)


def quality_self_test():
    # No Neo4j connection, embedding model, or third-party packages required.
    booking='මේ සවස මින්නේරියේ ජීප් සෆාරි වාහනයක් වෙන් කළ හැකිද?'
    count='සීගිරි කෞතුකාගාරය තුළ තිබෙන මුළු ප්‍රදර්ශන භාණ්ඩ සංඛ්‍යාව කීයද?'
    assert is_live_booking_request(booking), 'Expected live safari booking detection'
    assert not is_live_booking_request('මින්නේරියේ ජීප් සෆාරි ගැන සාමාන්‍ය තොරතුරු කියන්න.')
    assert not count_evidence_supported(count,'කෞතුකාගාරය උදෑසන 8.30 සිට සවස 5.30 දක්වා විවෘතය.'), 'Opening hours are NOT an exhibits count'
    assert count_evidence_supported(count,'කෞතුකාගාරයේ ප්‍රදර්ශන භාණ්ඩ 120ක් තිබේ.'), 'Explicit exhibit count should pass the negative check'
    orig=['sigiriya_010','sigiriya_004','sigiriya_003','sigiriya_020']
    multi=['sigiriya_004','sigiriya_010','sigiriya_005','sigiriya_003']
    rank,_=rank_chunks(orig,multi,[{'score':9,'edges':[{'chunk_id':'sigiriya_018'}]}])
    assert rank==orig[:3], 'Graph expansion must not evict original RRF top-3'
    print('PASS: time-specific booking detection, exhibit-count evidence check, and protected RRF ranking.')
    print('NOTE: Offline logic tests only; full Neo4j evaluation has NOT been executed here.')
    return 0


def main():
    parser=argparse.ArgumentParser(description='Sigiriya Stage 2: expanded KG + actual 1–2 hop retrieval')
    g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--offline-check',action='store_true')
    g.add_argument('--quality-self-test',action='store_true')
    g.add_argument('--preview-graph',action='store_true')
    g.add_argument('--apply-graph',action='store_true')
    g.add_argument('--inspect',action='store_true')
    g.add_argument('--question',type=str)
    g.add_argument('--file',type=Path)
    args=parser.parse_args()
    if args.quality_self_test:
        return quality_self_test()
    if args.offline_check:
        path=next(iter(BASE.glob('SIGIRIYA*Sinhalatxt*.txt')),None)
        if not path:
            print('Place original Sinhala Sigiriya source .txt in this folder to run offline validation.',file=sys.stderr)
            return 2
        return 0 if validate_offline(path) else 2
    try:
        doc=load_manifest()
        driver,database=db_connection()
        try:
            with driver.session(database=database) as session:
                if args.preview_graph:
                    _,bad=preview(session,doc)
                    return 0 if not bad else 2
                if args.apply_graph:
                    checked,bad=preview(session,doc)
                    if bad:
                        print('ABORTING: no graph writes. Fix source mismatches above.',file=sys.stderr)
                        return 2
                    if len(get_chunks(session))!=22:
                        print('ABORTING: expected 22 existing Sigiriya Chunk nodes.',file=sys.stderr)
                        return 2
                    apply_graph(session,doc,checked)
                    graph_inspection(session)
                    return 0
                if args.inspect:
                    graph_inspection(session)
                    return 0
                if not session.run("MATCH (e:TourismEntity {id:'sg:dest'}) RETURN count(e) AS n").single()['n']:
                    print('Graph expansion not applied. Run --preview-graph, then --apply-graph.',file=sys.stderr)
                    return 2
                from Sigiriya_Retrieval_Improvement_Batch_Reviewed import parse_question_lines
                if args.file:
                    questions=parse_question_lines(args.file.read_text(encoding='utf-8-sig'))
                else:
                    questions=[args.question]
                if not questions:
                    print('No questions supplied.',file=sys.stderr);return 2
                model,headings,sources=startup_retrieval(session)
                results=[]
                for i,q in enumerate(questions,1):
                    result=process_question(q,session,model,headings,sources,doc)
                    results.append(result)
                    print(format_result(result,i,len(questions)))
                OUTPUT_PATH.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
                print(f'\nSaved {OUTPUT_PATH}. Neo4j is read-only during question answering.')
        finally:
            driver.close()
        return 0
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
