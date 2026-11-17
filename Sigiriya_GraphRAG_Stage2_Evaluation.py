# -*- coding: utf-8 -*-
"""Read-only comparison: old rule baseline, RRF, Multi-Query, and graph-hybrid.

Uses a frozen candidate question CSV. No answer labels or gold chunk IDs are
passed to retrieval, anchor identification, or answer generation.

IMPORTANT: Candidate questions are source-informed and provisionally annotated.
Not externally blind. Verify labels with a supervisor/second annotator before
reporting final precision, recall, F1 or multi-hop accuracy.

Run: python Sigiriya_GraphRAG_Stage2_Evaluation.py
"""
import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

import Sigiriya_GraphRAG_Stage2 as engine

HERE=Path(__file__).resolve().parent
GOLD=HERE/'Sigiriya_GraphRAG_Stage2_Candidate_Evaluation.csv'
RESULT=HERE/'Sigiriya_GraphRAG_Stage2_Evaluation_Results.csv'
SUMMARY=HERE/'Sigiriya_GraphRAG_Stage2_Evaluation_Summary.json'


def load_cases(path):
    rows=list(csv.DictReader(Path(path).open('r',encoding='utf-8-sig',newline='')))
    if not rows or not all(r.get('kind') and r.get('question') for r in rows):
        raise ValueError('Test CSV empty/invalid')
    q=[r['question'].strip() for r in rows]
    if len(q)!=len(set(q)):raise ValueError('Duplicate evaluation questions')
    return rows


def split_field(s,delimiter=';'):
    return [x.strip() for x in (s or '').split(delimiter) if x.strip()]


def metrics(gold,topk):
    gold=set(gold);retrieved=set(topk[:3]);hits=len(gold&retrieved)
    if not gold:return {'precision_at3':'','recall_at3':'','f1_at3':'','all_required_at3':''}
    # Precision denominator = number retrieved in top3 (not total corpus).
    precision=hits/max(len(retrieved),1)
    recall=hits/len(gold)
    f1=2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.
    return dict(precision_at3=round(precision,4),recall_at3=round(recall,4),
                f1_at3=round(f1,4),all_required_at3=int(hits==len(gold)))


def evaluate(session,model,headings,sources,doc,tests):
    from Sigiriya_Retrieval_Improvement_Batch_Reviewed import answer_question,improved_gate
    from Sigiriya_Multi_Query_Retrieval import fused_search,multi_query_search
    from sigiriya_adaptive_retrieval_fixed import detect_attraction
    # Alias -> chunk ID determined from actual Neo4j link, not expected labels.
    legacy_lookup={r['attraction']:r['chunk_id'] for r in session.run('''
      MATCH (a:Attraction)-[:MENTIONS]-(c:Chunk)
      WHERE c.destination='Sigiriya'
      RETURN a.name AS attraction, min(c.id) AS chunk_id''')}
    results=[]
    for i,test in enumerate(tests,1):
        question=test['question'].strip()
        start=time.perf_counter()
        gate=improved_gate(question)
        stage_ms={'answer_ms':None,'original_rrf_ms':None,'multi_query_ms':None,'graph_expansion_ms':None}
        if gate['rejected']:
            base={'status':'LIVE_DATA_UNSUPPORTED','answer':'','sources':[]}
            original=[];multi=[];hybrid=[];paths=[];anchors=[]
        else:
            s0=time.perf_counter()
            base=answer_question(question,session,model,sources,headings)
            stage_ms['answer_ms']=round(1000*(time.perf_counter()-s0),2)
            s0=time.perf_counter()
            original=fused_search(session,model,question)
            stage_ms['original_rrf_ms']=round(1000*(time.perf_counter()-s0),2)
            s0=time.perf_counter()
            _,multi=multi_query_search(session,model,question,headings)
            stage_ms['multi_query_ms']=round(1000*(time.perf_counter()-s0),2)
            s0=time.perf_counter()
            anchors=engine.resolve_anchors(session,question,doc['entities'],original)
            paths=engine.graph_paths(session,question,doc['entities'],anchors,limit=30)
            hybrid,_=engine.rank_chunks(original,multi,paths)
            stage_ms['graph_expansion_ms']=round(1000*(time.perf_counter()-s0),2)
        ms=round(1000*(time.perf_counter()-start),2)
        name=detect_attraction(question)
        baseline=[legacy_lookup[name]] if name in legacy_lookup and not gate['rejected'] else []
        # GOLD LABELS APPEAR ONLY BELOW THIS LINE, AFTER EVERY RETRIEVAL.
        gold=split_field(test.get('gold_chunks'))
        gold_anchors=split_field(test.get('gold_anchors'))
        expected_path=split_field(test.get('gold_path'),'>')
        paths3=[p['entity_ids'] for p in paths if len(p['edges'])==2]
        correct_path=bool(expected_path and expected_path in paths3)
        start_recall=(len(set(gold_anchors)&set(anchors))/len(set(gold_anchors))) if gold_anchors else ''
        is_supported=test['kind'] in ('single','multi','multi_hop')
        safe_decline=base['status'] in ('INSUFFICIENT_DOCUMENT_EVIDENCE','LIVE_DATA_UNSUPPORTED')
        result={
            'id':test['id'],'kind':test['kind'],'question':question,
            'gold_chunks_scoring_only':';'.join(gold),
            'legacy_rule_top3':';'.join(baseline[:3]),
            'original_rrf_top3':';'.join(original[:3]),
            'multi_query_top3':';'.join(multi[:3]),
            'graph_hybrid_top3':';'.join(hybrid[:3]),
            'gold_anchors_scoring_only':';'.join(gold_anchors),
            'predicted_graph_anchors':';'.join(anchors),
            'starting_entity_recall':start_recall,
            'gold_2hop_path_scoring_only':'>'.join(expected_path),
            '2hop_path_found':int(correct_path) if expected_path else '',
            'found_two_hop_paths':sum(len(p['edges'])==2 for p in paths),
            'answer_status':base['status'],
            'unsupported_safely_declined':int(safe_decline) if not is_supported else '',
            'supported_falsely_declined':int(safe_decline) if is_supported else '',
            'latency_ms':ms,
            **stage_ms,
        }
        for tag,rank in [('legacy_rule',baseline),('original_rrf',original),('multi_query',multi),('graph_hybrid',hybrid)]:
            score=metrics(gold,rank)
            for key,value in score.items():result[f'{tag}_{key}']=value
            if test['kind']=='single':
                result[f'{tag}_top1_correct']=int(bool(rank) and rank[0]==gold[0])
            else:result[f'{tag}_top1_correct']=''
        results.append(result)
        print(f"Q{i:02d} [{test['kind']}] {test['id']} | RRF {original[:3]} | Graph {hybrid[:3]} | path={result['2hop_path_found']} | {base['status']}")
    return results


def mean(rows,key):
    nums=[float(r[key]) for r in rows if isinstance(r.get(key),(int,float)) and r[key]!='']
    return round(statistics.mean(nums),4) if nums else None


def summarize(rows):
    supported=[r for r in rows if r['kind'] in ('single','multi','multi_hop')]
    static=[r for r in rows if r['kind']=='unsupported_static']
    live=[r for r in rows if r['kind']=='unsupported_live']
    multis=[r for r in rows if r['kind'] in ('multi','multi_hop')]
    hops=[r for r in rows if r['kind']=='multi_hop']
    summary={'question_count':len(rows),'supported_count':len(supported),'unsupported_live_count':len(live),
      'unsupported_static_count':len(static),
      'methods':{},
      'graph_starting_entity_recall':mean(supported,'starting_entity_recall'),
      'graph_two_hop_exact_path_rate':mean(hops,'2hop_path_found'),
      'graph_two_hop_questions_with_any_path':sum(r['found_two_hop_paths']>0 for r in hops),
      'unsafe_supported_decline_rate':mean(supported,'supported_falsely_declined'),
      'live_safely_declined':sum(r['unsupported_safely_declined']==1 for r in live),
      'static_safely_declined':sum(r['unsupported_safely_declined']==1 for r in static),
      'unsupported_not_declined':[r['id'] for r in live+static if r['unsupported_safely_declined']!=1],
      'mean_latency_ms':mean(rows,'latency_ms'),
      'mean_original_rrf_ms':mean(rows,'original_rrf_ms'),
      'mean_multi_query_ms':mean(rows,'multi_query_ms'),
      'mean_graph_expansion_ms':mean(rows,'graph_expansion_ms'),
      'mean_answer_ms':mean(rows,'answer_ms'),
      'limitations':['Source-informed, author-authored provisional labels; not a blind external evaluation.',
       'A retrieved relevant chunk is not proof the generated answer is correct.',
       'Source passages may describe outdated prices/hours; live data remains unsupported.',
       'Multi-hop success here tests retrieved paths, not multi-step reasoning correctness.'],
    }
    for method in ('legacy_rule','original_rrf','multi_query','graph_hybrid'):
        summary['methods'][method]={
            'mean_precision_at3':mean(supported,method+'_precision_at3'),
            'mean_recall_at3':mean(supported,method+'_recall_at3'),
            'mean_f1_at3':mean(supported,method+'_f1_at3'),
            'all_required_at3':sum(r[method+'_all_required_at3']==1 for r in supported),
            'all_required_at3_denominator':len(supported),
            'multi_all_required_at3':sum(r[method+'_all_required_at3']==1 for r in multis),
            'multi_denominator':len(multis),
            'single_top1':sum(r[method+'_top1_correct']==1 for r in rows if r['kind']=='single'),
            'single_denominator':sum(r['kind']=='single' for r in rows),
        }
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tests',type=Path,default=GOLD,help='Frozen CSV of test questions and provisional gold annotations')
    args=parser.parse_args()
    cases=load_cases(args.tests)
    doc=engine.load_manifest()
    driver,database=engine.db_connection()
    try:
        with driver.session(database=database) as session:
            if not session.run("MATCH (e:TourismEntity {id:'sg:dest'}) RETURN count(e) AS n").single()['n']:
                raise RuntimeError('Expanded graph missing. First run Stage2.py --preview-graph and --apply-graph.')
            model,headings,sources=engine.startup_retrieval(session)
            rows=evaluate(session,model,headings,sources,doc,cases)
    finally:
        driver.close()
    with RESULT.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    summary=summarize(rows)
    summary['question_set_sha256']=hashlib.sha256(args.tests.read_bytes()).hexdigest()
    summary['evaluation_writes_to_neo4j']=False
    SUMMARY.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nSUMMARY\n'+json.dumps(summary,ensure_ascii=False,indent=2))
    print('\nSaved:',RESULT,'and',SUMMARY)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
