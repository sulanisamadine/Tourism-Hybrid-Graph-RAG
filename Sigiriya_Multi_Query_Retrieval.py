"""Sigiriya multi-query retrieval experiment (read-only).

Uses existing Sinhala E5 embeddings and Neo4j vector indexes.
For questions mentioning two topics, tries simple Sinhala rule-based
question decomposition, searches each sub-question independently, and
balances the top results so both topics can contribute evidence.

This is NOT LLM-based decomposition, answer generation, or a validated
answerability / unsupported-question detector. Gold IDs are used ONLY
for evaluation; they never influence retrieval or reranking.
"""

import argparse
import csv
import re
from getpass import getpass
from pathlib import Path

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

URI = "bolt://127.0.0.1:7687"
USERNAME = "neo4j"
DATABASE = "neo4j"
MODEL_NAME = "intfloat/multilingual-e5-base"
TEXT_INDEX = "sigiriya_chunk_vectors"
HEADING_INDEX = "sigiriya_heading_vectors"
OUTPUT_CSV = Path(__file__).with_name("Sigiriya_Multi_Query_Results.csv")
TOP_K = 3

# The same five single-topic and five multi-topic development questions
# from Sigiriya_Challenge_Evaluation.py. Gold IDs are never used to retrieve.
TESTS = [
    {"type": "single", "question": "සීගිරියේ ගල් අතර පැරණි ශිලා ලේඛන දැකිය හැක්කේ කොහේද?", "expected": ["sigiriya_008"]},
    {"type": "single", "question": "පර්වතය රාජකීය නගරයක් බවට පත් කළ පාලකයා ගැන විස්තර මොනවාද?", "expected": ["sigiriya_002"]},
    {"type": "single", "question": "පැරණි අමුත්තන් කවි ලියූ ඔප දැමූ බිත්තිය ගැන කියන්න.", "expected": ["sigiriya_005"]},
    {"type": "single", "question": "සර්පිලාකාර පඩිපෙළකින් ළඟාවෙන කාන්තා රූප තිබෙන තැන කුමක්ද?", "expected": ["sigiriya_004"]},
    {"type": "single", "question": "සීගිරිය බලන්න එන අයට බඹර කූඩු අසල හැසිරෙන්න ඕනේ කොහොමද?", "expected": ["sigiriya_021"]},
    {"type": "multi", "question": "සීගිරියේ ජල උද්‍යාන සහ දිය උල්පත් උද්‍යානයේ ලක්ෂණ දෙකම විස්තර කරන්න.", "expected": ["sigiriya_006", "sigiriya_007"]},
    {"type": "multi", "question": "සීගිරි බිතුසිතුවම් සහ කැටපත් පවුර ගැන වෙන් වෙන්ව කියන්න.", "expected": ["sigiriya_004", "sigiriya_005"]},
    {"type": "multi", "question": "සීගිරියට යන ගමන් මාර්ගයත් එහි යන්න සුදුසු වේලාවත් මොනවාද?", "expected": ["sigiriya_016", "sigiriya_019"]},
    {"type": "multi", "question": "පිදුරංගල නැගීම සහ සීගිරි පර්වතය නැගීම ගැන තොරතුරු දෙන්න.", "expected": ["sigiriya_013", "sigiriya_003"]},
    {"type": "multi", "question": "සීගිරියේ ගල් උද්‍යානය සහ ජල උද්‍යානය අතර වෙනස්කම් විස්තර කරන්න.", "expected": ["sigiriya_008", "sigiriya_006"]},
]

GRAPH_QUERY = """
UNWIND $ids AS chunk_id
MATCH (c:Chunk {id: chunk_id})
WHERE c.destination = 'Sigiriya'
OPTIONAL MATCH (a:Attraction)-[:MENTIONS]-(c)
RETURN c.id AS chunk_id,
       c.heading AS heading,
       c.text AS evidence,
       collect(DISTINCT a.name) AS attractions
"""

HEADINGS_QUERY = """
MATCH (c:Chunk)
WHERE c.destination = 'Sigiriya'
RETURN c.id AS chunk_id, c.heading AS heading
"""


def clean_topic(text):
    """Remove request wording, keeping the named topic (heuristic)."""
    text = text.strip(" .?!？")
    text = re.split(
        r"\s+(?:ගැන|අතර|ලක්ෂණ|දෙකම|වෙනස්කම්|මොනවාද|කුමක්ද)\b",
        text,
        maxsplit=1,
    )[0]
    return text.strip(" .?!？")


def decompose(question):
    """Split a few common Sinhala two-topic constructions.

    Unknown constructions are left unsplit; the system makes no claim to
    understand all Sinhala grammar. Neither gold labels nor chunk IDs are used.
    """
    question = " ".join(question.strip().split())

    # "A සහ B ..." (comparison or a request about two places/topics).
    if " සහ " in question:
        left, right = question.split(" සහ ", 1)
        left, right = clean_topic(left), clean_topic(right)
        if left and right:
            context = "සීගිරියේ " if left.startswith("සීගිරියේ ") else ""
            if context and not right.startswith(("සීගිරි", "සීගිරිය", "පිදුරංගල")):
                right = context + right
            return [f"{left} ගැන විස්තර කියන්න.", f"{right} ගැන විස්තර කියන්න."]

    # "Aත් Bත් මොනවාද?" (e.g., route AND visiting time).
    m = re.match(r"^(.+?)ත්\s+(.+?)ත්\s+(?:මොනවාද|කුමක්ද|මොකක්ද)[?？.]?$", question)
    if m:
        left, right = clean_topic(m.group(1)), clean_topic(m.group(2))
        if left and right:
            if right.startswith("එහි ") and "සීගිරිය" in left:
                right = right.replace("එහි ", "සීගිරියට ", 1)
            return [f"{left} ගැන විස්තර කියන්න.", f"{right} ගැන විස්තර කියන්න."]

    return [question]


def search_index(session, index_name, vector):
    if index_name not in (TEXT_INDEX, HEADING_INDEX):
        raise ValueError("Unknown vector index")
    cypher = f"""
    MATCH (c:Chunk)
    SEARCH c IN (
        VECTOR INDEX {index_name}
        FOR $vector
        LIMIT 22
    ) SCORE AS similarity
    WITH c, similarity
    WHERE c.destination = 'Sigiriya'
    RETURN c.id AS chunk_id, similarity
    ORDER BY similarity DESC
    """
    return [r["chunk_id"] for r in session.run(cypher, vector=vector)]


def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (k + rank)
    return [chunk_id for chunk_id, _ in sorted(
        scores.items(), key=lambda item: (-item[1], item[0])
    )]


def fused_search(session, model, question):
    vector = model.encode("query: " + question, normalize_embeddings=True).tolist()
    text_ranks = search_index(session, TEXT_INDEX, vector)
    heading_ranks = search_index(session, HEADING_INDEX, vector)
    return reciprocal_rank_fusion([text_ranks, heading_ranks])


def heading_match_score(subquestion, heading):
    """Small, non-gold lexical heading preference for a sub-question."""
    heading = re.sub(r"^\s*\d+\.\s*", "", heading or "").strip(" .?!؟")
    if not heading:
        return 0
    if heading in subquestion:
        return 2
    # Require at least TWO matching title words for partial matching.
    words = [w for w in heading.split() if len(w) >= 3]
    hits = sum(w in subquestion for w in words)
    return 1 if hits >= 2 else 0


def prefer_heading_match(ranking, question, headings):
    """Prefer an explicitly named source heading; otherwise keep vector order.

    Only the top 8 fused candidates are considered for the preference.
    All other candidates retain their original order.
    """
    first = ranking[:8]
    scored = [
        (heading_match_score(question, headings.get(cid, "")), rank, cid)
        for rank, cid in enumerate(first)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [cid for _, _, cid in scored] + ranking[8:]


def balanced_merge(ranked_lists):
    """Round-robin merge: give each sub-question a chance in the top three."""
    output = []
    seen = set()
    for depth in range(max((len(r) for r in ranked_lists), default=0)):
        for ranking in ranked_lists:
            if depth < len(ranking) and ranking[depth] not in seen:
                cid = ranking[depth]
                seen.add(cid)
                output.append(cid)
    return output


def multi_query_search(session, model, question, headings):
    subquestions = decompose(question)
    if len(subquestions) == 1:
        return subquestions, fused_search(session, model, question)

    per_topic_rankings = []
    for subquestion in subquestions:
        ranking = fused_search(session, model, subquestion)
        per_topic_rankings.append(
            prefer_heading_match(ranking, subquestion, headings)
        )
    return subquestions, balanced_merge(per_topic_rankings)


def get_evidence(session, chunk_ids):
    records = session.run(GRAPH_QUERY, ids=chunk_ids)
    by_id = {r["chunk_id"]: dict(r) for r in records}
    return [by_id[cid] for cid in chunk_ids if cid in by_id]


def scores_at_3(expected, ranking):
    expected = set(expected)
    found = len(expected.intersection(ranking[:TOP_K]))
    return int(found == len(expected)), found / len(expected)


def print_evidence(session, ranking):
    for i, evidence in enumerate(get_evidence(session, ranking[:TOP_K]), 1):
        snippet = " ".join((evidence["evidence"] or "").split())[:140]
        print(f"  {i}. {evidence['chunk_id']} | {evidence['heading']}")
        print(f"     Attractions: {evidence['attractions']}")
        print(f"     Evidence: {snippet}")


def run_evaluation(session, model, headings):
    results = []
    for i, test in enumerate(TESTS, 1):
        q, gold = test["question"], test["expected"]
        original = fused_search(session, model, q)
        subquestions, enhanced = multi_query_search(session, model, q, headings)

        print("\n" + "=" * 66)
        print(f"Q{i} [{test['type']}]: {q}")
        print("Expected:", gold)
        print("Generated sub-questions:")
        for n, part in enumerate(subquestions, 1):
            print(f"   {n}. {part}")

        for name, ranking in (("Original RRF", original),
                              ("Decomposed + Balanced RRF", enhanced)):
            all3, recall3 = scores_at_3(gold, ranking)
            print(f"{name}: {ranking[:3]}")
            print(f"  All-required@3: {all3}, Recall@3: {recall3:.2f}")
            results.append({
                "q_id": i, "type": test["type"], "question": q,
                "method": name, "expected": ";".join(gold),
                "subquestions": " | ".join(subquestions),
                "top3": ";".join(ranking[:3]),
                "all_required_at_3": all3, "recall_at_3": recall3,
                "top1_correct": int(test["type"] == "single" and bool(ranking) and ranking[0] == gold[0]),
            })
        print("Neo4j graph evidence (enhanced top 3):")
        print_evidence(session, enhanced)

    print("\n" + "=" * 66)
    print("DEVELOPMENT CHALLENGE COMPARISON - 10 SUPPORTED QUESTIONS")
    for typ in ("single", "multi"):
        for method in ("Original RRF", "Decomposed + Balanced RRF"):
            subset = [r for r in results if r["type"] == typ and r["method"] == method]
            count = len(subset)
            all3 = sum(r["all_required_at_3"] for r in subset)
            mean_recall = sum(r["recall_at_3"] for r in subset) / count
            print(f"{typ} | {method}: All-required@3 = {all3}/{count}; "
                  f"Recall@3 = {mean_recall * 100:.1f}%", end="")
            if typ == "single":
                print(f"; Top-1 = {sum(r['top1_correct'] for r in subset)}/{count}")
            else:
                print()

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("\nSaved CSV:", OUTPUT_CSV)
    print("Unsupported / live-data questions are NOT handled by this script.")
    print("These 10 development questions are not an independent held-out test.")
    print("Neo4j database and indexes were not modified.")


def main():
    parser = argparse.ArgumentParser(description="Sigiriya multi-query experiment")
    parser.add_argument("--question", help="Optional Sinhala question to try interactively")
    args = parser.parse_args()

    password = getpass("Neo4j password: ")
    with GraphDatabase.driver(URI, auth=(USERNAME, password)) as driver:
        driver.verify_connectivity()
        print("Connected to Neo4j.")
        print("Loading multilingual E5 model...")
        model = SentenceTransformer(MODEL_NAME)

        with driver.session(database=DATABASE) as session:
            headings = {r["chunk_id"]: r["heading"] for r in session.run(HEADINGS_QUERY)}
            if args.question:
                parts, ranking = multi_query_search(session, model, args.question, headings)
                print("\nSub-questions:", *parts, sep="\n - ")
                print("\nTop 3:", ranking[:3])
                print("Neo4j source evidence:")
                print_evidence(session, ranking)
                print("\nEvidence only; no answerability check or answer generation.")
            else:
                run_evaluation(session, model, headings)


if __name__ == "__main__":
    main()
