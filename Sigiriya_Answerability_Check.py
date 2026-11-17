"""Conservative answerability gate for the Sigiriya Sinhala Hybrid GraphRAG.

Purpose
-------
1. Block recognizable questions requiring live/current/future information that
   cannot be verified from the static Sigiriya source document.
2. For all other questions, show multi-query RRF + Neo4j evidence for REVIEW.
   Retrieved evidence is NOT automatically certified as a correct answer.
3. Report correct live-query rejection and false rejections on a small
   development set; do not claim generalization from these development tests.

Reuses Sigiriya_Multi_Query_Retrieval.py WITHOUT changing it, the Neo4j graph,
existing embeddings, or vector indexes. Place both Python files in Downloads.

Run:
    python Sigiriya_Answerability_Check.py
    python Sigiriya_Answerability_Check.py --question "හෙට සීගිරියේ කාලගුණය කොහොමද?"

Requires: neo4j, sentence-transformers (already used by previous scripts).
"""

import argparse
import csv
from getpass import getpass
from pathlib import Path

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

try:
    from Sigiriya_Multi_Query_Retrieval import (
        URI, USERNAME, DATABASE, MODEL_NAME, HEADINGS_QUERY,
        TESTS as SUPPORTED_DEV_TESTS,
        multi_query_search, get_evidence,
    )
except ImportError as exc:
    raise SystemExit(
        "Could not import the previous working script. Put "
        "Sigiriya_Multi_Query_Retrieval.py in the SAME folder as this file, "
        "and make sure neo4j and sentence-transformers are installed. "
        f"Details: {exc}"
    ) from exc

OUTPUT_CSV = Path(__file__).with_name("Sigiriya_Answerability_Results.csv")

# These terms are indicators, not a general Sinhala-language classifier.
# Requiring a live-time expression AND a changing fact lowers false rejections.
LIVE_TIME_MARKERS = (
    "අද", "හෙට", "අනිද්දා", "දැන්", "මේ මොහොතේ", "මේ වෙලාවේ",
    "මේ වේලාවේ", "මේ දවස්වල", "ඉදිරි සතියේ", "ලබන සතියේ",
    "මෙම සතියේ", "මේ සතියේ", "ඉදිරි දින", "මෙම මාසයේ",
    "මේ මාසයේ", "නවතම", "අලුත්ම", "යාවත්කාලීන", "සජීවී",
    "තත්කාලීන", "current", "today", "tomorrow", "right now",
    "this week", "next week", "live", "real-time", "realtime",
)

# Facts that can change after a static tourism document was written.
CHANGING_FACT_MARKERS = (
    "කාලගුණ", "අනාවැකි", "අනාවැකිය", "උෂ්ණත්ව", "වැසි",
    "පෝලිම", "පෝලිමේ", "ජනගහනය", "සංචාරකයන්", "සංචාරකයින්",
    "ගණන", "සංඛ්‍යාව", "වසා", "වැසී", "විවෘත", "නිවාඩු",
    "ප්‍රවේශපත්‍ර", "ඇතුල්වීම", "ගාස්තු", "මිල", "ආසන",
    "පවතින තත්ත්ව", "බස් වේල", "දුම්රිය වේල",
    "weather", "forecast", "temperature", "queue", "crowd",
    "visitors", "visitor count", "closures", "closed", "open now",
    "availability", "ticket price", "traffic",
)

REJECTED = "REJECT_LIVE_DATA"
REVIEW = "REVIEW_SOURCE_EVIDENCE"

REJECTION_MESSAGE_SI = (
    "මෙම ප්‍රශ්නයට අවශ්‍ය වර්තමාන/අනාගත තොරතුරු "
    "සීගිරිය පිළිබඳ ගබඩා කළ ලේඛනයෙන් තහවුරු කළ නොහැක. "
    "යාවත්කාලීන විශ්වාසදායක මූලාශ්‍රයක් පරීක්ෂා කරන්න."
)

REVIEW_MESSAGE_SI = (
    "පහත ලැබී ඇති ලේඛන කොටස් අදාළ විය හැක. "
    "ප්‍රශ්නයට අවශ්‍ය නිශ්චිත තොරතුරු ඒවායේ තිබේදැයි "
    "තහවුරු කර පසුව පමණක් පිළිතුරක් ලබා දෙන්න."
)

# Four previously observed unsupported/live-data questions.
# Labels are for scoring ONLY; gate() never reads them.
UNSUPPORTED_LIVE_DEV_TESTS = [
    "අද සීගිරියට පැමිණි සංචාරකයන්ගේ නිශ්චිත ගණන කීයද?",
    "හෙට සීගිරියේ පැයෙන් පැයට කාලගුණ අනාවැකිය මොකක්ද?",
    "සීගිරියට මේ මොහොතේ ඇති ප්‍රවේශපත්‍ර පෝලිමේ සිටින පුද්ගලයන් ගණන කීයද?",
    "ඉදිරි සතියේ සීගිරි කෞතුකාගාරය වසා තබන විශේෂ දිනයන් මොනවාද?",
]


def answerability_gate(question):
    """Return a conservative gate decision WITHOUT using test labels/gold IDs.

    Reject only time-sensitive changing facts. For all other questions, ask for
    evidence review rather than claiming that semantic similarity proves facts.
    """
    text = " ".join(question.lower().split())
    time_hit = next((part for part in LIVE_TIME_MARKERS if part in text), None)
    fact_hit = next((part for part in CHANGING_FACT_MARKERS if part in text), None)
    if time_hit and fact_hit:
        return {
            "decision": REJECTED,
            "reason": (
                "Static-document scope: changing live fact; "
                f"time marker={time_hit!r}, fact marker={fact_hit!r}"
            ),
        }
    return {
        "decision": REVIEW,
        "reason": (
            "Not rejected by the limited live-data rule. "
            "Source evidence must still be verified; no automatic answerability proof."
        ),
    }


def retrieve_for_review(session, model, headings, question):
    """Use the exact same multi-query retrieval as the working prior script."""
    subquestions, ranking = multi_query_search(session, model, question, headings)
    evidence = get_evidence(session, ranking[:3])
    return subquestions, ranking[:3], evidence


def print_evidence(evidence):
    for i, row in enumerate(evidence, 1):
        snippet = " ".join((row.get("evidence") or "").split())[:250]
        print(f"  {i}. {row['chunk_id']} | {row.get('heading')}")
        print(f"     Linked attractions: {row.get('attractions', [])}")
        print(f"     Source preview: {snippet}")


def evaluate_one(question, session, model, headings, *, test_type=None,
                 expected_chunks=None, index=None):
    """Evaluate a question; expected_chunks are never passed to the gate/retriever."""
    gate = answerability_gate(question)
    decision = gate["decision"]

    print("\n" + "=" * 70)
    if index is not None:
        print(f"Q{index} [{test_type}]: {question}")
    else:
        print("Sinhala question:", question)
    print("Decision:", decision)
    print("Reason:", gate["reason"])

    record = {
        "q_id": index if index is not None else "",
        "test_type": test_type or "user_question",
        "question": question,
        "decision": decision,
        "reason": gate["reason"],
        "expected_chunks_eval_only": ";".join(expected_chunks or []),
        "generated_subquestions": "",
        "retrieved_top3": "",
        "all_required_at_3_eval_only": "",
        "recall_at_3_eval_only": "",
    }

    if decision == REJECTED:
        print("Response:", REJECTION_MESSAGE_SI)
        print("No vector search performed for this rejected live-data query.")
        return record

    subquestions, top3, evidence = retrieve_for_review(
        session, model, headings, question
    )
    record["generated_subquestions"] = " | ".join(subquestions)
    record["retrieved_top3"] = ";".join(top3)
    print("Message:", REVIEW_MESSAGE_SI)
    print("Sub-questions:")
    for i, subquestion in enumerate(subquestions, 1):
        print(f"  {i}. {subquestion}")
    print("Retrieved top 3:", top3)
    print("Neo4j graph evidence (for human/source verification):")
    print_evidence(evidence)

    if expected_chunks:
        gold = set(expected_chunks)
        hits = len(gold.intersection(top3))
        record["all_required_at_3_eval_only"] = int(hits == len(gold))
        record["recall_at_3_eval_only"] = f"{hits / len(gold):.3f}"
        print(f"Evaluation only: {hits}/{len(gold)} gold chunks in top 3")

    print("NOT an automatically verified answer; no answer text is generated.")
    return record


def run_development_evaluation(session, model, headings):
    tests = [
        {"kind": "supported", "question": item["question"],
         "expected": item["expected"]}
        for item in SUPPORTED_DEV_TESTS
    ] + [
        {"kind": "unsupported_live", "question": q, "expected": []}
        for q in UNSUPPORTED_LIVE_DEV_TESTS
    ]

    records = []
    for i, t in enumerate(tests, 1):
        records.append(evaluate_one(
            t["question"], session, model, headings,
            test_type=t["kind"], expected_chunks=t["expected"], index=i,
        ))

    supported = [r for r in records if r["test_type"] == "supported"]
    unsupported = [r for r in records if r["test_type"] == "unsupported_live"]
    false_rejects = sum(r["decision"] == REJECTED for r in supported)
    correct_live_rejects = sum(r["decision"] == REJECTED for r in unsupported)
    evidence_review = [r for r in supported if r["decision"] == REVIEW]
    covered = sum(r["all_required_at_3_eval_only"] == 1 for r in evidence_review)

    print("\n" + "=" * 70)
    print("CONSERVATIVE ANSWERABILITY GATE - DEVELOPMENT SUMMARY")
    print(f"Known unsupported live questions rejected: "
          f"{correct_live_rejects}/{len(unsupported)}")
    print(f"Known supported questions incorrectly rejected: "
          f"{false_rejects}/{len(supported)}")
    print(f"Supported questions sent for evidence review: "
          f"{len(evidence_review)}/{len(supported)}")
    print(f"Among reviewed supported questions, all gold chunks found @3: "
          f"{covered}/{len(evidence_review)}")
    print("CAUTION: The gate is only a narrow rule for dynamic facts, not a "
          "general answerability classifier.")
    print("CAUTION: REVIEW_SOURCE_EVIDENCE means evidence to verify, NOT "
          "that the question is definitely answerable.")
    print("CAUTION: These are development examples already seen in prior tests; "
          "results are not held-out accuracy.")

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("Saved CSV:", OUTPUT_CSV)
    print("Completed: no Neo4j nodes, vectors, or indexes were modified.")


def main():
    parser = argparse.ArgumentParser(
        description="Conservative answerability gate on Sigiriya Hybrid GraphRAG"
    )
    parser.add_argument("--question", type=str,
                        help="One custom Sinhala question instead of evaluation set")
    args = parser.parse_args()

    # A live-data custom question can be rejected WITHOUT loading model/DB.
    if args.question and answerability_gate(args.question)["decision"] == REJECTED:
        evaluate_one(args.question, None, None, None)
        return

    password = getpass("Neo4j password: ")
    with GraphDatabase.driver(URI, auth=(USERNAME, password)) as driver:
        driver.verify_connectivity()
        print("Connected to Neo4j (read-only evaluation).")
        print("Loading multilingual E5 model...")
        model = SentenceTransformer(MODEL_NAME)
        with driver.session(database=DATABASE) as session:
            headings = {
                row["chunk_id"]: row["heading"]
                for row in session.run(HEADINGS_QUERY)
            }
            if args.question:
                evaluate_one(args.question, session, model, headings)
            else:
                run_development_evaluation(session, model, headings)


if __name__ == "__main__":
    main()
