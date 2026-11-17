# -*- coding: utf-8 -*-
"""Sigiriya Hybrid GraphRAG — batch answers with source evidence.

This file adds a user-facing response layer to the working 22-chunk Sigiriya
Neo4j retrieval system. It DOES NOT modify Neo4j, its embeddings or indexes.

OUTPUT FORMAT
-------------
ප්‍රශ්නය: ...
පිළිතුර: <a direct Sinhala sentence, or an explicit inability to verify>
මූලාශ්‍ර සාක්ෂි:
  [1] <chunk id> — <heading>
      <exact excerpt from the stored source>

By default the response is EXTRACTIVE: factual sentences are copied from the
stored Sinhala source and presented as a concise answer. Narrow person/king
requests use a grounded template extracted from an explicit source sentence. This is not an LLM
and cannot reliably paraphrase, infer missing facts or establish all answerability.
In doubtful cases it says that the source does not establish an answer.

Dependencies: the two earlier working scripts must be in the same folder:
  Sigiriya_Multi_Query_Retrieval.py
  Sigiriya_Answerability_Check.py
Install neo4j and sentence-transformers as for those programs.

Run:
  python Sigiriya_Retrieval_Improvement_Batch.py
  # Paste all questions, one per line; enter END on the last line.
  python Sigiriya_Retrieval_Improvement_Batch.py --file Sinhala_Questions.txt
  python Sigiriya_Retrieval_Improvement_Batch.py --demo
  python Sigiriya_Retrieval_Improvement_Batch.py --question "..."

No evaluation gold IDs are used for retrieval, selecting evidence or answers.
"""

import argparse
import csv
import io
import json
import math
import re
from collections import Counter
from getpass import getpass
from pathlib import Path

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from Sigiriya_Multi_Query_Retrieval import (
    URI, USERNAME, DATABASE, MODEL_NAME, HEADINGS_QUERY,
    fused_search, multi_query_search, decompose,
)
from Sigiriya_Answerability_Check import (
    answerability_gate, REJECTED, REJECTION_MESSAGE_SI,
    LIVE_TIME_MARKERS, CHANGING_FACT_MARKERS,
)

OUTPUT_JSON = Path(__file__).with_name("Sigiriya_Batch_Answers_Reviewed.json")
OUTPUT_TEXT = Path(__file__).with_name("Sigiriya_Batch_Answers_Reviewed.txt")
OUTPUT_CSV = Path(__file__).with_name("Sigiriya_Batch_Answers_Reviewed.csv")
SOURCE_QUERY = """
MATCH (c:Chunk)
WHERE c.destination = 'Sigiriya'
OPTIONAL MATCH (a:Attraction)-[:MENTIONS]-(c)
RETURN c.id AS chunk_id,
       c.heading AS heading,
       c.text AS evidence,
       collect(DISTINCT a.name) AS attractions
ORDER BY c.id
"""

NO_SOURCE_SI = (
    "ලබාදී ඇති සීගිරිය ලේඛනයෙන් මෙම ප්‍රශ්නයට අවශ්‍ය "
    "නිශ්චිත පිළිතුර තහවුරු කළ නොහැක."
)
NO_LIVE_SI = (
    "මෙම ප්‍රශ්නයට අවශ්‍ය යාවත්කාලීන තොරතුරු "
    "සීගිරිය පිළිබඳ ගබඩා කළ ලේඛනයෙන් තහවුරු කළ නොහැක. "
    "විශ්වාසදායක නවතම මූලාශ්‍රයක් බලන්න."
)

# General Sinhala stopwords and prompt boilerplate, not topic labels / gold IDs.
STOPWORDS = {
    "සීගිරිය", "සීගිරි", "සීගිරියේ", "සීගිරියට", "සීගිරියෙන්",
    "ගැන", "කියන්න", "විස්තර", "දෙන්න", "පැහැදිලි", "කරන්න",
    "කුමක්ද", "කොහෙද", "කොතැනද", "කෙසේද", "ඇයි", "මොනවාද",
    "ඇති", "තිබෙන", "තිබේද", "එහි", "මෙහි", "සහ", "හා", "අතර",
    "දෙකම", "එක", "අය", "වන්නේ", "කුමන", "පිළිබඳ", "පමණ",
    "තොරතුරු", "කොහොමද", "මොකක්ද", "ලෙස", "වැනි", "වෙන", "මොන",
    "ලැබෙන", "යුතු", "කළ", "එය", "මෙම", "නම්", "කියන", "ද", "දී",
}

# Extra dynamic concepts that the earlier gate did not cover on the 30-Q test.
# These are broad expressions that inherently need up-to-date information.
MORE_CHANGING_TERMS = (
    "වාහන තදබද", "මාර්ග තදබද", "ට්‍රැෆික්", "ගමනාගමන තත්ත්ව",
    "කාමර ලබාගන්න", "කාමර තිබෙන", "කාමර වෙන්කර", "හෝටල් කාමර",
    "නවාතැන් ලබාගන්න", "booking", "hotel availability", "room availability",
    "traffic jam", "traffic congestion", "available rooms", "book a room",
)

TOKEN_RE = re.compile(r"[\u0D80-\u0DFF]+|[A-Za-z]+|\d+", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?।])\s+|\n+", re.UNICODE)
NUMERIC_RE = re.compile(r"(?<!\w)\d+(?:[.,–-]\d+)*", re.UNICODE)


def tokens(text):
    return [w for w in TOKEN_RE.findall((text or "").lower())
            if len(w) > 2 and w not in STOPWORDS and not w.isdigit()]


def approximate_match(word, candidate):
    """Limited Sinhala inflection tolerance; no full morphological analyzer."""
    if word == candidate:
        return True
    # Require a fairly specific common prefix to avoid matching short words.
    n = min(len(word), len(candidate))
    return n >= 5 and word[:max(4, n - 2)] == candidate[:max(4, n - 2)]


def overlap_score(query, passage):
    q = set(tokens(query))
    if not q:
        return 0.0
    p = set(tokens(passage))
    if not p:
        return 0.0
    matches = sum(any(approximate_match(w, x) for x in p) for w in q)
    # A lightly normalized coverage score; long documents do not get rewarded
    # merely for having more words.
    return matches / len(q)


def split_sentences(text):
    chunks = SENTENCE_RE.split(text or "")
    return [x.strip(" \r\n\t-•") for x in chunks
            if len(x.strip(" \r\n\t-•")) >= 12]


def heading_clean(heading):
    return re.sub(r"^\s*\d+\.\s*", "", heading or "").strip()


def best_sentence(question, source):
    """Select a verbatim answer sentence from source, never a generated fact."""
    sentences = split_sentences(source.get("evidence") or "")
    if not sentences:
        return "", 0.0
    scored = []
    for i, sent in enumerate(sentences):
        coverage = overlap_score(question, sent)
        score = coverage
        # Prefer compact, specific sentences over long passages.
        score -= min(len(sent) / 2500, 0.13)
        # A 'what is shown/depicted?' question needs pictured CONTENT,
        # not the generic statement that there are famous paintings.
        # This uses source-language evidence rather than any gold chunk ID.
        art_q = (
            any(w in question for w in ("බිතුසිතුවම්", "සිතුවම්", "චිත්‍ර"))
            and any(w in question for w in (
                "දැකිය හැක්කේ", "දැකගත හැක්කේ", "මොනවාද",
                "නිරූපණ", "ඇඳ", "රූප",
            ))
        )
        if art_q:
            if "රූප" in sent and any(w in sent for w in ("සිතුවම්", "චිත්‍ර")):
                score += 1.5
            if any(w in sent for w in ("ප්‍රසිද්ධම අංගයක්", "වැදගත් වේ")):
                score -= 0.6
            if "පිහිටි" in sent and "රූප" not in sent:
                score -= 0.25
        if ("ඇයි" in question or "හේතුව" in question) and any(
            k in sent for k in ("නිසා", "හේතුව", "බැවින්", "සඳහා")
        ):
            score += 0.72
        if ("ගැන" in question or "විස්තර" in question) and len(sent) >= 70:
            score += 0.12
        if ("ගැන" in question or "විස්තර" in question) and any(
            phrase in sent for phrase in ("ප්‍රසිද්ධම අංගයක්", "වැදගත් ඓතිහාසික අංගයකි")
        ):
            score -= 0.12
        if ("වසර" in question or "අවුරුද්ද" in question) and re.search(r"\b\d{4}\b", sent):
            score += 0.16
        if "කොහෙද" in question or "කොතැනද" in question:
            if any(w in sent for w in ("පිහිටා", "දැකගත", "ප්‍රදේශ", "ගල්වල")):
                score += 0.10
        scored.append((score, coverage, i, sent))
    score, coverage, _, sent = max(scored, key=lambda x: (x[0], x[1], -x[2]))
    return sent, coverage


def select_grounded_quote(subquestion, source, preferred_sentence):
    """For open-ended descriptions quote two informative ORIGINAL sentences.

    For fact questions keep the one sentence selected by evidence matching.
    This does not create or paraphrase any factual statements.
    """
    broad = ("ගැන" in subquestion or "විස්තර" in subquestion)
    narrow = any(k in subquestion for k in (
        "ඇයි", "හේතුව", "කීය", "කුමක්ද", "කොහෙද", "කොතැනද",
        "නම", "වසර", "අවුරුද්ද", "කෙසේද", "කොහොමද",
        "ආරක්ෂා කිරීම", "තහනම්"
    ))
    if not broad or narrow:
        return preferred_sentence
    sentences = split_sentences(source.get("evidence") or "")
    if not sentences:
        return preferred_sentence
    # Exclude only generic opener sentences, NOT all short sentences.
    generic_opening = (
        "ප්‍රසිද්ධම අංගයක්", "තවත් වැදගත් ඓතිහාසික අංගයකි",
        "තවත් පුරාවිද්‍යාත්මක අංගයකි", "වැදගත් ඓතිහාසික සහ පුරාවිද්‍යාත්මක උරුමයකි",
    )
    if len(sentences) >= 3 and any(g in sentences[0] for g in generic_opening):
        sentences = sentences[1:]
    chosen = [x for x in sentences if len(x) >= 32][:2]
    return " ".join(chosen) if chosen else preferred_sentence


def is_numeric_or_name_request(question):
    q = question
    # Narrow check: if a precise number/person is requested, do not invent it.
    if "වසර" in q or "අවුරුද්ද" in q:
        return "year"
    if ("මීටර්" in q and ("කීය" in q or "උස" in q or "ගැඹුර" in q)):
        return "metres"
    if "ගැඹුර" in q:
        return "metres"
    if "පඩි කීය" in q or "පඩිපෙළ කීය" in q:
        return "count"
    if "කීයක්" in q or "ගණන කීය" in q:
        return "count"
    if "සම්පූර්ණ නම" in q or ("නම" in q and any(
        x in q for x in ("ඉංජිනේරු", "චිත්‍ර ශිල්පි", "නිර්මාණ ශිල්පි")
    )):
        return "specific_person"
    return None


def specific_detail_supported(question, quote):
    """Conservative *negative* check for some missing exact facts.

    Passing this check is NOT semantic proof that a sentence answers a question.
    """
    request = is_numeric_or_name_request(question)
    if request is None:
        return True
    if request == "year":
        return bool(re.search(r"(?<!\d)\d{4}(?!\d)", quote))
    if request == "metres":
        return ("මීටර්" in quote or "meter" in quote.lower()) and bool(NUMERIC_RE.search(quote))
    if request == "count":
        # Accept both numeric and spelled Sinhala quantity when explicit.
        return bool(NUMERIC_RE.search(quote) or re.search(r"(?:එක|දෙක|තුන|හතර|පහ|හය|හත|අට|නවය|දහය)(?:ක්|කි|කට)?\b", quote))
    if request == "specific_person":
        # This simple program cannot reliably validate a person's proper name.
        # Safer to decline than guess a name from a historical description.
        return False
    return False


def improved_gate(question):
    """Extend previous narrow dynamic-data rules; no hardcoded test answers."""
    old = answerability_gate(question)
    if old["decision"] == REJECTED:
        return {"rejected": True, "reason": old["reason"]}
    low = " ".join(question.lower().split())
    time_marker = next((x for x in LIVE_TIME_MARKERS if x in low), None)
    new_term = next((x for x in MORE_CHANGING_TERMS if x in low), None)
    if time_marker and new_term:
        return {"rejected": True, "reason": f"Live time={time_marker!r}, dynamic information={new_term!r}"}
    return {"rejected": False, "reason": "Retrieve source evidence and check whether it supports a direct answer."}


def load_sources(session):
    records = [dict(r) for r in session.run(SOURCE_QUERY)]
    if not records:
        raise RuntimeError("No Sigiriya :Chunk nodes found. Check your Neo4j database.")
    ids = [r["chunk_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate source chunk IDs detected in Neo4j.")
    return {r["chunk_id"]: r for r in records}


def candidate_scores(question, sources, whole_ranking, per_question_ranking,
                     extra_rankings=()):
    """Merge two semantic systems with source-lexical evidence.

    Scores are computed from question text + retrieved source ONLY. No test IDs.
    This is heuristic candidate ranking, not a calibrated confidence score.
    """
    ranks = [whole_ranking, per_question_ranking, *extra_rankings]
    seen = set()
    for ranking in ranks:
        seen.update(ranking[:12])
    # For this small current corpus, lexical matching across all existing
    # chunks also recovers rare explicit facts even when vector rank is poor.
    seen.update(sources)
    score_rows = []
    for cid in seen:
        source = sources.get(cid)
        if source is None:
            continue
        heading = heading_clean(source.get("heading", ""))
        quote, coverage = best_sentence(question, source)
        lex = max(coverage, overlap_score(question, heading) * 0.95)
        sem = 0.0
        for i, ranking in enumerate(ranks):
            if cid in ranking:
                pos = ranking.index(cid) + 1
                sem += (1.0 if i < 2 else 0.4) * 60.0 / (60.0 + pos)
        # Topic-specific lexical evidence matters more than generic proximity.
        title_bonus = 0.45 if heading and heading in question else 0
        score = 1.4 * lex + 0.20 * sem + title_bonus
        score_rows.append((score, lex, coverage, cid, quote))
    score_rows.sort(key=lambda r: (-r[0], -r[1], r[3]))
    return score_rows


def choose_evidence(question, sources, original_rank, enhanced_rank, session, model):
    """Select at most one main source per detected subquestion.

    For combined questions this keeps evidence from each topic, rather than
    letting one topic fill all top positions. No gold IDs participate.
    """
    subquestions = decompose(question)
    if len(subquestions) > 3:
        subquestions = subquestions[:3]
    selected = []
    all_scores = []
    for sub in subquestions:
        # For unsplit questions reuse the original search to avoid extra queries.
        ranking = original_rank if len(subquestions) == 1 else fused_search(session, model, sub)
        scored = candidate_scores(sub, sources, original_rank, ranking, (enhanced_rank,))
        all_scores.append(scored)
        # The same source may contain answers to BOTH topics; do not force
        # an unrelated second chunk merely to make the IDs different.
        choice = scored[0] if scored else None
        if choice:
            score, lex, cov, cid, quote = choice
            quote = select_grounded_quote(sub, sources[cid], quote)
            selected.append({"chunk_id": cid, "heading": sources[cid].get("heading", ""),
                             "quote": quote, "coverage": cov, "lex_score": lex,
                             "ranking_score": score, "subquestion": sub,
                             "attractions": sources[cid].get("attractions", [])})
    # For a single-topic query, one complete source sentence is more useful
    # than padding with unrelated nearest-neighbour evidence.
    return subquestions, selected, all_scores


def verify_selection(question, subquestions, selected):
    """Return cautious status, not a proof of correctness.

    At least one substantive query token must be reflected in each quote;
    questions seeking an exact absent datum are rejected without invention.
    """
    if not selected or len(selected) < len(subquestions):
        return False, "No matching document evidence was selected."
    for item in selected:
        if not item["quote"]:
            return False, f"No source sentence for: {item['subquestion']}"
        # Sinhala pronouns may refer to the chunk heading. For example,
        # the frescoes sentence starts 'මෙම සිතුවම්වල...' while the query
        # says 'සීගිරි බිතුසිතුවම්වල...'. Word overlap is then zero despite
        # the exact sentence answering what is depicted. Accept this *narrow*
        # evidence case only when the heading confirms the artwork topic and
        # the quote expressly states that figures are depicted.
        content_q = any(x in item["subquestion"] for x in (
            "දැකිය හැක්කේ", "දැකගත හැක්කේ", "මොනවාද", "නිරූපණ",
        ))
        artwork_q = any(x in item["subquestion"] for x in ("බිතුසිතුවම්", "සිතුවම්"))
        artwork_source = any(x in item["heading"] for x in ("බිතුසිතුවම්", "සිතුවම්"))
        depicted_quote = "රූප" in item["quote"] and any(
            x in item["quote"] for x in ("සිතුවම්", "චිත්‍ර")
        )
        artwork_content_supported = content_q and artwork_q and artwork_source and depicted_quote
        if item["coverage"] < 0.22 and not artwork_content_supported:
            return False, f"Weak textual evidence for: {item['subquestion']}"
        if not specific_detail_supported(item["subquestion"], item["quote"]):
            return False, f"Requested exact detail is not confirmed for: {item['subquestion']}"
    # For unsplit queries, ensure exact-detail requirement is tested on original.
    if len(subquestions) == 1 and not specific_detail_supported(question, selected[0]["quote"]):
        return False, "The source does not establish the requested exact value/name."
    return True, "Relevant exact source sentences identified (not independent factual verification)."



# ---- Intent-aware person / king extraction -------------------------------
# A relevant chunk is not, by itself, a valid answer. These checks prevent
# a question asking WHO from being answered with an unrelated construction
# sentence merely because that sentence shares the word "built".
WHO_MARKERS = (
    "කවුද", "කවුරුද", "කවුරුන්ද", "කවුරුන්", "කවරෙක්ද",
    "කුමන රජ", "කවර රජ", "කුමන පුද්ගල", "කවර පුද්ගල",
)
KING_MARKERS = ("රජතුමා", "රජු", "පාලකයා", "පාලකයාද", "king")
KING_PATTERN = re.compile(r"([\u0D80-\u0DFF\u200C\u200D]{2,30})\s+රජු(?!ගේ)")


def is_who_question(question):
    q = question.lower()
    return any(marker in q for marker in WHO_MARKERS)


def is_sigiriya_king_builder_question(question):
    """Recognize a *narrow* question the history chunk can support.

    This does not assume a particular king's name or chunk ID. We intentionally
    do not treat building the museum or painting the frescoes as the same act as
    making Sigiriya a royal capital.
    """
    q = question.lower()
    if not is_who_question(q):
        return False
    if any(other in q for other in (
        "කෞතුකාගාර", "බිතුසිතුවම්", "කැටපත්", "ගුහාව",
        "ජල උද්‍යාන", "ගල් උද්‍යානය", "චිත්‍ර ශිල්පි", "ඉංජිනේරු",
    )):
        return False
    king_role = any(x in q for x in KING_MARKERS)
    royal_action = any(x in q for x in (
        "ඉදික", "ඉදි කළ", "සංවර්ධනය", "ගොඩනැ", "අගනුවර",
        "රාජධානිය", "පිහිටුවා", "නිර්මාණය", "පාලනය",
    ))
    # Requiring the site OR capital context helps avoid unrelated king questions.
    site_or_capital = ("සීගිරි" in q or "සීගිරිය" in q or "අගනුවර" in q or "රාජධානිය" in q)
    return site_or_capital and royal_action and king_role


def extract_king_builder_evidence(source_row):
    """Return (king-name, verbatim historical evidence) if *both* occur.

    Require the king, Sigiriya, and development of the royal city to appear
    within the SAME source sentence. Mere presence of a king's name elsewhere
    in a chunk is not sufficient.
    """
    for sentence in split_sentences(source_row.get("evidence") or ""):
        if "සීගිරි" not in sentence:
            continue
        if "සංවර්ධනය" not in sentence or "අගනුවර" not in sentence:
            continue
        match = KING_PATTERN.search(sentence)
        if match:
            return match.group(1), sentence
    return None


def answer_who_from_verified_role(question, sources, original_rank, enhanced_rank):
    """Build a direct person answer only when a matching source role exists.

    Unsupported WHO requests explicitly abstain. This is a narrow rule-based
    extractor, not an all-purpose Sinhala person question-answering model.
    """
    if not is_sigiriya_king_builder_question(question):
        return None
    # Preserve semantic retrieval as the primary shortlist. Search the small
    # 22-chunk collection only as a recovery step, never using gold IDs.
    candidate_ids = list(dict.fromkeys(
        list(original_rank[:12]) + list(enhanced_rank[:12]) + list(sources)
    ))
    for cid in candidate_ids:
        row = sources.get(cid)
        if not row:
            continue
        match = extract_king_builder_evidence(row)
        if not match:
            continue
        king_name, quote = match
        answer = f"සීගිරිය රාජකීය නගරයක් ලෙස සංවර්ධනය කළේ {king_name} රජතුමාය."
        return {
            "question": question,
            "status": "SOURCE_GROUNDED_FACT_EXTRACTED",
            "answer": answer,
            "sources": [{
                "chunk_id": cid,
                "heading": row.get("heading", ""),
                "source_quote": quote,
                "attractions": row.get("attractions", []),
            }],
            "reason": "A named king and royal-city development are explicitly linked in one source sentence.",
            "subquestions": [question],
        }
    return None


def answer_question(question, session, model, sources, headings, debug=False):
    gate = improved_gate(question)
    result = {"question": question, "status": "", "answer": "", "sources": [],
              "reason": gate["reason"], "subquestions": []}
    if gate["rejected"]:
        result.update(status="LIVE_DATA_UNSUPPORTED", answer=NO_LIVE_SI)
        return result

    original_rank = fused_search(session, model, question)

    # Person questions must answer WHO, not select a coincidentally similar
    # sentence about buildings, gates, or dates. A special extraction path
    # uses only explicit actor + action evidence; otherwise abstain safely.
    if is_who_question(question):
        subqs, enhanced_rank = multi_query_search(session, model, question, headings)
        person_result = answer_who_from_verified_role(
            question, sources, original_rank, enhanced_rank
        )
        if person_result:
            if debug:
                person_result["debug"] = {
                    "original_rrf_top5": original_rank[:5],
                    "multi_query_top5": enhanced_rank[:5],
                    "intent": "WHO / named role + action",
                }
            return person_result
        result.update(
            status="INSUFFICIENT_DOCUMENT_EVIDENCE",
            answer=NO_SOURCE_SI,
            reason="A person was requested, but no matching named actor and action could be verified in the source.",
        )
        if debug:
            result["debug"] = {
                "original_rrf_top5": original_rank[:5],
                "multi_query_top5": enhanced_rank[:5],
                "intent": "WHO / no supported person-role match",
            }
        return result

    subqs, enhanced_rank = multi_query_search(session, model, question, headings)
    qs, selected, score_lists = choose_evidence(
        question, sources, original_rank, enhanced_rank, session, model
    )
    passed, reason = verify_selection(question, qs, selected)
    result["subquestions"] = qs
    if not passed:
        result.update(status="INSUFFICIENT_DOCUMENT_EVIDENCE", answer=NO_SOURCE_SI,
                      reason=reason)
        # Do not label near neighbours as supporting citations when no answer is
        # supported. Debug data can still be inspected separately.
    else:
        # This is a direct, short, EXTRACTIVE answer: each sentence is copied
        # from the source. The heading gives context without adding new facts.
        answer_pieces = []
        for i, item in enumerate(selected, 1):
            sentence = item["quote"].strip()
            location_question = any(x in question for x in ("කොහෙද", "කොහේද", "කොතැනද", "කොතැනකද"))
            title = heading_clean(item["heading"])
            if len(selected) > 1 or (location_question and title not in sentence):
                answer_pieces.append(f"{title}: {sentence}")
            else:
                answer_pieces.append(sentence)
        result["answer"] = " ".join(answer_pieces)
        result["status"] = "SOURCE_GROUNDED_EXTRACTIVE"
        result["reason"] = reason
        result["sources"] = []
        for s in selected:
            # A repeated source can supply a different sentence for another
            # subquestion; keep the quoted evidence auditable.
            src = {"chunk_id": s["chunk_id"], "heading": s["heading"],
                   "source_quote": s["quote"], "attractions": s["attractions"]}
            if src not in result["sources"]:
                result["sources"].append(src)

    if debug:
        result["debug"] = {
            "original_rrf_top5": original_rank[:5],
            "multi_query_top5": enhanced_rank[:5],
            "selected_candidates": [
                {k: v for k, v in s.items() if k in
                 ("chunk_id", "coverage", "lex_score", "ranking_score", "subquestion")}
                for s in selected
            ],
            "reason": reason,
        }
    return result



# ---------------------------------------------------------------------------
# ANSWER RELEVANCE IMPROVEMENT (v2)
# ---------------------------------------------------------------------------
# Applied after examining answer errors in a development batch. These are
# transparent domain vocabulary / intent rules, NOT a trained answer model.
# They do NOT read expected chunk IDs, use gold labels or modify Neo4j data.
# A rule configured after inspecting examples must be re-tested on unseen data.

# Each entry is a *heading substring*, followed by alternative user wordings.
# No chunk IDs are hard-coded here. The exact Neo4j heading must exist before
# a topic can be selected. Vector retrieval remains the fallback for unknown
# wordings; this is a corpus-specific heading/semantic hybrid.
TOPIC_CUES = (
    ("සීගිරියේ ඉතිහාසය", ("කාශ්‍යප", "රාජධානිය", "රාජකීය නගර", "ඉදිකළ රජ", "රජතුමා")),
    ("සීගිරි බිතුසිතුවම්", ("බිතුසිතුවම්", "සීගිරි ලලනා", "පුරාණ සිතුවම්")),
    ("කැටපත් පවුර", ("කැටපත් පවුර", "කැටපත් පවුරේ", "පැරණි අමුත්තන් කවි", "ඔප දැමූ බිත්තිය")),
    ("ජල උද්‍යාන", ("ජල උද්‍යාන", "ජල උද්‍යානය", "ජල තටාක")),
    ("දිය උල්පත් උද්‍යානය", ("දිය උල්පත්", "ජල උල්පත්", "උල්පත් ක්‍රියාත්මක", "ගුරුත්වාකර්ෂණ")),
    ("ගල් උද්‍යානය", ("ගල් උද්‍යාන", "ගල් උද්‍යානය", "ශිලා ලේඛන", "විශාල ගල් අතර", "ගලෙන් කළ ආසන")),
    ("නාග පෙන ගුහාව", ("නාග පෙන ගුහාව", "නාග පෙන ගුහාවට")),
    ("සීගිරි පර්වතය", ("සීගිරි පර්වතය", "සිංහ පාද", "සිංහ දොරටුව", "පර්වතය නැගීම")),
    ("පිදුරංගල", ("පිදුරංගල",)),
    ("සීගිරියට ළඟා වන්නේ", ("ගමන් මාර්ග", "යන මාර්ග", "සීගිරියට යන", "ප්‍රවාහන", "ළඟා වන්නේ", "කොළඹ සිට", "මහනුවර සිට")),
    ("සංචාරයට සුදුසු වේලාව", ("සුදුසු වේලාව", "සුදුසු කාලය", "උදේ කාලය", "උදෑසන", "දහවල්ට වඩා උදේ", "නැරඹීමට උදේ")),
    ("බඹරුන්ගෙන් ආරක්ෂාව", ("බඹර", "බඹරුන්", "බඹරුන්ගෙන්", "බඹර කූඩු")),
    ("ආරක්ෂාව", ("ආරක්ෂාව", "ආරක්ෂක උපදෙස්", "පාවහන්", "ලිස්සනසුලු", "වැසි සහිත අවස්ථා")),
    ("වගකීම් සහිත සංචාරක හැසිරීම", ("වගකීමෙන් හැසිර", "වගකීම් සහිත සංචාරක", "පුරාවිද්‍යාත්මක ස්ථාන මත ලිවීම")),
    ("ආහාර සහ පාන", ("කෑම බීම", "ආහාර සහ පාන", "ආහාර")),
    ("නවාතැන්", ("නවාතැන්", "රැඳී සිටීම", "හෝටල්")),
    ("සීගිරි කෞතුකාගාරය", ("කෞතුකාගාර",)),
    ("ප්‍රවේශපත්‍ර සහ සංචාරක මාර්ගෝපදේශකයන්", ("ප්‍රවේශපත්‍ර", "මාර්ගෝපදේශක")),
    ("සීගිරියේ කළ හැකි දේ", ("කළ හැකි දේ", "ක්‍රියාකාරකම්")),
    ("වෙනත් දර්ශන ස්ථාන", ("වෙනත් දර්ශන", "Melu's Rock", "Duwannagala", "Mapagala")),
    ("සීගිරිය ආසන්න වනජීවී සෆාරි", ("සෆාරි", "මින්නේරිය", "කවුඩුල්ල", "අලි රංචු")),
)


def find_topic_source(subquestion, sources):
    q = subquestion.lower()
    scored = []
    for heading_fragment, cues in TOPIC_CUES:
        rows = [r for r in sources.values() if heading_fragment in (r.get("heading") or "")]
        if not rows:
            continue
        matched = [cue for cue in cues if cue.lower() in q]
        if not matched:
            continue
        # Prefer the most informative matching phrase to generic words.
        specific = max(len(cue) for cue in matched)
        scored.append((specific, sum(len(cue) for cue in matched), heading_fragment, rows[0]))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return scored[0][3]


def intent_tags(question):
    q=question.lower()
    return {
        "art_content": any(t in q for t in ("බිතුසිතුවම්", "සිතුවම්")) and
            any(t in q for t in ("මොනවාද", "දැකිය", "නිරූපණ", "කාන්තා")),
        "old_inscriptions": any(t in q for t in ("සටහන්", "ලේඛන", "කවි")) and
            any(t in q for t in ("මොනවාද", "ඇති", "ලියූ", "දැකිය")),
        "morning_reason": any(t in q for t in ("උදේ", "උදෑසන", "සුදුසු වේලාව")),
        "travel_routes": any(t in q for t in ("ගමන් මාර්ග", "යන මාර්ග", "ප්‍රවාහන", "ළඟා වන්නේ")),
        "wasp_safety": "බඹර" in q,
        "general_safety": "ආරක්ෂාව" in q or "ආරක්ෂක උපදෙස්" in q,
        "gardens_compare": "උද්‍යාන" in q and any(t in q for t in ("වෙනස්කම්", "සසඳ", "අතර")),
        "why": "ඇයි" in q or "හේතුව" in q or "නිසා" in q,
        "where": any(t in q for t in ("කොහේද", "කොහෙද", "කොතැනද", "කොතැනකද")),
        "how_system": "ක්‍රියාත්මක" in q and any(t in q for t in ("ක්‍රමය", "කොහොමද", "කෙසේද")),
        "broad": any(t in q for t in ("විස්තර", "ගැන කියන්න", "කියන්න")),
    }


def select_answer_quote(subquestion, source):
    raw = source.get("evidence") or ""
    sentences = split_sentences(raw)
    if not sentences:
        return "", False
    tags = intent_tags(subquestion)
    q = subquestion.lower()
    heading = heading_clean(source.get("heading", ""))
    ranked=[]
    for idx, sent in enumerate(sentences):
        s=sent.lower()
        score = overlap_score(q, sent) * 2.0
        # Generic statements don't answer a specific tourism question.
        if any(z in s for z in ("ප්‍රසිද්ධම අංගයක්", "තවත් වැදගත් ඓතිහාසික අංගයකි", "තවත් පුරාවිද්‍යාත්මක අංගයකි")):
            score-=0.9
        if tags['art_content']:
            if 'කාන්තා රූප' in s or ('රූප' in s and 'සිතුවම්' in s): score+=3.4
            if 'සර්පිලාකාර' in s: score-=0.6
        if tags['old_inscriptions']:
            if 'කවි' in s and 'සටහන්' in s: score+=3.0
            if 'ලිවීම තහනම්' in s: score-=3.0
        if tags['morning_reason']:
            if 'උණුසුම' in s: score+=3.0
            if 'උදෑසන පැමිණීම' in s: score+=2.6
            if 'මඟහැර' in s: score+=1.5
            if 'කැමැත්තක්' in s: score-=0.8
        if tags['travel_routes']:
            if 'දඹුල්ල' in s and ('ගමන්' in s or 'පැමිණ' in s):score+=2.5
            if 'බස් රථ' in s:score+=1.8
            if 'මාර්ග' in s and 'සෆාරි' in s:score-=2.4
        if tags['wasp_safety']:
            if 'නිලධාරීන්ගේ උපදෙස්' in s: score+=3.2
            if 'නිශ්ශබ්ද' in s: score+=2.0
            if 'බඹරුන්ට බාධා' in s: score+=1.0
        if tags['general_safety'] and not tags['wasp_safety']:
            if 'පානීය ජල' in s:score+=2.0
            if 'පාවහන්' in s or 'ලිස්සනසුලු' in s:score+=1.9
        if tags['gardens_compare']:
            if 'ජල තටාක' in s or 'විශාල ස්වභාවික ගල්' in s:score+=2.8
        if tags['how_system']:
            if 'ගුරුත්වාකර්ෂණ' in s or 'ජල පීඩනය' in s:score+=3.0
        # A general description should show the source's identifying facts,
        # not a generic heritage statement or an unrelated restriction.
        if 'බිතුසිතුවම්' in heading:
            if 'කාන්තා රූප' in s: score+=2.8
            if 'සර්පිලාකාර' in s: score+=0.6
            if 'චිත්‍ර කලාව අධ්‍යයනය' in s: score-=1.2
        if 'කැටපත් පවුර' in heading:
            if 'කවි සහ විවිධ සටහන්' in s:score+=3.0
            if 'දිලිසෙන මතුපිට' in s:score+=1.8
            if 'ලිවීම තහනම්' in s and not ('තහනම්' in q or 'ආරක්ෂා' in q):score-=2.5
        if 'පිදුරංගල' in heading and ('නැගීම' in q or 'ගැන' in q):
            if 'අවසාන කොටසේ' in s:score+=2.0
            if 'විහාරස්ථානයක්' in s:score+=1.1
            if 'හිරු උදාව' in s and 'හිරු' not in q:score-=1.0
        if tags['where']:
            if 'ශිලා ලේඛන' in s or 'ගල්වල' in s:score+=1.8
            if 'පිහිටා' in s or 'දැකගත' in s:score+=0.5
        if tags['why']:
            if 'නිසා' in s or 'බැවින්' in s or 'හේතුව' in s:score+=1.5
        if heading and any(k in s for k in ("පර්වතය", "උද්‍යානය", "පවුර", "සිතුවම්")):
            score+=0.02
        score-=min(len(sent)/4000, .18)
        ranked.append((score,idx,sent))
    ranked.sort(key=lambda x:(-x[0],x[1]))
    chosen = [ranked[0][2]]

    # Specific user intents sometimes need a second independent source sentence.
    if tags['morning_reason']:
        for _,idx,s in ranked[1:]:
            if ('උදෑසන පැමිණීම' in s or 'උණුසුම' in s) and not any(x in chosen[0] for x in ('උදෑසන පැමිණීම', 'උණුසුම') if x in s):
                chosen.append(s)
                break
    elif tags['travel_routes']:
        # A route answer needs the next leg, not only a generic reference to
        # Dambulla. Prefer accompanying the selected bus sentence with onward
        # transport when the corpus gives it.
        if any('බස් රථ' in sent for sent in chosen):
            for s in sentences:
                if 'එතැන් සිට' in s and ('බස් රථ' in s or 'ටැක්සි' in s):
                    if s not in chosen:chosen.append(s)
                    break
        else:
            for s in sentences:
                if 'දඹුල්ල' in s and ('ගමන්' in s or 'පැමිණ' in s) and s not in chosen:
                    chosen.append(s)
                    break
    elif tags['general_safety'] and not tags['wasp_safety']:
        for _,idx,s in ranked[1:]:
            if ('ජල' in s or 'ලිස්සන' in s or 'පාවහන්' in s) and s not in chosen:
                chosen.append(s)
                break
    elif tags['broad'] and not (tags['why'] or tags['art_content'] or tags['old_inscriptions'] or tags['where'] or tags['how_system'] or tags['travel_routes'] or tags['gardens_compare']):
        for _,idx,s in ranked[1:]:
            if not any(k in s for k in ("ප්‍රසිද්ධම අංගයක්", "තවත් පුරාවිද්‍යාත්මක අංගයකි")) and s not in chosen:
                chosen.append(s)
                break
    # Keep quoted source sentences in original order when two are selected.
    chosen.sort(key=lambda s: sentences.index(s))
    # Separate nonadjacent excerpts visibly; each clause is copied verbatim.
    excerpt = ' '.join(chosen) if len(chosen)==1 or (len(chosen)==2 and sentences.index(chosen[1])==sentences.index(chosen[0])+1) else ' […] '.join(chosen)
    # Basic support checks: these are necessary-not-sufficient heuristics.
    if tags['art_content'] and not ('කාන්තා රූප' in excerpt or ('රූප' in excerpt and 'සිතුවම්' in excerpt)):
        return '',False
    if tags['old_inscriptions'] and 'කැටපත් පවුර' in heading and not ('කවි' in excerpt or 'ලියා' in excerpt):
        return '',False
    if tags['morning_reason'] and 'සංචාරයට සුදුසු වේලාව' in heading and not ('උණුසුම' in excerpt or 'උදෑසන පැමිණීම' in excerpt):
        return '',False
    if tags['travel_routes'] and 'සීගිරියට ළඟා' in heading and not ('දඹුල්ල' in excerpt or 'බස් රථ' in excerpt):
        return '',False
    if tags['wasp_safety'] and 'බඹරුන්' in heading and not ('බඹර' in excerpt or 'නිශ්ශබ්ද' in excerpt or 'නිලධාරීන්ගේ' in excerpt):
        return '',False
    if tags['how_system'] and 'උල්පත්' in heading and not ('ගුරුත්වාකර්ෂණ' in excerpt or 'ජල පීඩනය' in excerpt):
        return '',False
    if not specific_detail_supported(subquestion, excerpt):
        return '',False
    return excerpt,True


def enhanced_answer_question(question, session, model, sources, headings, debug=False):
    gate=improved_gate(question)
    result={"question":question,"status":"","answer":"","sources":[],
            "reason":gate['reason'],"subquestions":[]}
    if gate['rejected']:
        result.update(status='LIVE_DATA_UNSUPPORTED',answer=NO_LIVE_SI)
        return result
    original_rank=fused_search(session,model,question)
    subquestions,enhanced_rank=multi_query_search(session,model,question,headings)
    if is_who_question(question):
        person=answer_who_from_verified_role(question,sources,original_rank,enhanced_rank)
        if person:
            return person
        result.update(status='INSUFFICIENT_DOCUMENT_EVIDENCE',answer=NO_SOURCE_SI,
                      reason='No explicit named person and requested role in source evidence.')
        return result

    # Strictly separate facets of multi-topic questions; no substitution of
    # generic activity/safari chunks for missing directions or gardens.
    facets=decompose(question)[:3]
    selected=[]
    missing=[]
    for facet in facets:
        row=find_topic_source(facet,sources)
        select_method='explicit_heading_and_topic' if row else 'lexical_semantic_fallback'
        if row is None:
            ranked=fused_search(session,model,facet) if len(facets)>1 else original_rank
            alternatives=candidate_scores(facet,sources,original_rank,ranked, (enhanced_rank,))
            if alternatives and alternatives[0][1]>=.38:
                row=sources.get(alternatives[0][3])
        if row is None:
            missing.append(facet)
            continue
        excerpt, supported=select_answer_quote(facet,row)
        if not supported or not excerpt:
            missing.append(facet)
            continue
        selected.append({
           'chunk_id':row['chunk_id'],'heading':row.get('heading',''),
           'source_quote':excerpt,'attractions':row.get('attractions') or [],
           'subquestion':facet,'route':select_method
        })

    # Inspect evidence for original exact-detail questions, even if decomposed.
    if len(facets)==1 and selected and not specific_detail_supported(question,selected[0]['source_quote']):
        missing=facets
        selected=[]
    result['subquestions']=facets
    if not selected:
        result.update(status='INSUFFICIENT_DOCUMENT_EVIDENCE',answer=NO_SOURCE_SI,
                      reason='No direct source evidence was confirmed for the requested detail.')
    else:
        parts=[]
        for item in selected:
            title=heading_clean(item['heading'])
            body=item['source_quote']
            if len(facets)>1 or any(t in item['subquestion'] for t in ('කොහෙද','කොහේද','කොතැනද')):
                parts.append(f'{title}: {body}')
            else:
                parts.append(body)
        if missing:
            parts.append('සෙසු කරුණ සඳහා: '+NO_SOURCE_SI)
        result.update(status=('PARTIAL_SOURCE_EVIDENCE' if missing else 'SOURCE_GROUNDED_EXTRACTIVE'),
                      answer='\n'.join(parts),
                      sources=[{k:v for k,v in item.items() if k in ('chunk_id','heading','source_quote','attractions')} for item in selected],
                      reason=('One or more parts need human/source review.' if missing else 'Document excerpts selected per question topic; not externally verified.'))
    if debug:
        result['debug']={'original_rrf_top5':original_rank[:5],
                         'multi_query_top5':enhanced_rank[:5],
                         'facet_routing':[{'facet':f,'source':next((x['chunk_id'] for x in selected if x['subquestion']==f),None)} for f in facets],
                         'missing_facets':missing}
    return result


# Override the earlier overly broad sentence and source selector. All batch
# CLI, JSON/TXT/CSV export and original read-only Neo4j operations stay intact.
answer_question=enhanced_answer_question


def format_answer(result, index=None, total=None, debug=False):
    """Return the exact console/text-file response, answer before evidence."""
    lines = ["", "=" * 70]
    if index is not None:
        lines.append(f"QUESTION {index}/{total}")
    lines.extend(["ප්‍රශ්නය: " + result["question"], "", "පිළිතුර:", result["answer"],
                  "", "මූලාශ්‍ර සාක්ෂි:"])
    if result["sources"]:
        for i, src in enumerate(result["sources"], 1):
            lines.append(f"[{i}] {src['chunk_id']} — {src['heading']}")
            lines.append(f"    උපුටාගත් පෙළ: {src['source_quote']}")
            if src.get("attractions"):
                lines.append(f"    සම්බන්ධ ස්ථාන: {', '.join(src['attractions'])}")
    else:
        lines.append("තහවුරු කළ හැකි ලේඛන සාක්ෂි නොමැත.")
    lines.extend(["", "Status: " + result["status"]])
    if debug:
        lines.append("Reason: " + result.get("reason", ""))
        if "debug" in result:
            lines.append("Debug retrieval: " + json.dumps(result["debug"], ensure_ascii=False, indent=2))
    lines.append("=" * 70)
    return "\n".join(lines)


def parse_question_lines(raw_text):
    """Accept a multiline paste, optional '1.' numbering, and blank lines."""
    questions = []
    for line in raw_text.splitlines():
        line = line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        if line.upper() == "END":
            break
        # Remove list numbers from pasted WhatsApp/Word text, not Sinhala text.
        line = re.sub(r"^\s*[0-9]{1,3}[.)-]\s+", "", line).strip()
        if line:
            questions.append(line)
    return questions


def read_pasted_questions():
    print("\nඔබගේ සිංහල ප්‍රශ්න සියල්ල එකවර paste කරන්න (එක පේළියකට එක ප්‍රශ්නයක්).")
    print("සියල්ල ඇතුළත් වූ පසු අලුත් පේළියක END ලියා Enter කරන්න.")
    print("Example: 1. සීගිරිය ඉදිකළ රජතුමා කවුද?\n         2. සීගිරි බිතුසිතුවම්වල දැකිය හැක්කේ මොනවාද?\n         END\n")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return parse_question_lines("\n".join(lines))


def save_results(results, rendered_text):
    OUTPUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    # UTF-8 with BOM improves Sinhala display when Windows opens the text/CSV.
    OUTPUT_TEXT.write_text(rendered_text + "\n", encoding="utf-8-sig")
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "number", "question", "answer", "status", "chunk_ids",
            "headings", "source_evidence", "reason",
        ])
        writer.writeheader()
        for i, res in enumerate(results, 1):
            writer.writerow({
                "number": i,
                "question": res["question"],
                "answer": res["answer"],
                "status": res["status"],
                "chunk_ids": "; ".join(s["chunk_id"] for s in res["sources"]),
                "headings": "; ".join(s["heading"] for s in res["sources"]),
                "source_evidence": " | ".join(s["source_quote"] for s in res["sources"]),
                "reason": res.get("reason", ""),
            })


def safe_answer_one(question, session, model, sources, headings, debug=False):
    try:
        return answer_question(question, session, model, sources, headings, debug)
    except Exception as exc:
        # Avoid incorrectly presenting failed retrieval as an evidential answer.
        return {"question": question, "status": "RETRIEVAL_ERROR", "answer":
                "මෙම ප්‍රශ්නය සඳහා තොරතුරු ලබාගැනීමේදී දෝෂයක් ඇති විය.",
                "sources": [], "reason": f"{type(exc).__name__}: {exc}",
                "subquestions": []}


def main():
    parser = argparse.ArgumentParser(
        description="Batch Sinhala Sigiriya GraphRAG: all questions in one run")
    parser.add_argument("--file", type=Path, help="UTF-8/UTF-8-SIG text file, one question per line")
    parser.add_argument("--question", action="append", help="One question; repeat --question for many")
    parser.add_argument("--demo", action="store_true", help="Use four sample questions")
    parser.add_argument("--debug", action="store_true", help="Show retrieval diagnostics")
    parser.add_argument("--no-save", action="store_true", help="Do not write JSON/TXT/CSV files")
    args = parser.parse_args()

    if args.file:
        try:
            questions = parse_question_lines(args.file.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError) as exc:
            parser.error(f"Could not read question file: {exc}")
    elif args.question:
        questions = [q.strip() for q in args.question if q.strip()]
    elif args.demo:
        questions = [
            "සීගිරිය ඉදිකළ රජතුමා කවුද?",
            "සීගිරි බිතුසිතුවම්වල දැකිය හැක්කේ මොනවාද?",
            "නාග පෙන ගුහාවට එම නම ලැබුණේ ඇයි?",
            "හෙට සීගිරියේ කාලගුණය කොහොමද?",
        ]
    else:
        questions = read_pasted_questions()

    if not questions:
        parser.error("No questions were supplied; paste questions and finish with END.")
    print(f"\nReceived {len(questions)} Sinhala question(s).")
    results = [None] * len(questions)
    pending = [i for i, q in enumerate(questions) if not improved_gate(q)["rejected"]]
    for i, q in enumerate(questions):
        if i not in pending:
            gate = improved_gate(q)
            results[i] = {"question": q, "status": "LIVE_DATA_UNSUPPORTED",
                          "answer": NO_LIVE_SI, "sources": [],
                          "reason": gate["reason"], "subquestions": []}
    if pending:
        password = getpass("Neo4j password: ")
        with GraphDatabase.driver(URI, auth=(USERNAME, password)) as driver:
            driver.verify_connectivity()
            with driver.session(database=DATABASE) as session:
                sources = load_sources(session)
                headings = {r["chunk_id"]: r["heading"] for r in session.run(HEADINGS_QUERY)}
                print(f"Loaded {len(sources)} Sigiriya source chunks (read-only).")
                print("Loading multilingual E5 model ONCE for the whole batch...")
                model = SentenceTransformer(MODEL_NAME)
                for n, i in enumerate(pending, 1):
                    print(f"Processing {n}/{len(pending)}: Q{i + 1}", flush=True)
                    results[i] = safe_answer_one(
                        questions[i], session, model, sources, headings, args.debug
                    )

    rendered = "\n".join(
        format_answer(result, i, len(results), args.debug)
        for i, result in enumerate(results, 1)
    )
    # All answers are displayed together, only after all processing is finished.
    print("\n" + "#" * 70 + "\nALL ANSWERS — ANSWER FIRST, SOURCE EVIDENCE BELOW\n" + "#" * 70)
    print(rendered)
    if not args.no_save:
        save_results(results, rendered)
        print("\nSaved:", OUTPUT_TEXT)
        print("Saved:", OUTPUT_CSV)
        print("Saved:", OUTPUT_JSON)
    print("No Neo4j data, indexes or vectors were modified.")


if __name__ == "__main__":
    main()
