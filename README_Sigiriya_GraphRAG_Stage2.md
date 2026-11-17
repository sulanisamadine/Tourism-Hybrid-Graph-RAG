# Sigiriya Hybrid GraphRAG — Completing Steps 5, 6 and 7

**Version:** 8 November 2026 | **Local Neo4j database:** `neo4j`, `bolt://127.0.0.1:7687` | **Embedding model:** `intfloat/multilingual-e5-base`

This package builds on the **existing** Neo4j database containing 22 Sinhala Sigiriya `Chunk` nodes, two vector indexes, and eight legacy `Attraction` nodes linked to relevant chunks. It does not rebuild or overwrite those originals.

## The three incomplete steps

**Step 5 — Wider entity linking:** 49 curated, namespaced `TourismEntity` definitions and 54 explicitly sourced relationships have been prepared. The 54 evidence fragments were validated against the attached original Sinhala tourism document, covering all 22 sections. Every relationship is revalidated against your **actual Neo4j Chunk text before writes**. New nodes are connected to `Chunk` nodes with `LINKS_TO_ENTITY`, and, where exact matches exist, to the eight original `Attraction` nodes through `KG_EQUIVALENT`. Supporting facts are stored as `TourismFact` nodes with `SUPPORTS_FACT`, `FACT_SUBJECT`, and `FACT_OBJECT` edges. Existing nodes and vector indexes remain untouched.

**Step 6 — Graph retrieval:** The program performs both heading/text E5 vector retrieval using the original RRF scripts and **real directed graph traversal up to two hops**, selecting entity anchors from Sinhala entity aliases **and from entity links on semantically retrieved chunks**. Each returned relationship has a `fact_id`, `chunk_id`, and exact source quotation. A separate graph-hybrid top-three ranking incorporates graph-supported source chunks. User-facing answers keep the existing **Sinhala answer first, then source evidence** format, plus the source-auditable graph paths. The answer writer remains a heuristic extractive approach, not an LLM.

**Step 7 — Evaluation:** A read-only 30-question comparison framework is provided. It evaluates the legacy rule baseline, original RRF, Multi-Query RRF, and graph-hybrid retrieval with Top-1, Precision@3, Recall@3, F1@3, all-required@3, entity anchor recall, two-hop path recall, unsupported-question safe declines, and measured latency. A manual answer-quality review sheet is also provided. **All gold annotations are provisional and source-informed, not independently blind.** Do not describe later tuning on these questions as held-out generalization.

## Files to copy into the same Windows Downloads folder

**New files:**
- `Sigiriya_GraphRAG_Stage2.py` — preview/apply expanded graph and ask questions
- `Sigiriya_KG_Expansion_Manifest.json` — 49 entity definitions and 54 sourced relation definitions
- `Sigiriya_GraphRAG_Stage2_Evaluation.py` — comparative evaluator
- `Sigiriya_GraphRAG_Stage2_Candidate_Evaluation.csv` — 30 new provisional questions with labels
- `Sigiriya_GraphRAG_Stage2_Manual_Review_Template.csv` — second-annotator answer review
- `Sigiriya_GraphRAG_Stage2_Offline_Tests.py` — quick structural/provenance smoke tests
- `README_Sigiriya_GraphRAG_Stage2.md` — this guide

**Existing working files (included in the ZIP for convenience):**
- `Sigiriya_Multi_Query_Retrieval.py`
- `Sigiriya_Answerability_Check.py`
- `Sigiriya_Retrieval_Improvement_Batch_Reviewed.py`
- `sigiriya_adaptive_retrieval_fixed.py`
- `SIGIRIYA – TOURISM INFORMATIONSinhalatxt(3).txt`
- `Sigiriya_Questions_Batch.txt` (earlier example questions for batch mode)

## Run in PowerShell

Start the **Sinhala Tourism Graph** in Neo4j Desktop, then:

```powershell
cd "C:\Users\ASUS\Downloads"
python -m pip install neo4j sentence-transformers
```

**1. No-database check (does not require Neo4j):**

```powershell
python ".\Sigiriya_GraphRAG_Stage2_Offline_Tests.py"
```

**2. Preview changes. This DOES NOT write to Neo4j:**

```powershell
python ".\Sigiriya_GraphRAG_Stage2.py" --preview-graph
```

Check `Sigiriya_GraphRAG_Graph_Preview.json`: `facts_with_exact_source_passages` should be **54**, `failures` should be `[]`, and `chunks_with_structured_fact_links` should be **22**. If any fail, **stop and check the source mismatch**.

**3. Explicitly apply the new subgraph (idempotent, single transaction):**

```powershell
python ".\Sigiriya_GraphRAG_Stage2.py" --apply-graph
python ".\Sigiriya_GraphRAG_Stage2.py" --inspect
```

**Only `--apply-graph` writes** new entities, facts, and sourced relationships. It makes no edits to existing `Chunk` text, embedding arrays, vector indexes or old `Attraction` nodes. Re-running uses `MERGE` and does not duplicate the new facts.

**4. Ask a Sinhala multi-hop question:**

```powershell
python ".\Sigiriya_GraphRAG_Stage2.py" --question "සීගිරිය අසල පිදුරංගල වෙත නැගීමේදී පාවහන් ගැන දී ඇති උපදෙස පැහැදිලි කරන්න."
```

Or process the prior questions in one run:

```powershell
python ".\Sigiriya_GraphRAG_Stage2.py" --file ".\Sigiriya_Questions_Batch.txt"
```

Answer output: `Sigiriya_GraphRAG_Stage2_Answers.json` with the Sinhala answer, chunk source evidence, retrieved entity anchors, sourced one/two-hop graph paths and hybrid top-three source IDs.

**5. Evaluate after successful graph apply:**

```powershell
python ".\Sigiriya_GraphRAG_Stage2_Evaluation.py"
```

The evaluator **reads** Neo4j only and saves:
- `Sigiriya_GraphRAG_Stage2_Evaluation_Results.csv`
- `Sigiriya_GraphRAG_Stage2_Evaluation_Summary.json`

The comparison uses the same frozen questions for every method. Gold labels are used **only after retrieval** to calculate scores; they are never fed into vector search, decomposition, entity anchoring, graph traversal, or answer selection. The question-set SHA-256 checksum appears in the summary.

## Neo4j Graph visualization

Use the Neo4j browser's **Graph** view after applying the expansion. Example two-hop route:

```cypher
MATCH p=(s:TourismEntity {id:'sg:dest'})-[:HAS_ATTRACTION]->(a:TourismEntity)-[:HAS_FEATURE]->(f:TourismEntity)
RETURN p LIMIT 30;
```

See all new sourced entities and one-hop edges:

```cypher
MATCH (a:TourismEntity)-[r]->(b:TourismEntity)
WHERE r.scope = 'Sigiriya'
RETURN a,r,b LIMIT 100;
```

Verify the legacy graph connections:

```cypher
MATCH (a:Attraction)-[r:KG_EQUIVALENT]->(e:TourismEntity)
RETURN a,r,e;
```

Check 22 chunk coverage:

```cypher
MATCH (c:Chunk)-[:LINKS_TO_ENTITY]->(e:TourismEntity)
WHERE c.destination='Sigiriya'
RETURN count(DISTINCT c) AS connected_chunks,
       count(DISTINCT e) AS linked_entities;
```

## Important limitations and scientific reporting

- The complete Step 5/6/7 **code** has been prepared and structurally tested offline. It is **not yet verified against your running Windows Neo4j** until you execute the preview, apply, graph queries, and evaluation.
- A two-hop route is actual traversal of persisted, sourced graph edges, but does **not** prove that the question has been fully understood or the final answer is factually perfect.
- Legacy rule baseline is restricted to the eight legacy attractions. New graph entities and relations were **manually curated from this one Sigiriya document**, not automatically extracted from all tourism documents. Broader destination coverage remains future work.
- The new questions were authored with knowledge of the source and source-specific rules were developed earlier. They are **not an externally blind or representative benchmark**. Freeze code and questions, ask a second person to validate labels and score the answer-review sheet, and subsequently use a truly independent question set to establish generalization.
- Data about prices, times, hotel availability and weather may be stale or unavailable in the source. Do not present such content as live information.
- Measured times depend on caching and exclude model startup from per-question evaluation. Compare the same repeated-run conditions; avoid claims of production latency.
- If preview fails, **do not force the build**. Share the `Sigiriya_GraphRAG_Graph_Preview.json` report to resolve differences between the provided tourism document and your graph's stored chunks.
