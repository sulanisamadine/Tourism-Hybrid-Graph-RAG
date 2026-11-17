# Sigiriya Stage 2 — Answer Refinement (after N05 / N06 / N18 diagnostics)

## Purpose
The source retrieval for N05, N06 and N18 already identified the correct chunks, but the answer extractor declined or returned partial information. This read-only patch rechecks narrowly-defined source-backed semantic intents (Buddhist donation inscriptions; cobra-hood-shaped rock; figures in frescoes; museum history). It uses exact sentences stored in the `Chunk` nodes and does not look at test IDs or gold labels during answering.

## Installation
Place both Python files in the **same existing Stage 2 project folder** next to:

- `Sigiriya_GraphRAG_Stage2_Quality_Fix.py`
- `Sigiriya_GraphRAG_Stage2_Quality_Fix_Evaluation.py`
- `Sigiriya_GraphRAG_Stage2_Candidate_Evaluation.csv`
- `Sigiriya_Retrieval_Improvement_Batch_Reviewed.py`
- `Sigiriya_Multi_Query_Retrieval.py`
- Your original `SIGIRIYA...Sinhalatxt...txt` source document

Do not replace or delete original files. No `--apply-graph` is required.

## Step 1 — Offline check

```powershell
cd "C:\Users\ASUS\Downloads\Sigiriya_Hybrid_GraphRAG_Stage2"
python ".\Sigiriya_GraphRAG_Stage2_Answer_Refinement.py" --self-test
```

The test should show four PASS lines (three supported question variants plus one unsupported protection check). It does not contact Neo4j.

## Step 2 — Live 30-question evaluation (after Step 1 passes)

Start your Neo4j database and run:

```powershell
python ".\Sigiriya_GraphRAG_Stage2_Answer_Refinement_Evaluation.py"
```

Saved as new files, without overwriting previous evaluation results:

- `Sigiriya_GraphRAG_Stage2_Answer_Refinement_Evaluation_Results.csv`
- `Sigiriya_GraphRAG_Stage2_Answer_Refinement_Evaluation_Summary.json`

The script reuses the Stage 2 quality-fix graph and retrieval logic, and only revisits declined or partially answered results using verbatim stored source passages. It makes no Neo4j writes.

## Interpretation

The 30-question set is a source-informed **development** set; these three failed questions helped shape this patch. Passing them after the fix is a regression check, **not** independent validation. Test unseen Sinhala paraphrases and unsupported cases for a fair accuracy claim. A relevant chunk, matching rule, and matching sentence do not establish all-purpose answer correctness.
