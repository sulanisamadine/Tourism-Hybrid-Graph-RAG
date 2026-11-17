"""Rule-based Sinhala retrieval for the Sigiriya Neo4j knowledge graph."""
import getpass
import re
from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USERNAME = "neo4j"

# Each alias is a clue, not a semantic model. Extend after evaluating failures.
ALIASES = {
    "Mirror Wall": ["කැටපත් පවුර", "කැටපත් බිත්තිය", "කවි ලියා ඇති පවුර",
                    "කවි ලියා", "පැරණි කවි", "කැටපත්"],
    "Sigiriya Frescoes": ["සීගිරි බිතුසිතුවම්", "බිතුසිතුවම්", "සිතුවම්",
                          "සීගිරි ලලනාවන්", "අප්සරාවන්"],
    "Water Gardens": ["ජල උද්‍යාන", "ජල උයන", "ජල තටාක", "ජල මාර්ග"],
    "Fountain Garden": ["දිය උල්පත් උද්‍යානය", "දිය උල්පත්", "උල්පත්"],
    "Boulder Garden": ["ගල් උද්‍යානය", "ගල් උයන"],
    "Cobra Hood Cave": ["නාග පෙන ගුහාව", "නාග පෙන", "ගුහාව"],
    "Sigiriya Museum": ["සීගිරි කෞතුකාගාරය", "කෞතුකාගාරය"],
    "Sigiriya Rock": ["සීගිරි පර්වතය", "පර්වතය", "සිංහ පාද", "රාජ මාලිගාව"],
}
SINHALA_NAMES = {
    "Mirror Wall": "කැටපත් පවුර",
    "Sigiriya Frescoes": "සීගිරි බිතුසිතුවම්",
    "Water Gardens": "ජල උද්‍යාන",
    "Fountain Garden": "දිය උල්පත් උද්‍යානය",
    "Boulder Garden": "ගල් උද්‍යානය",
    "Cobra Hood Cave": "නාග පෙන ගුහාව",
    "Sigiriya Museum": "සීගිරි කෞතුකාගාරය",
    "Sigiriya Rock": "සීගිරි පර්වතය",
}

def detect_attraction(question):
    matches = []
    for name, aliases in ALIASES.items():
        found = [alias for alias in aliases if alias in question]
        if found:
            matches.append((max(map(len, found)), name))
    if not matches:
        return None
    return sorted(matches, reverse=True)[0][1]

def detect_intent(question):
    if any(s in question for s in ["කුමක්ද", "මොකක්ද", "කුමන", "මොන"]):
        return "identify"
    if any(s in question for s in ["ඉහළ", "උස", "මීටර්"]):
        return "height"
    if any(s in question for s in ["විවෘත", "වේලාව", "වෙලාව"]):
        return "hours"
    return "description"

def fetch_attraction(tx, name):
    # Undirected MENTIONS match because query output did not establish direction.
    query = """
    MATCH (a:Attraction)-[:MENTIONS]-(c:Chunk)
    WHERE a.name = $name AND c.destination = 'Sigiriya'
    RETURN a.name AS attraction, properties(a) AS props,
           c.id AS chunk_id, c.heading AS heading, c.text AS evidence
    ORDER BY c.id
    LIMIT 5
    """
    return [dict(row) for row in tx.run(query, name=name)]

def retrieve(driver, question):
    name = detect_attraction(question)
    intent = detect_intent(question)
    if not name:
        return {"intent": intent, "answer": None, "reason": "No matching Sigiriya attraction rule."}
    with driver.session() as session:
        records = session.execute_read(fetch_attraction, name)
    if not records:
        return {"intent": intent, "attraction": name, "answer": None,
                "reason": "No matching Sigiriya chunk in Neo4j."}
    record = records[0]
    answer = None
    if intent == "identify":
        answer = SINHALA_NAMES[name]
    elif intent == "height" and name == "Sigiriya Frescoes":
        answer = record["props"].get("height_description_si")
    elif intent == "hours" and name == "Sigiriya Museum":
        # The hours are quoted from the document, not verified live opening hours.
        answer = "ලේඛනයට අනුව උදෑසන 8.30 සිට සවස 5.30 පමණ දක්වා."
    return {"intent": intent, "attraction": name, "answer": answer,
            "chunk_id": record["chunk_id"], "heading": record["heading"],
            "evidence": record["evidence"]}

def main():
    print("Sigiriya Sinhala adaptive retrieval (type exit to quit)")
    password = getpass.getpass("Neo4j password: ")
    try:
        with GraphDatabase.driver(URI, auth=(USERNAME, password)) as driver:
            driver.verify_connectivity()
            print("Connected to Neo4j.")
            while True:
                question = input("\nSinhala question: ").strip()
                if question.lower() in {"exit", "quit"}:
                    break
                if not question:
                    continue
                result = retrieve(driver, question)
                print("Intent:", result["intent"])
                print("Attraction:", result.get("attraction", "Not recognized"))
                if result.get("answer"):
                    print("Answer:", result["answer"])
                elif result.get("evidence"):
                    print("Answer: No short rule-based answer; source evidence below.")
                else:
                    print("No answer:", result.get("reason"))
                if result.get("evidence"):
                    print("Source:", result["chunk_id"], "|", result["heading"])
                    print("Evidence:", result["evidence"])
    except Exception as exc:
        print("Neo4j error:", exc)
        print("Check that Neo4j is running, URI is correct, and credentials are valid.")

if __name__ == "__main__":
    main()
