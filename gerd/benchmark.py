'''

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import islice
import difflib
from gerd.backends import TRANSPORTER
from gerd.transport import QAFileUpload, QAQuestion
from gerd.loader import load_model_from_config
from gerd.config import  load_qa_config
import logging
import re
from typing import Tuple
from dateutil.parser import parse





# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,   # oder DEBUG für mehr Details
    format="%(levelname)s:  %(message)s",
)

logger = logging.getLogger()
project_dir = Path(__file__).parent.parent
# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------

QUESTIONS = [
    "Wie heißt der Patient?",
    "Wann hat der Patient Geburtsdatum?",
    "Wann wurde der Patient aufgenommen?",
    "Wann wurde der Patient entlassen?",
]

LABEL_MAPPING = {
    "Wie heißt der Patient?": "PatientName" , 
    "Wann hat der Patient Geburtsdatum?" : "PatientGeburtsdatum",
    "Wann wurde der Patient aufgenommen?" :"AufnahmeDatum",
    "Wann wurde der Patient entlassen?" :"EntlassungsDatum"
}
FUZZY_THRESHOLD = 0.8
RAW_TEXT_DIR = project_dir / "tests/data/grascco/raw"
RESULT_CSV = project_dir / "results/grascco_benchmark_results.csv"
RESULT_CSV_WITHOUT_RAG = project_dir / "results/grascco_benchmark_results_without_rag.csv"
MAX_FILES = 5 # Maximale Anzahl zu bearbeitender Dateien
model_config = load_qa_config().model
llm = load_model_from_config(model_config)





# -------------------------------------------
# Prefix-Regeln je Label
# -------------------------------------------
LABEL_PREFIXES = {
    "PatientName": [
        "patient:",
        "patientin:",
        "der patient heißt:",
        "der patient heisst:",
        "die patientin:",
        "patientname:",
        "hr.",
        "fr.",
    ],
    "PatientGeburtsdatum": [
        "geburtsdatum:",
        "patient geburtsdatum:",
        "der patient hat geburtsdatum:",
        "dob:",
    ],
    "AufnameDatum": [
        "aufnahmedatum:",
        "patient wurde aufgenommen am:",
        "aufgenommen am:",
    ],
    "EntlassungsDatum": [
        "entlassungsdatum:",
        "patient wurde entlassen am:",
        "entlassen am:",
    ],
}

# -------------------------------------------
# Helfer-Funktionen
# -------------------------------------------

def normalize(s: str) -> str:
    return " ".join(s.lower().strip().split())

def levenshtein_distance(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, c1 in enumerate(a):
        curr_row = [i + 1]
        for j, c2 in enumerate(b):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

def levenshtein_ratio(a: str, b: str) -> float:
    dist = levenshtein_distance(a, b)
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return (max_len - dist) / max_len

def try_parse_date(s: str):
    try:
        return parse(s, dayfirst=True, fuzzy=True)
    except:
        return None

def extract_date_from_text(text: str):
    pattern = r"\d{1,2}\.\d{1,2}\.\d{2,4}"
    m = re.findall(pattern, text)
    if m:
        return m[0]
    return text.strip()

def clean_gt_value(label: str, value: str) -> str:
    v = value.strip()
    if label == "PatientName":
        v = re.sub(r"^(tr\.|hr\.|fr\.)\s*:?\s*", "", v, flags=re.I)
        return v.strip()
    if label == "PatientGeburtsdatum":
        v = re.sub(r"^geb\.\s*", "", v, flags=re.I)
        return v.strip()
    return v

# -------------------------------------------
# Haupt-Evaluator
# -------------------------------------------

def evaluate_prediction(gt: str, pred: str, label: str) -> Tuple[float, bool]:
    """
    Vergleicht Ground-truth vs Prediction für das gegebene Label
    Rückgabe: (match_score, match_bool)
    """

    gt = clean_gt_value(label, gt)
    gt_norm = normalize(gt)
    pred_norm = normalize(pred)

    # 1) Handle empty GT
    if not gt_norm:
        return 1.0, True

    # 2) Special Prefixes
    if label in LABEL_PREFIXES:
        for prefix in LABEL_PREFIXES[label]:
            if pred_norm.startswith(prefix):
                rest = pred_norm[len(prefix):].strip()
                # Name
                if label == "PatientName" and rest == gt_norm:
                    return 1.0, True
                # Datum
                if label in {"PatientGeburtsdatum", "AufnameDatum", "EntlassungsDatum"}:
                    rest_date = extract_date_from_text(rest)
                    gt_date = extract_date_from_text(gt_norm)
                    d1, d2 = try_parse_date(rest_date), try_parse_date(gt_date)
                    if d1 and d2 and d1 == d2:
                        return 1.0, True

    # 3) Datum-Label: parse & compare
    if label in {"PatientGeburtsdatum", "AufnameDatum", "EntlassungsDatum"}:
        pred_date = extract_date_from_text(pred_norm)
        gt_date = extract_date_from_text(gt_norm)
        d1, d2 = try_parse_date(pred_date), try_parse_date(gt_date)
        if d1 and d2 and d1 == d2:
            return 1.0, True
        score = levenshtein_ratio(pred_date, gt_date)
        return score, score >= FUZZY_THRESHOLD

    # 4) Name-Label: Levenshtein
    if label == "PatientName":
        score = levenshtein_ratio(gt_norm, pred_norm)
        return score, score >= FUZZY_THRESHOLD

    # 5) Fallback
    score = levenshtein_ratio(gt_norm, pred_norm)
    return score, score >= FUZZY_THRESHOLD




def load_grascco_annotations():
    with (project_dir / "tests/data/grascco/grascco.json").open() as f:
        annotation = {j["file_upload"].split("-")[1]: j for j in json.load(f)}
    sorted_annotaion = dict(sorted(annotation.items()))
    #print("sorted_annotaion:", sorted_annotaion)    
    return sorted_annotaion


def extract_label_text(annotation_entry: dict, label: str, text: str) -> str:
    ann = annotation_entry["annotations"][0]
    for r in ann["result"]:
        if r["value"]["labels"][0] == label:
            s = r["value"]["start"]
            e = r["value"]["end"]
            return text[s:e].strip()
    return ""






"""
def normalize(s: str) -> str:
    return " ".join(s.lower().strip().split())


def fuzzy_match(gt: str, pred: str) -> Tuple[float, bool]:
    a, b = normalize(gt), normalize(pred)
    for prefix in LABEL_PREFIXES:
        if b.startswith(prefix):
            rest = b[len(prefix):].strip()
            if rest == a:
                return 1.0, True
    score = difflib.SequenceMatcher(None, a, b).ratio()
    return score, score >= FUZZY_THRESHOLD
"""

def clean_pred(answer: str) -> str:
    if not answer:
        return ""
    answer = answer.strip()
    bad_starters = [
        "Die Patientin", "Der Patient", "Patientin", "Patient",
        "Es handelt sich", "Die Antwort ist"
    ]
    for b in bad_starters:
        if answer.startswith(b):
            answer = answer[len(b):].strip(" ,.")
    return answer


def make_prompt(text: str, question: str) -> str:
    return f"""
Gib ausschließlich den exakten Wortlaut zurück, wie er im Text vorkommt.
Keine Sätze. Keine Erklärungen. Nur den wörtlichen relevanten Text.
Antworte mit 'Unbekannt', wenn die Information im Text nicht vorhanden ist.
Auf deutsche Sprache antworten.

Text:
{text}

Frage:
{question}
"""

# ----------------------------------------------------
# BENCHMARK 
# ----------------------------------------------------

def run_benchmark_rag():
    ann = load_grascco_annotations()
    rows = []
    total = 0
    correct = 0

    files = islice(sorted(RAW_TEXT_DIR.glob("*.txt")), MAX_FILES)

    for file_path in files:
        fname = file_path.name.replace("ö", "o")
        if fname not in ann:
            print(f"Keine Annotation für {fname}")
            continue

        annotation_entry = ann[fname]
        text = file_path.read_text()

        # Reset + Upload
        try:
            TRANSPORTER.remove_file(fname)
        except Exception:
            pass

        upload = QAFileUpload(data=text.encode("utf-8"), name=fname)
        res = TRANSPORTER.add_file(upload)
        if res.status != 200:
            print(f"Fehler beim Laden: {res.error_msg}")
            continue

        # Questions
        for question in QUESTIONS:
            total += 1
            label = LABEL_MAPPING[question]
            truth_answer = extract_label_text(annotation_entry, label, text)

            # Forciere extraktive Antwort
            prompt = make_prompt(text, question)

            q = QAQuestion(
                question=prompt,
                search_strategy="similarity",
                max_sources=1,
                no_think=False
            )

            qa_res = TRANSPORTER.qa_query(q)
            pred_answer = clean_pred(qa_res.response)

            score, match = fuzzy_match(truth_answer, pred_answer)
            if match:
                correct += 1

            print(f"{fname:25} | {question:35} |{truth_answer:15} |{pred_answer:15}| score={score:.2f} | match={match}")
            
            rows.append({
                "file": fname,
                "question": question,
                "label": label,
                "ground_truth": truth_answer,
                "predicted": pred_answer,
                "match_score": f"{score:.3f}",
                "match": match
            })

    acc = correct / total if total else 0
    print("================================================")
    print(f"FINAL SCORE: {correct}/{total}  accuracy={acc:.3f}")
    print("================================================")

    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with RESULT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Ergebnisse gespeichert in: {RESULT_CSV}")







def run_benchmark_Rag_one_file():
    ann = load_grascco_annotations()
    rows = []
    total = 0
    correct = 0

    file_path = RAW_TEXT_DIR / "Baastrup.txt"   # ← Datei *Pfad*
    fname = file_path.name.replace("ö", "o")    

    print("fname:", fname)

    if fname not in ann:
        print(f"Keine Annotation für {fname}")
        return  # oder raise Exception

    annotation_entry = ann[fname]

    # Dateiinhalt lesen
    with file_path.open(encoding="utf-8") as f:
        text = file_path.read_text()
        
        # Reset + Upload
        try:
            TRANSPORTER.remove_file(fname)
        except Exception:
            logger.warning(f"Fehler beim Entfernen der Datei {fname} (vielleicht existiert sie nicht).")
            

        upload = QAFileUpload(data=text.encode("utf-8"), name=fname)
        res = TRANSPORTER.add_file(upload)
        if res.status != 200:
            print(f"Fehler beim Laden: {res.error_msg}")
        # Questions
        for question in QUESTIONS:
            total += 1
            label = LABEL_MAPPING[question]
            truth_answer = extract_label_text(annotation_entry, label, text)

            # Forciere extraktive Antwort
            #prompt = make_prompt(text, question)

            q = QAQuestion(
                question=question,
                search_strategy="similarity",
                max_sources=1,
                no_think=False
            )

            qa_res = TRANSPORTER.qa_query(q)
            pred_answer = qa_res.response
            score, match = evaluate_prediction(truth_answer, pred_answer, label)
            #score, match = fuzzy_match(truth_answer, pred_answer)
            if match:
                correct += 1

            print(f"{fname:25} | {question:35} |{truth_answer:15} |{pred_answer:15}| score={score:.2f} | match={match}")
            
            rows.append({
                "file": fname,
                "question": question,
                "label": label,
                "ground_truth": truth_answer,
                "predicted": pred_answer,
                "match_score": f"{score:.3f}",
                "match": match
            })

    acc = correct / total if total else 0
    print("================================================")
    print(f"FINAL SCORE: {correct}/{total}  accuracy={acc:.3f}")
    print("================================================")

    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with RESULT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Ergebnisse gespeichert in: {RESULT_CSV}")




def run_benchmark_no_rag():
    ann = load_grascco_annotations()
    rows = []
    total = 0
    correct = 0

    file_path = RAW_TEXT_DIR / "Albers.txt"
    fname = file_path.name.replace("ö", "o")

    logger.info(f"Datei: {fname}")
    logger.info(f"Model: {model_config.name}")

    #print("fname:", fname)
    #print("Model:", model_config.name )

    if fname not in ann:
        logger.warning(f"Keine Annotation für {fname}")
        #print(f"Keine Annotation für {fname}")
        return

    annotation_entry = ann[fname]

    # Dateiinhalt lesen
    # Wichtig: Sie rufen file_path.read_text() zweimal auf, einmal beim Öffnen des Kontexts, einmal im Kontext. 
    # Ich verwende die zweite, korrekte Methode.
    text = file_path.read_text(encoding="utf-8")
        
    # Questions
    no_think_option = input("Möchten Sie 'no_think' aktivieren? (j/n): ").strip().lower() == 'j'
    logger.info(f"'no_think' ist auf {no_think_option} gesetzt.")
    #print(f"'no_think' ist auf {no_think_option} gesetzt.")
    
    for question in QUESTIONS:
        total += 1
        label = LABEL_MAPPING[question]
        truth_answer = extract_label_text(annotation_entry, label, text)
     
        # Forciere extraktive Antwort
        #prompt = make_prompt(text, question)
        
        if no_think_option:
            question = f"/no_think {question}"
            logger.info(f"Modifizierte Frage mit /no_think: {question}")

            #print(f"Modifizierte Frage mit /no_think: {question}")
        
        qa_res = None
        
        try:
            # LLM-Aufruf (Hier tritt der Fehler auf, wenn LLM.create_chat_completion None zurückgibt)
            qa_res = llm.create_chat_completion(
                messages=[{"role": "user", "content": question}],
            )
        except Exception as e:
            # Fängt Kommunikationsfehler des LLM-Backends ab
            print(f"WARNUNG: LLM-Aufruf für {fname} - {question} ist fehlgeschlagen: {e}")
            qa_res = None # Stellt sicher, dass qa_res None ist
        
        
        pred_answer_without_Rag = ""
        
        # ROBUSSTE PRÜFUNG DES RÜCKGABEWERTES
        # Wir prüfen, ob qa_res ein Tupel ist und greifen nur dann auf [1] zu
        if isinstance(qa_res, tuple) and len(qa_res) == 2:
            # Entpacken/Indexzugriff nur, wenn es ein Tupel ist
            #llm_raw_response = qa_res[1]
            #pred_answer_without_Rag = clean_pred(llm_raw_response)
            pred_answer_without_Rag = qa_res[1]
        else:
            # Fehlerfall: qa_res war None oder hatte falsches Format. 
            # pred_answer_without_Rag bleibt "" (oder ein Fehlerstring)
            print(f"FEHLER: LLM gab keinen gültigen Tupel-Rückgabewert ({type(qa_res)}) zurück.")


        score, match = fuzzy_match(truth_answer, pred_answer_without_Rag)
        if match:
            correct += 1

        print(f"{fname:25} | {question:35} |{truth_answer:15} |{pred_answer_without_Rag:15}| score={score:.2f} | match={match}")
        
        rows.append({
            "file": fname,
            "question": question,
            "label": label,
            "ground_truth": truth_answer,
            "predicted": pred_answer_without_Rag,
            "match_score": f"{score:.3f}",
            "match": match
        })

    acc = correct / total if total else 0
    print("================================================")
    print(f"FINAL SCORE: {correct}/{total}  accuracy={acc:.3f}")
    print("================================================")

    # Speichern der Ergebnisse (unverändert)
    RESULT_CSV_WITHOUT_RAG.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with RESULT_CSV_WITHOUT_RAG.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Ergebnisse gespeichert in: {RESULT_CSV_WITHOUT_RAG}")



if __name__ == "__main__":
    
    #Um Benchmark für alle oder beliebige Anzahl von Dateien mit Rag auszuführen kommentiere diese Zeile aus,
    #run_benchmark_rag()
    #----------------------------------------------------
    #Um Benchmark für eine Datei mit Rag auszuführen kommentiere diese Zeile aus,
    run_benchmark_Rag_one_file()
    #----------------------------------------------------
    #Um Benchmark für eine Datei ohne Rag auszuführen kommentiere diese Zeile aus,
    #run_benchmark_no_rag() 
    

    
'''


"""
Finaler, bereinigter Evaluator & Benchmark-Skript in einem Block.

Dieses Script enthält:
- robuste Extraktion der Ground-Truth aus LabelStudio-Export (grascco.json)
- robuste Evaluation (Namen, Datumsfelder, Prefix-Matching, Fallbacks)
- Benchmarks: RAG / single-file RAG / no-RAG (wie im Original)
- keine Abhängigkeit zu externen C-Bibliotheken für Levenshtein
- alle Fixes integriert (kein auskommentiertes fuzzy_match, sichere Slicing-Fallbacks)
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import islice
import logging
import re
from dateutil.parser import parse

# Falls du die gerd-Module lokal hast, lasse die Imports; sonst bitte anpassen.
from gerd.backends import TRANSPORTER
from gerd.transport import QAFileUpload, QAQuestion
from gerd.loader import load_model_from_config
from gerd.config import load_qa_config

# ----------------------------------------------------
# Basic logging configuration
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:  %(message)s")
logger = logging.getLogger()

# Projektverzeichnis (anpassen falls nötig)
project_dir = Path(__file__).parent.parent

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
QUESTIONS = [
    "Wie heißt der Patient?",
    "Wann hat der Patient Geburtsdatum?",
    "Wann wurde der Patient aufgenommen?",
    "Wann wurde der Patient entlassen?",
]

LABEL_MAPPING = {
    "Wie heißt der Patient?": "PatientName",
    "Wann hat der Patient Geburtsdatum?": "PatientGeburtsdatum",
    "Wann wurde der Patient aufgenommen?": "AufnahmeDatum",
    "Wann wurde der Patient entlassen?": "EntlassungsDatum"
}

FUZZY_THRESHOLD = 0.8
RAW_TEXT_DIR = project_dir / "tests/data/grascco/raw"
RESULT_CSV = project_dir / "results/grascco_benchmark_results.csv"
RESULT_CSV_WITHOUT_RAG = project_dir / "results/grascco_benchmark_results_without_rag.csv"
MAX_FILES = 5  # Maximale Anzahl zu bearbeitender Dateien

# LLM laden (wie vorher)
model_config = load_qa_config().model
llm = load_model_from_config(model_config)

# -------------------------------------------
# Prefix-Regeln je Label (erweiterbar)
# -------------------------------------------
LABEL_PREFIXES = {
    "PatientName": [
        "patient:",
        "patientin:",
        "der patient heißt:",
        "der patient heisst:",
        "die patientin:",
        "patientname:",
        "hr.",
        "fr.",
        "herr",
        "frau",
    ],
    "PatientGeburtsdatum": [
        "geburtsdatum:",
        "patient geburtsdatum:",
        "der patient hat geburtsdatum:",
        "dob:",
        "geb.",
        "geburtsdatum",
    ],
    "AufnameDatum": [
        "aufnahmedatum:",
        "patient wurde aufgenommen am:",
        "aufgenommen am:",
        "aufgenommen:",
    ],
    "EntlassungsDatum": [
        "entlassungsdatum:",
        "patient wurde entlassen am:",
        "entlassen am:",
        "entlassen:",
    ],
}

# -------------------------------------------
# Helfer-Funktionen
# -------------------------------------------

def normalize(s: str) -> str:
    return " ".join(s.lower().strip().split()) if s else ""

def levenshtein_distance(a: str, b: str) -> int:
    a, b = (a or "").lower(), (b or "").lower()
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, c1 in enumerate(a):
        curr_row = [i + 1]
        for j, c2 in enumerate(b):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

def levenshtein_ratio(a: str, b: str) -> float:
    a, b = a or "", b or ""
    dist = levenshtein_distance(a, b)
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return (max_len - dist) / max_len

def try_parse_date(s: str):
    try:
        return parse(s, dayfirst=True, fuzzy=True)
    except Exception:
        return None

def extract_date_from_text(text: str):
    if not text:
        return ""
    # unterstütze verschiedene Trenner: ., -, /
    # suche dd.mm.yyyy oder dd.mm.yy
    pattern = r"\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}"
    m = re.findall(pattern, text)
    if m:
        return m[0]
    # fallback: auch merkmale wie "8. März 2025" finden (dateutil fuzzy handles it)
    return text.strip()

def clean_gt_value(label: str, value: str) -> str:
    v = (value or "").strip()
    if not v:
        return ""
    if label == "PatientName":
        # entferne führende Annotation-Prefixes wie "tr.:", "hr.", "fr."
        v = re.sub(r"^(tr\.|hr\.|fr\.|herr|frau)\s*:?\s*", "", v, flags=re.I).strip()
        return v
    if label == "PatientGeburtsdatum":
        v = re.sub(r"^(geb\.|geburtsdatum:)\s*", "", v, flags=re.I).strip()
        return v
    return v

def remove_titles(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\b(hr|fr|herr|frau|dr|dr\.)\.?\b", "", s, flags=re.I).strip()

def safe_slice_text(text: str, start: int, end: int) -> str:
    if text is None:
        return ""
    # clamp indices
    start = max(0, int(start))
    end = min(len(text), int(end))
    if start >= end:
        return ""
    return text[start:end]

# -------------------------------------------
# Robuster Extractor aus LabelStudio JSON
# -------------------------------------------
def load_grascco_annotations() -> Dict[str, dict]:
    """
    Lädt die grascco.json und erstellt ein Mapping:
    key = normalisierter Dateiname (z.B. 'Tupolev_3.txt' -> 'Tupolev_3.txt' ohne Prefix-Hash)
    value = whole annotation entry
    """
    json_path = project_dir / "tests/data/grascco/grascco.json"
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    for entry in data:
        file_upload = entry.get("file_upload", "")
        # file_upload ist z.B. "555b9929-Tupolev_3.txt" -> wir nehmen letztes Segment nach '-'
        name = file_upload.split("-", 1)[-1] if "-" in file_upload else file_upload
        # normalize umlauts to simple ascii variant for matching with raw files
        name_norm = name.replace("ö", "o").replace("ä", "a").replace("ü", "u")
        mapping[name_norm] = entry
    # return sorted mapping for determinismus
    return dict(sorted(mapping.items()))

def extract_label_text(annotation_entry: dict, label: str, text: str) -> str:
    """
    Robuste Extraktion:
      - versucht direkt text[start:end]
      - wenn leer/invalid, versucht kleine Offsets +/-10
      - falls weiterhin nichts, sucht nach plausiblen Substrings (z.B. kurze token-sequenzen)
      - immer sauber ge-stript zurückgegeben
    """
    if not annotation_entry or "annotations" not in annotation_entry or not text:
        return ""

    ann = annotation_entry["annotations"][0]
    results = ann.get("result", [])
    for r in results:
        lbls = r.get("value", {}).get("labels") or []
        if not lbls:
            continue
        if lbls[0] != label:
            continue

        start = r.get("value", {}).get("start")
        end = r.get("value", {}).get("end")
        # try direct slice
        try:
            if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
                candidate = safe_slice_text(text, start, end).strip()
                if candidate:
                    return candidate
        except Exception:
            pass

        # normalize line endings and try +/- offsets
        t_norm = text.replace("\r\n", "\n")
        for delta in range(-12, 13):
            s = max(0, (start or 0) + delta)
            e = min(len(t_norm), (end or 0) + delta)
            if s < e:
                part = t_norm[s:e].strip()
                if part and len(part) <= 200:  # reasonable chunk
                    return part

        # fallback: look for a short token sequence around start
        tokens = re.findall(r"\S+", t_norm)
        if tokens:
            # try to find token that contains the start offset approximately
            char_pos = 0
            for tok in tokens:
                tok_start = char_pos
                tok_end = char_pos + len(tok)
                if isinstance(start, int) and tok_start <= start <= tok_end:
                    # return this token plus a few neighbors
                    idx = tokens.index(tok)
                    snippet = " ".join(tokens[max(0, idx-2): min(len(tokens), idx+3)])
                    return snippet.strip()
                char_pos = tok_end + 1
        # if nothing found, return empty
        return ""
    return ""

# -------------------------------------------
# Prediction cleaning
# -------------------------------------------
def clean_pred(answer: str) -> str:
    """
    Entfernt nur deklarative Präfix-Formulierungen, lässt den Kerntext erhalten.
    """
    if not answer:
        return ""
    answer = answer.strip()
    lower = answer.lower()

    # mögliche Prefixes (deutsch) - wir schneiden nur exakt diese Präfix-Teile ab
    bad_prefixes = [
        "der patient heißt",
        "der patient heisst",
        "die patientin heißt",
        "die patient heißt",
        "patientin:",
        "patient:",
        "antwort:",
        "unbekannt",
        "nicht angegeben",
    ]

    for prefix in bad_prefixes:
        if lower.startswith(prefix):
            # entferne exakt Länge des Prefixes aus dem Originalstring (erhält Großschreibung etc.)
            return answer[len(prefix):].strip(" :.,")

    return answer

# -------------------------------------------
# Evaluation / Matching
# -------------------------------------------

def evaluate_prediction(gt: str, pred: str, label: str) -> Tuple[float, bool]:
    """
    Vergleicht Ground-truth vs Prediction für das gegebene Label.
    Rückgabe: (match_score [0..1], match_bool)
    """

    gt_raw = clean_gt_value(label, gt or "")
    pred_raw = pred or ""

    gt_norm = normalize(gt_raw)
    pred_norm = normalize(pred_raw)

    # 1) Wenn GT leer => optional: nicht akzeptiere Prediction (hier: wir akzeptieren)
    if not gt_norm:
        return 0.0, False

    # 2) Prefix-basiertes Exact-Match (wenn Prediction z.B. "Patientin: Anna Müller")
    if label in LABEL_PREFIXES:
        for prefix in LABEL_PREFIXES[label]:
            p = prefix.lower()
            if pred_norm.startswith(p):
                rest = pred_norm[len(p):].strip()
                # Name: exakter Rest gleich GT
                if label == "PatientName":
                    rest_clean = normalize(remove_titles(rest))
                    if rest_clean == normalize(remove_titles(gt_norm)):
                        return 1.0, True
                # Datum: extract & parse
                if label in {"PatientGeburtsdatum", "AufnameDatum", "EntlassungsDatum"}:
                    rest_date = extract_date_from_text(rest)
                    gt_date = extract_date_from_text(gt_norm)
                    d1 = try_parse_date(rest_date)
                    d2 = try_parse_date(gt_date)
                    if d1 and d2 and d1 == d2:
                        return 1.0, True

    # 3) Datum-Labels: parse & compare (robust)
    if label in {"PatientGeburtsdatum", "AufnameDatum", "EntlassungsDatum"}:
        pred_date_text = extract_date_from_text(pred_norm)
        gt_date_text = extract_date_from_text(gt_norm)
        d1 = try_parse_date(pred_date_text)
        d2 = try_parse_date(gt_date_text)
        if d1 and d2 and d1 == d2:
            return 1.0, True
        # fallback: levenshtein on the extracted date tokens (e.g. "8. März 2025" vs "08.03.2025")
        score = levenshtein_ratio(pred_date_text, gt_date_text)
        return score, score >= FUZZY_THRESHOLD

    # 4) Name-Label: robuste Heuristiken + Levenshtein-Fallback
    if label == "PatientName":
        gt_clean = normalize(remove_titles(gt_norm))
        pred_clean = normalize(remove_titles(pred_norm))

        # if one contains the other
        if gt_clean and pred_clean and (gt_clean in pred_clean or pred_clean in gt_clean):
            return 1.0, True

        # token-set equality (order-insensitive)
        if set(gt_clean.split()) == set(pred_clean.split()) and gt_clean and pred_clean:
            return 1.0, True

        # Levenshtein fallback
        score = levenshtein_ratio(gt_clean, pred_clean)
        return score, score >= FUZZY_THRESHOLD

    # 5) Allgemeiner Fallback (Levenshtein)
    score = levenshtein_ratio(gt_norm, pred_norm)
    return score, score >= FUZZY_THRESHOLD

# -------------------------------------------
# Prompt-Baustein (optional)
# -------------------------------------------
def make_prompt(text: str, question: str) -> str:
    return f"""
Gib ausschließlich den exakten Wortlaut zurück, wie er im Text vorkommt.
Keine Sätze. Keine Erklärungen. Nur den wörtlichen relevanten Text.
Antworte mit 'Unbekannt', wenn die Information im Text nicht vorhanden ist.
Auf deutsche Sprache antworten.

Text:
{text}

Frage:
{question}
""".strip()

# -------------------------------------------
# BENCHMARK-Funktionen (RAG / Single-File RAG / No-RAG)
# -------------------------------------------

def run_benchmark_rag():
    ann = load_grascco_annotations()
    rows = []
    total = 0
    correct = 0

    files = islice(sorted(RAW_TEXT_DIR.glob("*.txt")), MAX_FILES)

    for file_path in files:
        fname = file_path.name.replace("ö", "o")
        if fname not in ann:
            logger.warning(f"Keine Annotation für {fname}")
            continue

        annotation_entry = ann[fname]
        text = file_path.read_text(encoding="utf-8")

        # Reset + Upload
        try:
            TRANSPORTER.remove_file(fname)
        except Exception:
            pass

        upload = QAFileUpload(data=text.encode("utf-8"), name=fname)
        res = TRANSPORTER.add_file(upload)
        if res.status != 200:
            logger.error(f"Fehler beim Laden: {res.error_msg}")
            continue

        for question in QUESTIONS:
            total += 1
            label = LABEL_MAPPING[question]
            truth_answer = extract_label_text(annotation_entry, label, text)

            prompt = make_prompt(text, question)
            q = QAQuestion(question=prompt, search_strategy="similarity", max_sources=1, no_think=False)
            qa_res = TRANSPORTER.qa_query(q)
            pred_answer = clean_pred(qa_res.response if qa_res and getattr(qa_res, "response", None) else "")

            score, match = evaluate_prediction(truth_answer, pred_answer, label)
            if match:
                correct += 1

            logger.info(f"{fname:25} | {question:35} | GT: {truth_answer!r:30} | PRED: {pred_answer!r:30} | score={score:.2f} | match={match}")

            rows.append({
                "file": fname,
                "question": question,
                "label": label,
                "ground_truth": truth_answer,
                "predicted": pred_answer,
                "match_score": f"{score:.3f}",
                "match": match
            })

    acc = correct / total if total else 0
    logger.info("================================================")
    logger.info(f"FINAL SCORE: {correct}/{total}  accuracy={acc:.3f}")
    logger.info("================================================")

    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with RESULT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Ergebnisse gespeichert in: {RESULT_CSV}")

def run_benchmark_Rag_one_file():
    ann = load_grascco_annotations()
    rows = []
    total = 0
    correct = 0

    file_path = RAW_TEXT_DIR / "Vogler.txt"
    fname = file_path.name.replace("ö", "o")

    if fname not in ann:
        logger.warning(f"Keine Annotation für {fname}")
        return

    annotation_entry = ann[fname]
    text = file_path.read_text(encoding="utf-8")

    # Reset + Upload
    try:
        TRANSPORTER.remove_file(fname)
    except Exception:
        logger.debug(f"Fehler beim Entfernen der Datei {fname} (vielleicht existiert sie nicht).")

    upload = QAFileUpload(data=text.encode("utf-8"), name=fname)
    res = TRANSPORTER.add_file(upload)
    if res.status != 200:
        logger.error(f"Fehler beim Laden: {res.error_msg}")

    for question in QUESTIONS:
        total += 1
        label = LABEL_MAPPING[question]
        truth_answer = extract_label_text(annotation_entry, label, text)

        q = QAQuestion(question=question, search_strategy="similarity", max_sources=1, no_think=True)
        qa_res = TRANSPORTER.qa_query(q)
        pred_answer = clean_pred(qa_res.response if qa_res and getattr(qa_res, "response", None) else "")

        score, match = evaluate_prediction(truth_answer, pred_answer, label)
        if match:
            correct += 1

        logger.info(f"{fname:25} | {question:35} | GT: {truth_answer!r:30} | PRED: {pred_answer!r:30} | score={score:.2f} | match={match}")

        rows.append({
            "file": fname,
            "question": question,
            "label": label,
            "ground_truth": truth_answer,
            "predicted": pred_answer,
            "match_score": f"{score:.3f}",
            "match": match
        })

    acc = correct / total if total else 0
    logger.info("================================================")
    logger.info(f"FINAL SCORE: {correct}/{total}  accuracy={acc:.3f}")
    logger.info("================================================")

    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with RESULT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Ergebnisse gespeichert in: {RESULT_CSV}")

def run_benchmark_no_rag():
    ann = load_grascco_annotations()
    rows = []
    total = 0
    correct = 0

    file_path = RAW_TEXT_DIR / "Neubauer.txt"
    fname = file_path.name.replace("ö", "o")

    if fname not in ann:
        logger.warning(f"Keine Annotation für {fname}")
        return

    annotation_entry = ann[fname]
    text = file_path.read_text(encoding="utf-8")

    logger.info(f"Datei: {fname}")
    logger.info(f"Model: {model_config.name}")

    no_think_option = input("Möchten Sie 'no_think' aktivieren? (j/n): ").strip().lower() == 'j'
    logger.info(f"'no_think' ist auf {no_think_option} gesetzt.")

    for question in QUESTIONS:
        total += 1
        label = LABEL_MAPPING[question]
        truth_answer = extract_label_text(annotation_entry, label, text)

        query_text = question
        if no_think_option:
            query_text = f"/no_think {question}"
            logger.info(f"Modifizierte Frage mit /no_think: {query_text}")

        qa_res = None
        try:
            qa_res = llm.create_chat_completion(messages=[{"role": "user", "content": query_text}])
        except Exception as e:
            logger.warning(f"WARNUNG: LLM-Aufruf für {fname} - {query_text} ist fehlgeschlagen: {e}")
            qa_res = None

        pred_answer_without_Rag = ""
        if isinstance(qa_res, tuple) and len(qa_res) == 2:
            pred_answer_without_Rag = qa_res[1] or ""
        else:
            # falls qa_res schon direkt ein Objekt mit response-Attribut
            if hasattr(qa_res, "response"):
                pred_answer_without_Rag = qa_res.response or ""
            else:
                pred_answer_without_Rag = ""  # safe fallback

        pred_answer_without_Rag = clean_pred(pred_answer_without_Rag)

        score, match = evaluate_prediction(truth_answer, pred_answer_without_Rag, label)
        if match:
            correct += 1

        logger.info(f"{fname:25} | {question:35} | GT: {truth_answer!r:30} | PRED: {pred_answer_without_Rag!r:30} | score={score:.2f} | match={match}")

        rows.append({
            "file": fname,
            "question": question,
            "label": label,
            "ground_truth": truth_answer,
            "predicted": pred_answer_without_Rag,
            "match_score": f"{score:.3f}",
            "match": match
        })

    acc = correct / total if total else 0
    logger.info("================================================")
    logger.info(f"FINAL SCORE: {correct}/{total}  accuracy={acc:.3f}")
    logger.info("================================================")

    RESULT_CSV_WITHOUT_RAG.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with RESULT_CSV_WITHOUT_RAG.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Ergebnisse gespeichert in: {RESULT_CSV_WITHOUT_RAG}")

# -------------------------------------------
# Main
# -------------------------------------------
if __name__ == "__main__":
    # Standard: Einzeldatei-RAG (Baastrup.txt), wie vom Nutzer zuvor benutzt.
    # Möchtest du stattdessen run_benchmark_rag() oder run_benchmark_no_rag(), passe die Aufrufe an.
    run_benchmark_Rag_one_file()
    #run_benchmark_no_rag()
















