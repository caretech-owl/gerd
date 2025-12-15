"""
Benchmarking-Skript for GRASCCO dataset with RAG and non-RAG approaches.
gool is to evaluate the performance of the QA system on specific questions with RAG and without RAG.
benchmark reasining vs non-reasining model related to patient information extraction from medical documents.
The results are logged into CSV files for further analysis.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import islice
import logging
import re
import os

import dateparser

# Falls du die gerd-Module lokal hast, lasse die Imports; sonst bitte anpassen.
from gerd.backends import TRANSPORTER
from gerd.transport import QAFileUpload, QAQuestion
from gerd.loader import load_model_from_config
from gerd.config import load_qa_config

# ----------------------------------------------------
# Basic logging configuration
# ----------------------------------------------------
_LOGGER = logging.getLogger("gerd.benchmark")
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
    "Wann wurde der Patient entlassen?": "EntlassungsDatum",
}

FUZZY_THRESHOLD = 0.95
RAW_TEXT_DIR = project_dir / "tests/data/grascco/raw"
RESULTS_DIR = project_dir / "results"
RESULT_CSV_RAG = RESULTS_DIR / "grascco_benchmark_rag.csv"
RESULT_CSV_RAG_ONE_FILE = RESULTS_DIR / "grascco_benchmark_rag_one_file.csv"
RESULT_CSV_WITHOUT_RAG = RESULTS_DIR / "grascco_benchmark_no_rag.csv"
RESULT_CSV_WITHOUT_RAG_ONE_FILE = RESULTS_DIR / "grascco_benchmark_no_rag_one_file.csv"

# LLM laden (wie vorher)
model_config = load_qa_config().model
llm = load_model_from_config(model_config)

# -------------------------------------------
# Prefix-Regeln je Label (erweiterbar) - korrekte Keys
# -------------------------------------------
LABEL_PREFIXES = {
    "PatientName": [
        "patient:",
        "patientin:",
        "der patient heißt",
        "der patient heisst",
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
    "AufnahmeDatum": [
        "aufnahmedatum:",
        "patient wurde aufgenommen am",
        "aufgenommen am",
        "aufgenommen:",
    ],
    "EntlassungsDatum": [
        "entlassungsdatum:",
        "patient wurde entlassen am",
        "entlassen am",
        "entlassen:",
    ],
}

# -------------------------------------------
# Helpers 
# -------------------------------------------

def normalize(s: str) -> str:
    return " ".join((s or "").lower().strip().split())


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


def safe_slice_text(text: Optional[str], start: int, end: int) -> str:
    if not text:
        return ""
    start = max(0, int(start))
    end = min(len(text), int(end))
    if start >= end:
        return ""
    return text[start:end]


def normalize_date(s: Optional[str]):
    """Parst Strings zu date-Objekten (dateparser). Gibt None zurück, wenn kein Datum gefunden wurde."""
    if not s:
        return None
    try:
        # dateparser ist sehr flexibel; wir forcen deutsche Sprache als Hint
        dt = dateparser.parse(s, languages=["de"])
        if not dt:
            return None
        return dt.date()
    except Exception:
        return None


DATE_NUMERIC_RE = re.compile(r"\b\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}\b")
DATE_DAYMONTH_RE = re.compile(r"\b\d{1,2}\.\s*[A-Za-zäöüÄÖÜß]+\s*\d{4}\b")


def extract_date(text: Optional[str]):
    """Versucht zuerst explizite numerische Datumsformate zu finden, dann Monatsnamen, dann fuzzy parse."""
    if not text:
        return None
    t = str(text)
    # 1) numerics like 21.01.2023 or 21/01/2023
    m = DATE_NUMERIC_RE.search(t)
    if m:
        return normalize_date(m.group(0))
    # 2) day + monthname + year (z.B. 21. Januar 2023)
    m2 = DATE_DAYMONTH_RE.search(t)
    if m2:
        return normalize_date(m2.group(0))
    # 3) attempt fuzzy parse
    return normalize_date(t)


def clean_gt_value(label: str, value: Optional[str]) -> str:
    v = (value or "").strip()
    if not v:
        return ""
    if label == "PatientName":
        v = re.sub(r"^(tr\.|hr\.|fr\.|herr|frau)\s*:??\s*", "", v, flags=re.I).strip()
        return v
    if label == "PatientGeburtsdatum":
        v = re.sub(r"^(geb\.|geburtsdatum:)\s*", "", v, flags=re.I).strip()
        return v
    return v


def remove_titles(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\b(?:herr|frau|hr\.|fr\.|dr\.?|dr)\b", "", s, flags=re.I).strip()


# -------------------------------------------
# robust extraction from GRASCCO annotations
# -------------------------------------------

def load_grascco_annotations() -> Dict[str, dict]:
    json_path = project_dir / "tests/data/grascco/grascco.json"
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[str, dict] = {}
    for entry in data:
        file_upload = entry.get("file_upload", "") or ""
        name = file_upload.split("-", 1)[-1] if "-" in file_upload else file_upload
        # normalize simple umlauts for file matching
        name_norm = name.replace("ö", "o").replace("ä", "a").replace("ü", "u")
        mapping[name_norm] = entry
    return dict(sorted(mapping.items()))


def extract_label_text(annotation_entry: dict, label: str, text: str) -> str:
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
        # direct slice
        try:
            if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
                candidate = safe_slice_text(text, start, end).strip()
                if candidate:
                    return candidate
        except Exception:
            pass

        # try small offsets
        t_norm = text.replace("\r\n", "\n")
        for delta in range(-12, 13):
            s = max(0, (start or 0) + delta)
            e = min(len(t_norm), (end or 0) + delta)
            if s < e:
                part = t_norm[s:e].strip()
                if part and len(part) <= 200:
                    return part

        # fallback: token neighborhood
        tokens = re.findall(r"\S+", t_norm)
        if tokens:
            char_pos = 0
            for idx, tok in enumerate(tokens):
                tok_start = char_pos
                tok_end = char_pos + len(tok)
                if isinstance(start, int) and tok_start <= start <= tok_end:
                    snippet = " ".join(tokens[max(0, idx - 2): min(len(tokens), idx + 3)])
                    return snippet.strip()
                char_pos = tok_end + 1
        return ""
    return ""


# -------------------------------------------
# Prediction cleaning
# -------------------------------------------

def clean_pred(answer: Optional[str]) -> str:
    if not answer:
        return ""
    answer = answer.strip()
    lower = answer.lower()

    # try to extract common numeric date patterns first
    m = DATE_NUMERIC_RE.search(answer)
    if m:
        return m.group(0)
    m2 = DATE_DAYMONTH_RE.search(answer)
    if m2:
        return m2.group(0)

    # common declarative prefixes
    bad_prefixes = [
        "der patient wurde aufgenommen am",
        "der patient wurde entlassen am",
        "der patient heißt",
        "der patient heisst",
        "die patientin heißt",
        "patientin:",
        "patient:",
        "antwort:",
        "unbekannt",
        "nicht angegeben",
    ]

    for prefix in bad_prefixes:
        if lower.startswith(prefix):
            return answer[len(prefix):].strip(" :.,")

    # if nothing matched, return stripped answer
    return answer


# -------------------------------------------
# Evaluation / Matching
# -------------------------------------------

def evaluate_prediction(gt: str, pred: str, label: str) -> Tuple[float, bool]:
    gt_raw = clean_gt_value(label, gt or "")
    pred_raw = (pred or "")

    gt_norm = normalize(gt_raw)
    pred_norm = normalize(pred_raw)

    # 1) empty GT => no match
    if not gt_norm:
        return 0.0, False

    # 2) Prefix-basiertes Exact-Match (falls prediction "patient: Anna")
    if label in LABEL_PREFIXES:
        for prefix in LABEL_PREFIXES[label]:
            p = prefix.lower()
            if pred_norm.startswith(p):
                rest = pred_norm[len(p):].strip()
                if label == "PatientName":
                    rest_clean = normalize(remove_titles(rest))
                    if rest_clean == normalize(remove_titles(gt_norm)):
                        return 1.0, True
                if label in {"PatientGeburtsdatum", "AufnahmeDatum", "EntlassungsDatum"}:
                    d1 = extract_date(gt_raw)
                    d2 = extract_date(pred_raw)
                    if d1 and d2 and d1 == d2:
                        return 1.0, True

    # 3) Datum-Labels: extrahiere und vergleiche robust
    if label in {"PatientGeburtsdatum", "AufnahmeDatum", "EntlassungsDatum"}:
        d1 = extract_date(gt_raw)
        d2 = extract_date(pred_raw)
        if d1 and d2 and d1 == d2:
            return 1.0, True
        # fallback: try normalized string equality (z.B. same dd.mm.yyyy)
        if gt_norm and pred_norm and gt_norm == pred_norm:
            return 1.0, True

    # 4) Name-Label: heuristiken + levensthein
    if label == "PatientName":
        gt_clean = normalize(remove_titles(gt_norm))
        pred_clean = normalize(remove_titles(pred_norm))

        if gt_clean and pred_clean and (gt_clean in pred_clean or pred_clean in gt_clean):
            return 1.0, True

        if gt_clean and pred_clean and set(gt_clean.split()) == set(pred_clean.split()):
            return 1.0, True

        score = levenshtein_ratio(gt_clean, pred_clean)
        return score, score >= FUZZY_THRESHOLD

    # 5) Allgemeiner Fallback
    score = levenshtein_ratio(gt_norm, pred_norm)
    return score, score >= FUZZY_THRESHOLD


# -------------------------------------------
# Prompt
# -------------------------------------------

def make_prompt(text: str, question: str) -> str:
    return f"""Gib ausschließlich den exakten Wortlaut zurück, wie er im Text vorkommt.
Keine Sätze. Keine Erklärungen. Nur den wörtlichen relevanten Text.
Antworte mit 'Unbekannt', wenn die Information im Text nicht vorhanden ist.
Auf deutsche Sprache antworten.

Text:
{text}

Frage:
{question}
""".strip()



def make_prompt_no_rag(text:str, question:str, no_think: bool) -> str:
    question = f"/no_think {question}" if no_think else question
    
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
# BENCHMARK-Funktions (RAG / Single-File RAG / No-RAG)
# -------------------------------------------


def run_benchmark_rag(max_files: int , no_think_option: bool):
    """Run benchmark with RAG approach.
    tests a set of files against the provided questions .
    truth answers are extracted from the grascco annotations.
    compares the returnned predicted answers with truth answers using evaluate_prediction.
    computes accuracy over all questions and files 
        and logs the results into a CSV file.
        
        Parameters:
            max_files (int): Maximum number of files to process.
            no_think_option (bool): Whether to use the 'no_think' option in the QAQuestion.

        Returns:
            None
    """
   
    ann = load_grascco_annotations()
    rows = []
    total = 0
    correct = 0

    files = islice(sorted(RAW_TEXT_DIR.glob("*.txt")), max_files)

    for file_path in files:
        fname = file_path.name.replace("ö", "o")
        if fname not in ann:
            _LOGGER.warning(f"no annotation for {fname}")
            continue

        annotation_entry = ann[fname]
        text = file_path.read_text(encoding="utf-8")

        # Reset + Upload
        try:
            _LOGGER.info("reset Vectorstore...")
            TRANSPORTER.clear_vectorstore()  # oder reset_vectorstore()
            _LOGGER.info("Vectorstore succesfuly reseted.")
        except Exception as e:
            _LOGGER.error("Error by reset vectorstore", exc_info=True)
            raise e


        upload = QAFileUpload(data=text.encode("utf-8"), name=fname)
        res = TRANSPORTER.add_file(upload)
        if getattr(res, "status", None) != 200:
            _LOGGER.error(f"Error by loading: {getattr(res, 'error_msg', res)}")
            continue

        for question in QUESTIONS:
            total += 1
            label = LABEL_MAPPING[question]
            truth_answer = extract_label_text(annotation_entry, label, text)

            #prompt = make_prompt(text, question)
            q = QAQuestion(question=question, search_strategy="similarity", max_sources=3, no_think= no_think_option)
            qa_res = TRANSPORTER.qa_query(q)
            pred_answer = qa_res.response
            #pred_answer = clean_pred(getattr(qa_res, "response", ""))

            score, match = evaluate_prediction(truth_answer, pred_answer, label)
            if match:
                correct += 1

            _LOGGER.info(f"{fname:25} | {question:35} | GT: {truth_answer!r:30} | PRED: {pred_answer!r:30} | score={score:.2f} | match={match}")

            rows.append({
                "file": fname,
                "question": question,
                "label": label,
                "ground_truth": truth_answer,
                "predicted": pred_answer,
                "match_score": f"{score:.3f}",
                "match": match,
            })

    acc = correct / total if total else 0
    _LOGGER.info("================================================")
    _LOGGER.info(f"FINAL SCORE: {correct}/{total}  accuracy={acc:.3f}")
    _LOGGER.info("================================================")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        with RESULT_CSV_RAG.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        _LOGGER.info(f"Resultes saved in: {RESULT_CSV_RAG}")


def run_benchmark_Rag_one_file(no_think_option: bool):
    
    """Run benchmark with RAG approach for any file separatly .
    test one file against the provided questions .
    truth answers are extracted from the grascco annotations.
    compares the returnned predicted answers with truth answers using evaluate_prediction.
    computes accuracy over all questions and files 
        and logs the results into a CSV file.
        
        Parameters:
            no_think_option (bool): Whether to use the 'no_think' option in the QAQuestion.

        Returns:
            None
    """
     
    _LOGGER.info("**************Benchmark___Rag___one___file*********")
    ann = load_grascco_annotations()
    rows = []
    total = 0
    correct = 0

    file_path = RAW_TEXT_DIR / "Neubauer.txt"
    fname = file_path.name.replace("ö", "o")
    _LOGGER.info(f"Datei: {fname}")
    _LOGGER.info(f"Model: {model_config.name}")
    _LOGGER.info(f"'no_think': {no_think_option}")   

    if fname not in ann:
        _LOGGER.warning(f"Keine Annotation für {fname}")
        return

    annotation_entry = ann[fname]
    text = file_path.read_text(encoding="utf-8")

    # Reset + Upload
    try:
        TRANSPORTER.remove_file(fname)
    except Exception:
        _LOGGER.debug(f"Fehler beim Entfernen der Datei {fname} (vielleicht existiert sie nicht).")

    upload = QAFileUpload(data=text.encode("utf-8"), name=fname)
    res = TRANSPORTER.add_file(upload)
    if getattr(res, "status", None) != 200:
        _LOGGER.error(f"Fehler beim Laden: {getattr(res, 'error_msg', res)}")

    for question in QUESTIONS:
        total += 1
        label = LABEL_MAPPING[question]
        truth_answer = extract_label_text(annotation_entry, label, text)
        #prompt = make_prompt(text, question)
        qa_question = QAQuestion(question=question ,  search_strategy="similarity", max_sources=3, no_think= no_think_option)
        qa_answer = TRANSPORTER.qa_query(qa_question) 
       
        
        pred_answer = qa_answer.response
        #pred_answer = clean_pred(getattr(qa_res, "response", ""))

        score, match = evaluate_prediction(truth_answer, pred_answer, label)
        if match:
            correct += 1

        _LOGGER.info(f"{fname:25} | {question:35} | GT: {truth_answer!r:30} | PRED: {pred_answer!r:30} | score={score:.2f} | match={match}")

        rows.append({
            "file": fname,
            "question": question,
            "label": label,
            "ground_truth": truth_answer,
            "predicted": pred_answer,
            "match_score": f"{score:.3f}",
            "match": match,
        })

    acc = correct / total if total else 0
    _LOGGER.info("================================================")
    _LOGGER.info(f"FINAL SCORE: {correct}/{total}  accuracy={acc:.3f}")
    _LOGGER.info("================================================")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        with RESULT_CSV_RAG_ONE_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        _LOGGER.info(f"Ergebnisse gespeichert in: {RESULT_CSV_RAG_ONE_FILE}")


def run_benchmark_no_rag(max_files: int ,  no_think_option: bool):
    
    """Run benchmark withot RAG approach using llm directly.
    acsess the llm directly without qa_service 
    tests a set of files against the provided questions .
    truth answers are extracted from the grascco annotations.
    compares the returnned predicted answers with truth answers using evaluate_prediction.
    computes accuracy over all questions and files 
        and logs the results into a CSV file.
        
        Parameters:
            max_files (int): Maximum number of files to process.
            no_think_option (bool): Whether to use the 'no_think' option in the QAQuestion.

        Returns:
            None
    """
    _LOGGER.info("************************Benchmark___no___Rag************************")
    ann = load_grascco_annotations()
    rows = []
    total = 0
    correct = 0
    files = islice(sorted(RAW_TEXT_DIR.glob("*.txt")), max_files) 
    for file_path in files:
        fname = file_path.name.replace("ö", "o")
        if fname not in ann:
            _LOGGER.warning(f"Keine Annotation für {fname}")
            continue

        annotation_entry = ann[fname]
        text = file_path.read_text(encoding="utf-8")

        _LOGGER.info(f"Datei: {fname}")
        _LOGGER.info(f"Model: {model_config.name}")
        _LOGGER.info(f"'no_think' ist auf {no_think_option} gesetzt.")

        for question in QUESTIONS:
            total += 1
            label = LABEL_MAPPING[question]
            truth_answer = extract_label_text(annotation_entry, label, text)
            prompt = make_prompt_no_rag(text, question, no_think_option)

            try:
                qa_res = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}])
            except Exception as e:
                _LOGGER.warning(f"WARNUNG: LLM-Aufruf für {fname} ist fehlgeschlagen: {e}")
                qa_res = None

            pred_answer_without_Rag = ""
            if isinstance(qa_res, tuple) and len(qa_res) == 2:
                pred_answer_without_Rag = qa_res[1] or ""
            else:
                if hasattr(qa_res, "response"):
                    pred_answer_without_Rag = qa_res.response or ""
                elif isinstance(qa_res, dict) and "response" in qa_res:
                    pred_answer_without_Rag = qa_res.get("response") or ""
                else:
                    pred_answer_without_Rag = ""

            score, match = evaluate_prediction(truth_answer, pred_answer_without_Rag, label)
            if match:
                correct += 1

            _LOGGER.info(f"{fname:25} | {question:35} | GT: {truth_answer!r:30} | PRED: {pred_answer_without_Rag!r:30} | score={score:.2f} | match={match}")
            #_LOGGER.info("------------------------------------------------")
            rows.append({
                "file": fname,
                "question": question,
                "label": label,
                "ground_truth": truth_answer,
                "predicted": pred_answer_without_Rag,
                "match_score": f"{score:.3f}",
                "match": match,
            })
    acc = correct / total if total else 0
    _LOGGER.info("================================================")
    _LOGGER.info(f"FINAL SCORE: {correct}/{total}  accuracy={acc:.3f}")
    _LOGGER.info("================================================")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        with RESULT_CSV_WITHOUT_RAG.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        _LOGGER.info(f"Ergebnisse gespeichert in: {RESULT_CSV_WITHOUT_RAG}")



def run_benchmark_no_rag_one_file(no_think_option: bool):
    
    """Run benchmark withot RAG approach for any file accessing llm directly for one file.
    acsess the llm directly without qa_service
    test one file against the provided questions .
    truth answers are extracted from the grascco annotations.
    compares the returnned predicted answers with truth answers using evaluate_prediction.
    computes accuracy over all questions and files 
        and logs the results into a CSV file.
        
        Parameters:
            no_think_option (bool): Whether to use the 'no_think' option in the QAQuestion.

        Returns:
            None
    """
    _LOGGER.info("************************Benchmark___no___Rag__one__file************************")


    ann = load_grascco_annotations()
    rows = []
    total = 0
    correct = 0

    file_path = RAW_TEXT_DIR / "Beuerle.txt"
    fname = file_path.name.replace("ö", "o")

    if fname not in ann:
        _LOGGER.warning(f"Keine Annotation für {fname}")
        return

    annotation_entry = ann[fname]
    text = file_path.read_text(encoding="utf-8")

    _LOGGER.info(f"Datei: {fname}")
    _LOGGER.info(f"Model: {model_config.name}")
    _LOGGER.info(f"'no_think' ist auf {no_think_option} gesetzt.")
    

    for question in QUESTIONS:
        total += 1
        label = LABEL_MAPPING[question]
        truth_answer = extract_label_text(annotation_entry, label, text)
        prompt = make_prompt_no_rag(text, question, no_think_option)
        #_LOGGER.info("prompt: %s", prompt)


        
        try:
            qa_res = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}])
        except Exception as e:
            _LOGGER.warning(f"WARNUNG: LLM-Aufruf für {fname} ist fehlgeschlagen: {e}")
            qa_res = None

        pred_answer_without_Rag = ""
        if isinstance(qa_res, tuple) and len(qa_res) == 2:
            pred_answer_without_Rag = qa_res[1] or ""
        else:
            if hasattr(qa_res, "response"):
                pred_answer_without_Rag = qa_res.response or ""
            elif isinstance(qa_res, dict) and "response" in qa_res:
                pred_answer_without_Rag = qa_res.get("response") or ""
            else:
                pred_answer_without_Rag = ""

        #pred_answer_without_Rag = clean_pred(pred_answer_without_Rag)

        score, match = evaluate_prediction(truth_answer, pred_answer_without_Rag, label)
        if match:
            correct += 1

        _LOGGER.info(f"{fname:25} | {question:35} | GT: {truth_answer!r:30} | PRED: {pred_answer_without_Rag!r:30} | score={score:.2f} | match={match}")

        rows.append({
            "file": fname,
            "question": question,
            "label": label,
            "ground_truth": truth_answer,
            "predicted": pred_answer_without_Rag,
            "match_score": f"{score:.3f}",
            "match": match,
        })

    acc = correct / total if total else 0
    _LOGGER.info("================================================")
    _LOGGER.info(f"FINAL SCORE: {correct}/{total}  accuracy={acc:.3f}")
    _LOGGER.info("================================================")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        with RESULT_CSV_WITHOUT_RAG.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        _LOGGER.info(f"Ergebnisse gespeichert in: {RESULT_CSV_WITHOUT_RAG_ONE_FILE}")


# -------------------------------------------
# Main
# -------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s"
    )

    #run_benchmark_rag(max_files = 5, no_think_option= True)
    #run_benchmark_Rag_one_file(no_think_option=True)
    run_benchmark_no_rag(max_files = 5 , no_think_option=True)
    #run_benchmark_no_rag_one_file(no_think_option=True)
    
