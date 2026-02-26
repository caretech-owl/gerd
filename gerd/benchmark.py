"""
Benchmarking-Skript for GRASCCO dataset with RAG and non-RAG approaches.
gool is to evaluate the performance of the QA system on specific questions with RAG and without RAG.
benchmark reasining vs non-reasining model related to patient information extraction from medical documents.
The results are logged into CSV files for further analysis.
"""

import json
import csv
from pathlib import Path
from pydoc import text
from typing import Dict, List, Tuple, Optional
from itertools import islice
import logging
import re
import os

import dateparser
from sympy import use

# Falls du die gerd-Module lokal hast, lasse die Imports; sonst bitte anpassen.
from gerd import rag
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


# questions.py
QUESTIONS = [
    "Wie heißt der Patient?",
    "Wann hat der Patient Geburtstag?",
    "Wann wurde der Patient bei uns aufgenommen?",
    "Wann wurde der Patient bei uns entlassen?",
]


LABEL_MAPPING = {
    "Wie heißt der Patient?": "PatientName",
    "Wann hat der Patient Geburtstag?": "PatientGeburtsdatum",
    "Wann wurde der Patient bei uns aufgenommen?": "AufnahmeDatum",
    "Wann wurde der Patient bei uns entlassen?": "EntlassungsDatum",
}


FUZZY_THRESHOLD = 0.95
RAW_TEXT_DIR = project_dir / "tests/data/grascco/raw"
RESULTS_DIR = project_dir / "results"
RESULT_CSV_RAG = RESULTS_DIR / "grascco_benchmark_rag.csv"
RESULT_CSV_RAG_ONE_FILE = RESULTS_DIR / "grascco_benchmark_rag_one_file.csv"
RESULT_CSV_WITHOUT_RAG = RESULTS_DIR / "grascco_benchmark_no_rag.csv"
RESULT_CSV_WITHOUT_RAG_ONE_FILE = RESULTS_DIR / "grascco_benchmark_no_rag_one_file.csv"

# load llm
model_config = load_qa_config().model
llm = load_model_from_config(model_config)

# -------------------------------------------
# Prefix-rules for each  Label to handle common patterns in predictions (e.g. "Patient: Anna Müller" for PatientName)
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




import re

BAD_PREFIXES = (
    "okay",
    "let's see",
    "the user is asking",
    "based on",
    "from the context",
    "wurde am",
    "entließ",
)

BAD_PREFIX_RE = re.compile(
    rf"^\s*(?:{'|'.join(map(re.escape, BAD_PREFIXES))})[\s,.:;-]*",
    re.IGNORECASE
)





THINK_BLOCK_RE = re.compile(r"<\s*tool_call\s*>.*?<\s*/\s*tool_call\s*>", re.IGNORECASE | re.DOTALL)

# ----------------------------
# 1. Regex-Definitionen für die Extraktion von Namen, Geburtstagen, Aufnahmedaten und Entlassungsdaten
# ----------------------------

# trigger for patient name: look for "Patient: Anna Müller" or "Der Patient heißt Anna Müller" etc.
NAME_TRIGGER = re.compile(
    r"\bpatient\b\s*[:\-]?\s*"
    r"(?:is|ist|named)?\s*"
    r"(?:herrn|herr|frau)?\s*",
    re.IGNORECASE
)


# Name-Regex: 1–3 Words, Unicode, starting with capital letter, allowing common name characters
NAME_RE = re.compile(
    r"\b([A-ZÄÖÜA-Za-z][a-zäöüßà-öø-ÿ]{2,}"
    r"(?:\s+[A-ZÄÖÜA-Za-z][a-zäöüßà-öø-ÿ]{2,}){0,2})\b",
    re.UNICODE
)

# These Titels should be excluded from names)
TITLE_RE = re.compile(
    r"\b(dr|doctor|prof|professor|mr|mrs|ms|md|phd)\b", re.IGNORECASE
)

# clinics / Organisation keywords, to exclude cases like "Patientin wurde in der Charité aufgenommen"
ORG_KEYWORDS = re.compile(
    r"\b(hospital|clinic|klinik|medical|center|centre|university|charité|health|care)\b",
    re.IGNORECASE
)

# eliminate some non-usefull words
NON_NAME_PREFIX_RE = re.compile(
    r"\b(it|the|mentions|looking|okay|this|that|these|those|user|wie|was|wo|wann)\b",
    re.IGNORECASE
)

# elliminate Words after dr-Titele 
DOCTOR_CUTOFF_RE = re.compile(
    r"\b(dr|doctor|prof|professor|md)\b",
    re.IGNORECASE
)

# ----------------------------
# 2. Preprocessing: Think-Tags entfernen
# ----------------------------
def remove_think_tags(text: str) -> str:
    """
    Entfernt nur die <think>-Tags, behält den Inhalt.
    """
    THINK_TAG_RE = re.compile(r"</?\s*think\s*>", re.IGNORECASE)
    return THINK_TAG_RE.sub("", text)

# ----------------------------
# 3. Validation funktion
# ----------------------------
def is_valid_person_name(name: str) -> bool:
    """Prüft, ob ein Name gültig ist (kein Titel, keine Klinik, kein Funktionswort)."""
    if TITLE_RE.search(name):
        return False
    if ORG_KEYWORDS.search(name):
        return False
    if NON_NAME_PREFIX_RE.match(name):
        return False
    if name.lower() == "think":  # Sicherheit gegen Artefakte
        return False
    return True

# ----------------------------
# 4. Extraktion funktion
# ----------------------------
def extract_name_from_text(text: str) -> str:
    # eliminate Think-Tags 
    cleaned_text = remove_think_tags(text)

    # eliminate words after  Arzt-Titeln 
    cutoff = DOCTOR_CUTOFF_RE.search(cleaned_text)
    if cutoff:
        cleaned_text = cleaned_text[:cutoff.start()]

    #  Trigger-Search
    trigger_match = NAME_TRIGGER.search(cleaned_text)
    if trigger_match:
        after_trigger = cleaned_text[trigger_match.end():]
        candidates = NAME_RE.findall(after_trigger)
        for name in candidates:
            if is_valid_person_name(name):
                return name.strip()

    # 4️⃣ Fallback: global search
    candidates = NAME_RE.findall(cleaned_text)
    for name in candidates:
        if is_valid_person_name(name):
            return name.strip()

    return "Nicht angegeben"




BIRTHDAY_TRIGGERS = re.compile(
    r"(?:"
    r"geburtsdatum|geb\.|geboren|geburtstag||"
    r"born on|date of birth|dob|birthday"
    r")"
    r"(?:\s+(?:am|im|on))?"
    r"[:\s]*",
    re.IGNORECASE
)


MONTHS = (
    "januar|februar|märz|maerz|april|mai|juni|juli|"
    "august|september|oktober|november|dezember|"
    "january|february|march|april|may|june|july|"
    "august|september|october|november|december"
)




BIRTHDAY_RE = re.compile(
    rf"\b("
    # 15. März 1980 / 15 März 1980
    rf"\d{{1,2}}\.?\s+(?:{MONTHS})\s+\d{{4}}"
    rf"|"
    # March 23, 1968
    rf"(?:{MONTHS})\s+\d{{1,2}},\s*\d{{4}}"
    rf"|"
    # 23.03.1968 / 23-03-68
    rf"\d{{1,2}}[.\-/]\d{{1,2}}[.\-/]\d{{2,4}}"
    rf"|"
    # 1968-03-23
    rf"\d{{4}}[.\-/]\d{{1,2}}[.\-/]\d{{1,2}}"
    rf")\b",
    re.IGNORECASE
)


def extract_birthday_from_text(text: str) -> str:
    # <tool_call>-Block observe
    think_matches = THINK_BLOCK_RE.findall(text)
    combined_text = " ".join(think_matches) if think_matches else text

    #  Trigger-Search
    trigger_match = BIRTHDAY_TRIGGERS.search(combined_text)
    #print("birthday trigger match:", trigger_match)
    if trigger_match:
        after_trigger = combined_text[trigger_match.end():]
        before_trigger = combined_text[:trigger_match.start()]  
        #print("after birthday trigger:", after_trigger)
        #print("before birthday trigger:", before_trigger)
        
        candidates = BIRTHDAY_RE.findall(after_trigger)
        #print("candidates after trigger:", candidates)
        if not candidates:
            candidates = BIRTHDAY_RE.findall(before_trigger)
            #print("candidates before trigger:", candidates)
        
        if candidates:
            return candidates[0].strip()
        

    # No Fallback anymore
    return "Nicht angegeben"    


RECORDING_TRIGGER = re.compile(
    r"\b(?:aufnahmedatum|aufnahme|admission date|treated|to|wurde|was checked from)"
    r"(?:\s+(?:am|on))?"
    r"[:\s]*",
    re.IGNORECASE
)

RECORDING_DATE_RE = re.compile(
    r"\b("
    # 15. März 1980 / 15 März 1980
    r"\d{1,2}\.?\s+"
    r"(?:januar|februar|märz|maerz|april|mai|juni|juli|"
    r"august|september|oktober|november|dezember)\s+\d{4}"
    r"|"
    # 23 March 1968
    r"\d{1,2}\s+"
    r"(?:january|february|march|april|may|june|july|"
    r"august|september|october|november|december)\s+\d{4}"
    r"|"
    # March 23, 1968 / Jan 24, 2028 / Sep 3, 2021
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+\d{1,2},\s+\d{4}"
    r"|"
    # 23.03.1968 / 23-03-68 / 23/03/1968
    r"\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}"
    r"|"
    # 1968-03-23
    r"\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}"
    r")\b",
    re.IGNORECASE
)




def extract_recording_date_from_text(text: str) -> str:
    # <tool_call>-Block berücksichtigen
    think_matches = THINK_BLOCK_RE.findall(text)
    combined_text = " ".join(think_matches) if think_matches else text

    # Nach Triggern suchen
    trigger_match = RECORDING_TRIGGER.search(combined_text)
    #print("recording date trigger match:", trigger_match)
    if trigger_match:
        after_trigger = combined_text[trigger_match.end():]
        #print("after recording date trigger:", after_trigger)
        
        candidates = RECORDING_DATE_RE.findall(after_trigger)
        #print("candidates after trigger:", candidates)
        
        if candidates:
            return candidates[0].strip()
        
    # Kein Fallback mehr
    return "Nicht angegeben"




RELEASE_DATE_TRIGGER = re.compile(
    r"\b(?:entlassungsdatum|entlassung|entlassen|entließ|release date|discharge date|released|discharged|to|bis zum)"
    r"(?:\s+(?:am|on))?"
    r"[:\s]*",
    re.IGNORECASE
)

RELEASE_DATE_RE = re.compile(
    r"\b("
    # 15. März 1980 / 15 März 1980
    r"\d{1,2}\.?\s+"
    r"(?:januar|februar|märz|maerz|april|mai|juni|juli|"
    r"august|september|oktober|november|dezember)\s+\d{4}"
    r"|"
    # 23 March 1968
    r"\d{1,2}\s+"
    r"(?:january|february|march|april|may|june|july|"
    r"august|september|october|november|december)\s+\d{4}"
    r"|"
    # March 23, 1968 / Jan 24, 2028 / Sep 3, 2021
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+\d{1,2},\s+\d{4}"
    r"|"
    # 23.03.1968 / 23-03-68 / 23/03/1968
    r"\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}"
    r"|"
    # 1968-03-23
    r"\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}"
    r")\b",
    re.IGNORECASE
)




def extract_release_date_from_text(text: str) -> str:
    # <tool_call>-Block berücksichtigen
    think_matches = THINK_BLOCK_RE.findall(text)
    combined_text = " ".join(think_matches) if think_matches else text

    # Nach Triggern suchen
    trigger_match = RELEASE_DATE_TRIGGER.search(combined_text)
    #print("release date trigger match:", trigger_match)
    if trigger_match:
        after_trigger = combined_text[trigger_match.end():]
        #print("after release date trigger:", after_trigger)

        candidates = RELEASE_DATE_RE.findall(after_trigger)
        #print("candidates after trigger:", candidates)
        
        if candidates:
            return candidates[0].strip()
        
    # No Fallback anymore
    return "Nicht angegeben"









def clean_answer_strict(value: str) -> str:
    if not value:
        return "Nicht angegeben"

    # 1. <tool_call>...<tool_call> eliminate
    value = THINK_BLOCK_RE.sub("", value)

    # 2. BAD_PREFIXES eliminate at Begin 
    value = BAD_PREFIX_RE.sub("", value)

    # 3.just the first non-empty line
    for line in value.splitlines():
        line = line.strip()
        if line:
            return line

    return "Nicht angegeben"





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




def make_prompt_german(text: str, question: str, no_think: bool) -> str:
    question = f"/no_think {question}" if no_think else question

    return f"""Gib ausschließlich den exakten Wortlaut zurück, wie er im Text vorkommt.
Keine vollständigen Sätze. Keine Erklärungen. Keine zusätzlichen Wörter.
Gib nur den wörtlich relevanten Text zurück.
Antworte mit 'Unbekannt', falls die Information im Text nicht enthalten ist.


Text:
{text}

Frage:
{question}
""".strip()


def make_prompt_englich(text: str, question: str, no_think: bool) -> str:
    question = f"/no_think {question}" if no_think else question

    return f"""Return only the exact wording as it appears in the text.
No full sentences. No explanations. No additional words.
Return the literal relevant text only.
Answer with 'Unknown' if the information is not present in the text.


Text:
{text}

Question:
{question}
""".strip()

# -------------------------------------------
# BENCHMARK-Funktions 
# -------------------------------------------






def run_benchmark(max_files: int, use_rag: bool, no_think_option: bool):
    """Run benchmark with RAG approach.
    tests a set of files against the provided questions .
    truth answers are extracted from the grascco annotations.
    compares the returnned predicted answers with truth answers using evaluate_prediction.
    computes accuracy over all questions and files 
        and logs the results into a CSV file.
        
        Parameters:
            max_files (int): Maximum number of files to process.
            use_rag (bool): Whether to use RAG or not.
            no_think_option (bool): Whether to use the 'no_think' option in the QAQuestion.

        Returns:
            None
    """
    _LOGGER.info("**************Benchmark___Rag*********")
    _LOGGER.info(f"Model: {model_config.name}")
    _LOGGER.info(f"'no_think': {no_think_option}")
    ann = load_grascco_annotations()
    rows = []
    total = 0
    correct = 0

    files = islice(sorted(RAW_TEXT_DIR.glob("*.txt")), max_files)

    for file_path in files:
        fname = file_path.name.replace("ö", "o")

        if fname not in ann:
            _LOGGER.warning(f"No annotation for {fname}")
            continue

        annotation_entry = ann[fname]
        text = file_path.read_text(encoding="utf-8")

        # ---------------------------
        # Reset + Upload Vectorstore
        # ---------------------------
        if use_rag:
            try:
                TRANSPORTER.clear_vectorstore()
            except Exception:
                _LOGGER.exception("Failed to reset vectorstore")
                continue
            upload = QAFileUpload(data=text.encode("utf-8"), name=fname)
            res = TRANSPORTER.add_file(upload)
            if getattr(res, "status", None) != 200:
                _LOGGER.error(f"Upload failed for {fname}")
                continue

        if use_rag:
            for question in QUESTIONS:
                total += 1
                label = LABEL_MAPPING[question]
                truth_answer = extract_label_text(annotation_entry, label, text)
                prompt = make_prompt_englich(text=text, question=question, no_think=no_think_option)
                #_LOGGER.info(prompt)
                q = QAQuestion(question = prompt, search_strategy="similarity", max_sources=3,no_think=no_think_option)
            
                qa_res = TRANSPORTER.qa_query(q)
                pred_answer = qa_res.response 
        else:
            for question in QUESTIONS:
                total += 1
                label = LABEL_MAPPING[question]
                truth_answer = extract_label_text(annotation_entry, label, text)
                prompt = make_prompt_englich(text=text, question=question, no_think=no_think_option)
                try:
                    qa_res = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}])
                except Exception as e:
                    _LOGGER.warning(f"WARNUNG: LLM-Aufruf für {fname} ist fehlgeschlagen: {e}")
                    qa_res = None

                pred_answer = ""
                if isinstance(qa_res, tuple) and len(qa_res) == 2:
                    pred_answer = qa_res[1] or ""
                else:
                    if hasattr(qa_res, "response"):
                        pred_answer = qa_res.response or ""
                    elif isinstance(qa_res, dict) and "response" in qa_res:
                        pred_answer = qa_res.get("response") or ""
                    else:
                        pred_answer = ""

        #pred = clean_pred(pred_answer)


            score, match = evaluate_prediction(truth_answer, pred_answer, label)
            
            if match:
                correct += 1

            _LOGGER.info( f"{fname:25} | {label:18} | Q: {question:35} | GT: {truth_answer!r:25} | "
                          f"PRED: {pred_answer!r:25} | score={score:.2f} | match={match}"
            )

            rows.append({
                "file": fname,
                "label": label,
                "question_used": question,
                "ground_truth": truth_answer,
                "predicted": pred_answer,
                "match_score": f"{score:.3f}",
                "match": match,
            })

    # ---------------------------
    # Final metrics
    # ---------------------------
    acc = correct / total if total else 0.0
    _LOGGER.info("================================================")
    _LOGGER.info(f"FINAL SCORE: {correct}/{total}  accuracy={acc:.3f}")
    _LOGGER.info("================================================")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        with RESULT_CSV_RAG.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        _LOGGER.info(f"Results saved in: {RESULT_CSV_RAG}")




def run_benchmark_one_file(file_path: Path, use_rag: bool, no_think_option: bool):
    
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
    _LOGGER.info("************************Benchmark__one__file************************")


    ann = load_grascco_annotations()
    rows = []
    total = 0
    correct = 0

    
    fname = file_path.name.replace("ö", "o")

    if fname not in ann:
        _LOGGER.warning(f"Keine Annotation für {fname}")
        return
    annotation_entry = ann[fname]
    text = file_path.read_text(encoding="utf-8")

    _LOGGER.info(f"Datei: {fname}")
    _LOGGER.info(f"Model: {model_config.name}")
    _LOGGER.info(f"'no_think' ist auf {no_think_option} gesetzt.")

    if use_rag:
        try:
            TRANSPORTER.remove_file(fname)
        except Exception:
            _LOGGER.debug(f"Fehler beim Entfernen der Datei {fname} (vielleicht existiert sie nicht).")

        upload = QAFileUpload(data=text.encode("utf-8"), name=fname)
        res = TRANSPORTER.add_file(upload)
        if getattr(res, "status", None) != 200:
            _LOGGER.error(f"Fehler beim Laden: {getattr(res, 'error_msg', res)}")

    if use_rag:
        for question in QUESTIONS:
            total += 1
            label = LABEL_MAPPING[question]
            truth_answer = extract_label_text(annotation_entry, label, text)
            prompt = make_prompt_englich(text, question, no_think_option)
            #_LOGGER.info("prompt: %s", prompt)
            qa_question = QAQuestion(question= prompt ,  search_strategy="similarity", max_sources=3, no_think= no_think_option)  
            qa_answer = TRANSPORTER.qa_query(qa_question) 
            pred_answer = qa_answer.response
            #_LOGGER.info("RAG-Antwort: %s", pred_answer)
            
    else:
        for question in QUESTIONS:
            total += 1
            label = LABEL_MAPPING[question]
            truth_answer = extract_label_text(annotation_entry, label, text)
            prompt = make_prompt_englich(text, question, no_think_option)    
        try:
            qa_res = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}])
        except Exception as e:
            _LOGGER.warning(f"WARNUNG: LLM-Aufruf für {fname} ist fehlgeschlagen: {e}")
            qa_res = None

        pred = ""
        if isinstance(qa_res, tuple) and len(qa_res) == 2:
            pred = qa_res[1] or ""
        else:
            if hasattr(qa_res, "response"):
                pred = qa_res.response or ""
            elif isinstance(qa_res, dict) and "response" in qa_res:
                pred = qa_res.get("response") or ""
            else:
                pred = ""

        #pred = clean_pred(pred)

        score, match = evaluate_prediction(truth_answer, pred, label)
        if match:
            correct += 1

        _LOGGER.info(f"{fname:25} | {question:35} | GT: {truth_answer!r:30} | PRED: {pred!r:30} | score={score:.2f} | match={match}")

        rows.append({
            "file": fname,
            "question": question,
            "label": label,
            "ground_truth": truth_answer,
            "predicted": pred,
            "match_score": f"{score:.3f}",
            "match": match,
        })

    acc = correct / total if total else 0
    _LOGGER.info("================================================")
    _LOGGER.info(f"FINAL SCORE: {correct}/{total}  accuracy={acc:.3f}")
    _LOGGER.info("================================================")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        with RESULT_CSV_WITHOUT_RAG_ONE_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        _LOGGER.info(f"Ergebnisse gespeichert in: {RESULT_CSV_WITHOUT_RAG_ONE_FILE}")



def profile(file_path: Path,use_rag: bool,no_think: bool):
    """
    Beantwortet definierte Fragen zu einem Entlassungsbrief
    und speichert alle Antworten gesammelt in einer Textdatei.
    """
    print(f"********Profile*********")
    text = file_path.read_text(encoding="utf-8")
    _LOGGER.info(f"Datei: {file_path.name}")
    _LOGGER.info(f"Model: {model_config.name}")
    _LOGGER.info("RAG: %s", use_rag)
    _LOGGER.info("no_think: %s", no_think)
    answers: dict[str, str] = {}

    # =========================
    # RAG-SETUP
    # =========================
    if use_rag:
        try:
            TRANSPORTER.clear_vectorstore()
        except Exception:
            pass

        upload = QAFileUpload(data=text.encode("utf-8"), name=file_path.name)
        res = TRANSPORTER.add_file(upload)
        if getattr(res, "status", None) != 200:
            raise RuntimeError("Upload in Vectorstore fehlgeschlagen")

    # =========================
    # FRAGEN-SCHLEIFE
    # =========================
    for question in QUESTIONS:
        if use_rag:
            prompt = make_prompt_englich(text, question, no_think)
            q = QAQuestion( 
                question= prompt,
                search_strategy="similarity",
                max_sources=3,
                no_think=no_think
            )
            
            res = TRANSPORTER.qa_query(q)
            raw_answer = res.response
            answers[question] = raw_answer
            #print(" *****Antwort*****", answers[question])

        else:
            prompt = make_prompt_englich(text, question, no_think)
            res = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )

            if isinstance(res, tuple) and len(res) == 2:
                raw_answer = res[1] or ""

            else:
                raw_answer = getattr(res, "response", "") or ""
            answers[question] = raw_answer

        #answers[question] = raw_answer
        #print(" *****Alle Antworten*****", answers)
    
    # =========================
    # SORTIERTE TEXTAUSGABE
    # =========================
    print("************")
    print(answers.get("Wie heißt der Patient?", "Nicht angegeben"))
    print("************")
    print(answers.get("Wann hat der Patient Geburtstag?", "Nicht angegeben"))
    print("************")
    print(answers.get("Wann wurde der Patient bei uns aufgenommen?", "Nicht angegeben"))
    print("************")
    print(answers.get("Wann wurde der Patient bei uns entlassen?", "Nicht angegeben"))
    lines = [
        "Kurzprofil Entlassungsbrief\n",
        f"Patient: {extract_name_from_text(answers.get('Wie heißt der Patient?', 'Nicht angegeben'))}",
        f"Geburtsdatum: {extract_birthday_from_text(answers.get('Wann hat der Patient Geburtstag?', 'Nicht angegeben'))}",
        f"Aufnahme: {extract_recording_date_from_text(answers.get('Wann wurde der Patient bei uns aufgenommen?', 'Nicht angegeben'))}",
        f"Entlassung: {extract_release_date_from_text(answers.get('Wann wurde der Patient bei uns entlassen?', 'Nicht angegeben'))}",
    ]

    summary = "\n".join(lines).strip()
    print(summary)

    # =========================
    # SPEICHERN
    # =========================
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{file_path.stem}_profile.txt"
    out_path.write_text(summary, encoding="utf-8")
    _LOGGER.info("Profil gespeichert unter: %s", out_path)


    









# -------------------------------------------
# Main
# -------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s")

    #run_benchmark(max_files = 5, use_rag=True, no_think_option= True)
    #run_benchmark_one_file(file_path=RAW_TEXT_DIR / "Cajal.txt", use_rag=True, no_think_option=False)
    profile(file_path=RAW_TEXT_DIR / "Cajal.txt", use_rag=True, no_think= True)
    
