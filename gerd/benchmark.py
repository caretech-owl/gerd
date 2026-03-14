"""Benchmarking.py: benchmark between diffrent llms and aproaches.

for GRASCCO dataset with RAG and non-RAG approaches.
gool is to evaluate the performance of the QA system
on specific questions
with RAG and without RAG and
with reasining and non-reasining model
related to patient information extraction from medical documents.
The results are logged into CSV files for further analysis.
"""

import csv
import json
import logging
import re
from contextlib import suppress
from datetime import date
from itertools import islice
from pathlib import Path
from typing import Dict, Optional, Tuple

from gerd.backends import TRANSPORTER
from gerd.config import load_qa_config
from gerd.loader import load_model_from_config
from gerd.transport import QAFileUpload, QAQuestion

# Basic logging configuration
_LOGGER = logging.getLogger("gerd.benchmark")
project_dir = Path(__file__).parent.parent


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


# Prefix-rules for each  Label to handle common patterns in predictions
# (e.g. "Patient: Anna Müller" for PatientName)

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

# Helpers
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
    rf"^\s*(?:{'|'.join(map(re.escape, BAD_PREFIXES))})[\s,.:;-]*", re.IGNORECASE
)


THINK_BLOCK_RE = re.compile(
    r"<\s*tool_call\s*>.*?<\s*/\s*tool_call\s*>", re.IGNORECASE | re.DOTALL
)


# 1. Regex-Definitionen für die Extraktion von Namen,
# Geburtstagen, Aufnahmedaten und Entlassungsdaten


# trigger for patient name: look for "Patient: Anna Müller" or
# "Der Patient heißt Anna Müller" etc.
NAME_TRIGGER = re.compile(
    r"\bpatient\b\s*[:\-]?\s*" r"(?:is|ist|named)?\s*" r"(?:herrn|herr|frau)?\s*",
    re.IGNORECASE,
)


# Name-Regex: 1–3 Words, Unicode, starting with capital letter,
#  allowing common name characters
NAME_RE = re.compile(
    r"\b([A-ZÄÖÜA-Za-z][a-zäöüßà-öø-ÿ]{2,}"
    r"(?:\s+[A-ZÄÖÜA-Za-z][a-zäöüßà-öø-ÿ]{2,}){0,2})\b",
    re.UNICODE,
)

# These Titels should be excluded from names)
TITLE_RE = re.compile(r"\b(dr|doctor|prof|professor|mr|mrs|ms|md|phd)\b", re.IGNORECASE)

# clinics / Organisation keywords, to exclude cases like
# "Patientin wurde in der Charité aufgenommen"
ORG_KEYWORDS = re.compile(
    r"\b(hospital|clinic|klinik|medical|center|centre|university|charité|health|care)\b",
    re.IGNORECASE,
)

# eliminate some non-usefull words
NON_NAME_PREFIX_RE = re.compile(
    r"\b(it|the|mentions|looking|okay|this|that|these|those|user|wie|was|wo|wann)\b",
    re.IGNORECASE,
)

# elliminate Words after dr-Titele
DOCTOR_CUTOFF_RE = re.compile(r"\b(dr|doctor|prof|professor|md)\b", re.IGNORECASE)


# 2. Preprocessing: Think-Tags entfernen
def remove_think_tags(text: str) -> str:
    """Remove think-tags from text.

    Parameter:
        text (str): The input text containing think-tags.

    Returns:
        str: The text with think-tags removed.
    """
    think_tag_re = re.compile(r"</?\s*think\s*>", re.IGNORECASE)
    return think_tag_re.sub("", text)


# 3. Validation funktion


def is_valid_person_name(name: str) -> bool:
    """Validate if a candidate string is a valid person name.

    Parameter:
        name (str): The candidate name to validate.

    Returns:
        bool: True if the name is valid, False otherwise.
    """
    return (
        not TITLE_RE.search(name)
        and not ORG_KEYWORDS.search(name)
        and not NON_NAME_PREFIX_RE.match(name)
        and name.lower() != "think"  # Sicherheit gegen Artefakte
    )


def extract_name_from_text(text: str) -> str:
    """Extract patient name from text.

    Parameter:
        text (str): The input text containing the patient name.

    Returns:
        str: The extracted patient name or a default message.
    """
    # eliminate Think-Tags
    cleaned_text = remove_think_tags(text)

    # eliminate words after  Arzt-Titeln
    cutoff = DOCTOR_CUTOFF_RE.search(cleaned_text)
    if cutoff:
        cleaned_text = cleaned_text[: cutoff.start()]

    #  Trigger-Search
    trigger_match = NAME_TRIGGER.search(cleaned_text)
    if trigger_match:
        after_trigger = cleaned_text[trigger_match.end() :]
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
    re.IGNORECASE,
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
    re.IGNORECASE,
)


def extract_birthday_from_text(text: str) -> str:
    """Extract patient birthday from text.

    Parameter:
        text (str): The input text containing the patient birthday.

    Returns:
        str: The extracted patient birthday or a default message.
    """
    # <tool_call>-Block observe
    think_matches = THINK_BLOCK_RE.findall(text)
    combined_text = " ".join(think_matches) if think_matches else text

    #  Trigger-Search
    trigger_match = BIRTHDAY_TRIGGERS.search(combined_text)
    # _LOGGER.info("birthday trigger match:", trigger_match)
    if trigger_match:
        after_trigger = combined_text[trigger_match.end() :]
        before_trigger = combined_text[: trigger_match.start()]
        # _LOGGER.info("after birthday trigger:", after_trigger)
        # _LOGGER.info("before birthday trigger:", before_trigger)

        candidates = BIRTHDAY_RE.findall(after_trigger)
        # _LOGGER.info("candidates after trigger:", candidates)
        if not candidates:
            candidates = BIRTHDAY_RE.findall(before_trigger)
            # _LOGGER.info("candidates before trigger:", candidates)

        if candidates:
            return candidates[0].strip()

    # No Fallback anymore
    return "Nicht angegeben"


RECORDING_TRIGGER = re.compile(
    r"\b(?:aufnahmedatum|aufnahme|admission date|treated|to|wurde|was checked from)"
    r"(?:\s+(?:am|on))?"
    r"[:\s]*",
    re.IGNORECASE,
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
    re.IGNORECASE,
)


def extract_recording_date_from_text(text: str) -> str:
    """Extract patient recording date from text.

    Parameter:
        text (str): The input text containing the patient recording date.

    Returns:
        str: The extracted patient recording date or a default message.
    """
    # <tool_call>-Block berücksichtigen
    think_matches = THINK_BLOCK_RE.findall(text)
    combined_text = " ".join(think_matches) if think_matches else text

    # Nach Triggern suchen
    trigger_match = RECORDING_TRIGGER.search(combined_text)
    # _LOGGER.info("recording date trigger match:", trigger_match)
    if trigger_match:
        after_trigger = combined_text[trigger_match.end() :]
        # _LOGGER.info("after recording date trigger:", after_trigger)

        candidates = RECORDING_DATE_RE.findall(after_trigger)
        # _LOGGER.info("candidates after trigger:", candidates)

        if candidates:
            return candidates[0].strip()

    # Kein Fallback mehr
    return "Nicht angegeben"


RELEASE_DATE_TRIGGER = re.compile(
    r"\b(?:entlassungsdatum|entlassung|entlassen|entließ|release date"
    r"|discharge date|released|discharged|to|bis zum)"
    r"(?:\s+(?:am|on))?"
    r"[:\s]*",
    re.IGNORECASE,
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
    re.IGNORECASE,
)


def extract_release_date_from_text(text: str) -> str:
    """Extract patient release date from text.

    Parameter:
        text (str): The input text containing the patient release date.

    Returns:
        str: The extracted patient release date or a default message.
    """
    # <tool_call>-Block berücksichtigen
    think_matches = THINK_BLOCK_RE.findall(text)
    combined_text = " ".join(think_matches) if think_matches else text

    # Nach Triggern suchen
    trigger_match = RELEASE_DATE_TRIGGER.search(combined_text)
    # _LOGGER.info("release date trigger match:", trigger_match)
    if trigger_match:
        after_trigger = combined_text[trigger_match.end() :]
        # _LOGGER.info("after release date trigger:", after_trigger)

        candidates = RELEASE_DATE_RE.findall(after_trigger)
        # _LOGGER.info("candidates after trigger:", candidates)

        if candidates:
            return candidates[0].strip()

    # No Fallback anymore
    return "Nicht angegeben"


def clean_answer_strict(value: str) -> str:
    """Clean answer string strictly.

    Parameter:
        value (str): The input string to clean.

    Returns:
        str: The cleaned string or a default message.
    """
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
    """Normalize a string.

    Parameter:
        s(str): string to be normalized.

    Returns:
        str
    """
    return " ".join((s or "").lower().strip().split())


def levenshtein_distance(a: str, b: str) -> int:
    """Calculate the Levenshtein distance between two strings.

    Parameter:
        a (str): The first string.
        b (str): The second string.

    Returns:
        int: The Levenshtein distance between the two strings.
    """
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
    """Calculate the Levenshtein similarity ratio between two strings.

    Parameter:
        a (str): The first string.
        b (str): The second string.

    Returns:
        float: The Levenshtein similarity ratio between the two strings.
    """
    a, b = a or "", b or ""
    dist = levenshtein_distance(a, b)
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return (max_len - dist) / max_len


def safe_slice_text(text: Optional[str], start: int, end: int) -> str:
    """Safely slice text between start and end indices, handling edge cases.

    Parameter:
        text (Optional[str]): The text to slice.
        start (int): The starting index.
        end (int): The ending index.

    Returns:
        str: The sliced text or an empty string.
    """
    if not text:
        return ""
    start = max(0, int(start))
    end = min(len(text), int(end))
    if start >= end:
        return ""
    return text[start:end]


def normalize_date(s: Optional[str]) -> Optional[str]:
    """Normalize date string to date object.

    Parameter:
        s (Optional[str]): The date string to normalize.

    Returns:
        Optional[date]: The normalized date object or None.
    """
    if not s:
        return None
    try:
        # dateparser ist flexibel
        dt = dateparser.parse(s, languages=["de"])
        if not dt:
            return None
        return dt.date()
    except Exception:
        return None


DATE_NUMERIC_RE = re.compile(r"\b\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}\b")
DATE_DAYMONTH_RE = re.compile(r"\b\d{1,2}\.\s*[A-Za-zäöüÄÖÜß]+\s*\d{4}\b")


def extract_date(text: Optional[str]) -> Optional[date]:
    """Extract date from text using multiple strategies.

    Parameter:
        text (Optional[str]): The text from which to extract the date.

    Returns:
        Optional[date]: The extracted date or None.
    """
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
    """Clean the ground truth value based on the label.

    Parameter:
        label (str): The label for which to clean the value.
        value (Optional[str]): The ground truth value to clean.

    Returns:
        str: The cleaned ground truth value.
    """
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
    """Remove common titles like "Herr", "Frau", etc.

    Parameter:
        s (Optional[str]): Der Text, aus dem Titel entfernt werden sollen.

    Returns:
        str: Der Text ohne Titel.
    """
    if not s:
        return ""
    return re.sub(r"\b(?:herr|frau|hr\.|fr\.|dr\.?|dr)\b", "", s, flags=re.I).strip()


# robust extraction from GRASCCO annotations


def load_grascco_annotations() -> Dict[str, dict]:
    """Load GRASCCO annotations from JSON file and create a mapping.

    Returns:
        Dict[str, dict]: The created mapping.
    """
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
    """Extract the text for a given label from the annotation entry.

    Parameter:
        annotation_entry (dict): Der Annotationseintrag.
        label (str): Das Label, für das der Text extrahiert werden soll.
        text (str): Der Text, aus dem der Wert extrahiert werden soll.

    Returns:
        str: Der extrahierte Text oder ein leerer String.
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
        # direct slice
        try:
            if (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end <= len(text)
            ):
                candidate = safe_slice_text(text, start, end).strip()
                if candidate:
                    return candidate
        except Exception:
            _LOGGER.exception("Fehler beim direkten Slice der Annotation")

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
                    snippet = " ".join(
                        tokens[max(0, idx - 2) : min(len(tokens), idx + 3)]
                    )
                    return snippet.strip()
                char_pos = tok_end + 1
        return ""
    return ""


# Prediction cleaning


def clean_pred(answer: Optional[str]) -> str:
    """Clean the prediction answer."""
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
            return answer[len(prefix) :].strip(" :.,")

    # if nothing matched, return stripped answer
    return answer


# Evaluation / Matching


def evaluate_prediction(gt: str, pred: str, label: str) -> Tuple[float, bool]:
    """Evaluates the prediction against the ground truth with heuristics.

    Parameter:
        gt (str): The ground truth value.
        pred (str): The prediction value.
        label (str): The label for the evaluation.

    Returns:
        Tuple[float, bool]: A tuple containing the evaluation score
        and a boolean indicating if the prediction is correct.
    """
    gt_raw = clean_gt_value(label, gt or "")
    pred_raw = pred or ""

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
                rest = pred_norm[len(p) :].strip()
                if label == "PatientName":
                    rest_clean = normalize(remove_titles(rest))
                    if rest_clean == normalize(remove_titles(gt_norm)):
                        return 1.0, True
                if label in {
                    "PatientGeburtsdatum",
                    "AufnahmeDatum",
                    "EntlassungsDatum",
                }:
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

        if (
            gt_clean
            and pred_clean
            and (gt_clean in pred_clean or pred_clean in gt_clean)
        ):
            return 1.0, True

        if gt_clean and pred_clean and set(gt_clean.split()) == set(pred_clean.split()):
            return 1.0, True

        score = levenshtein_ratio(gt_clean, pred_clean)
        return score, score >= FUZZY_THRESHOLD

    # 5) Allgemeiner Fallback
    score = levenshtein_ratio(gt_norm, pred_norm)
    return score, score >= FUZZY_THRESHOLD


def make_prompt_german(text: str, question: str, no_think: bool) -> str:
    """Create german prompts.

    Parameter:
        text (str): The input text for the prompt.
        question (str): The question to be answered based on the text.
        no_think (bool): Whether to activate the '/no_think' tag.

    Returns:
        str: The generated prompt string.
    """
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
    """Create english prompts.

    Parameter:
        text (str): The input text for the prompt.
        question (str): The question to be answered based on the text.
        no_think (bool): Whether to activate the '/no_think' tag.

    Returns:
        str: The generated prompt string.
    """
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


# BENCHMARK-Funktions


def run_bmk(max_files: int, use_rag: bool, no_think_option: bool) -> None:
    """Run benchmark for multiple files.

    run benchmark with or without RAG approach for multiple files.
    run benchmark with or without 'no_think' option for multiple files.
    test multiple files against the provided questions .
    truth answers are extracted from the grascco annotations.

    Parameters:
        max_files (int): The maximum number of files to process.
        use_rag (bool): Whether to use the RAG approach.
        no_think_option (bool): Whether to use the 'no_think' option.

    Returns:
        None
    """
    _LOGGER.info("**************Benchmark___Rag*********")
    _LOGGER.info("Model: %s", model_config.name)
    _LOGGER.info("'no_think': %s", no_think_option)
    ann = load_grascco_annotations()
    rows = []
    total = 0
    correct = 0

    files = islice(sorted(RAW_TEXT_DIR.glob("*.txt")), max_files)

    for file_path in files:
        fname = file_path.name.replace("ö", "o")

        if fname not in ann:
            _LOGGER.warning("No annotation for %s", fname)
            continue

        annotation_entry = ann[fname]
        text = file_path.read_text(encoding="utf-8")

        # ---
        # Reset + Upload Vectorstore
        # ---
        if use_rag:
            try:
                TRANSPORTER.clear_vectorstore()
            except Exception:
                _LOGGER.exception("Failed to reset vectorstore")
                continue
            upload = QAFileUpload(data=text.encode("utf-8"), name=fname)
            res = TRANSPORTER.add_file(upload)
            if getattr(res, "status", None) != 200:
                _LOGGER.error("Upload failed for %s", fname)
                continue

        if use_rag:
            for question in QUESTIONS:
                total += 1
                label = LABEL_MAPPING[question]
                truth_answer = extract_label_text(annotation_entry, label, text)
                prompt = make_prompt_englich(
                    text=text, question=question, no_think=no_think_option
                )
                # _LOGGER.info(prompt)
                q = QAQuestion(
                    question=prompt,
                    search_strategy="similarity",
                    max_sources=3,
                    no_think=no_think_option,
                )

                qa_res = TRANSPORTER.qa_query(q)
                pred_answer = qa_res.response
        else:
            for question in QUESTIONS:
                total += 1
                label = LABEL_MAPPING[question]
                truth_answer = extract_label_text(annotation_entry, label, text)
                prompt = make_prompt_englich(
                    text=text, question=question, no_think=no_think_option
                )
                try:
                    qa_res = llm.create_chat_completion(
                        messages=[{"role": "user", "content": prompt}]
                    )
                except Exception as e:
                    _LOGGER.warning(
                        "WARNUNG: LLM-Aufruf für %s ist fehlgeschlagen: %s", fname, e
                    )
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

            # pred = clean_pred(pred_answer)

            score, match = evaluate_prediction(truth_answer, pred_answer, label)

            if match:
                correct += 1

            _LOGGER.info(
                "%s | %s | Q: %s | GT: %s | " "PRED: %s | score=%.2f | match=%s",
                (fname, label, question, truth_answer, pred_answer, score, match),
            )

            rows.append(
                {
                    "file": fname,
                    "label": label,
                    "question_used": question,
                    "ground_truth": truth_answer,
                    "predicted": pred_answer,
                    "match_score": f"{score:.3f}",
                    "match": match,
                }
            )

    # ---
    # Final metrics
    # ---
    acc = correct / total if total else 0.0
    _LOGGER.info("================================================")
    _LOGGER.info("FINAL SCORE: %d / %d accuracy=%.3f", correct, total, acc)
    _LOGGER.info("================================================")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        with RESULT_CSV_RAG.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        _LOGGER.info("Results saved in: %s", RESULT_CSV_RAG)


def run_bmk_onefile(file_path: Path, use_rag: bool, no_think_option: bool) -> None:
    """Run benchmark for one file.

    run benchmark with or without RAG approach for one file.
    run benchmark with or without 'no_think' option for one file.
    test one file against the provided questions .
    truth answers are extracted from the grascco annotations.

    Parameters:
        file_path (Path): The path to the file to process.
        use_rag (bool): Whether to use the RAG approach.
        no_think_option (bool): Whether to use the 'no_think' option.

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
        _LOGGER.warning("Keine Annotation für %s", fname)
        return
    annotation_entry = ann[fname]
    text = file_path.read_text(encoding="utf-8")
    _LOGGER.info("Datei: %s", fname)
    _LOGGER.info("Model: %s", model_config.name)
    _LOGGER.info("'no_think' ist auf %s gesetzt.", no_think_option)
    if use_rag:
        try:
            TRANSPORTER.remove_file(fname)
        except Exception:
            _LOGGER.debug(
                "Fehler beim Entfernen der Datei %s (vielleicht existiert sie nicht).",
                fname,
            )

        upload = QAFileUpload(data=text.encode("utf-8"), name=fname)
        res = TRANSPORTER.add_file(upload)
        if getattr(res, "status", None) != 200:
            _LOGGER.error("Fehler beim Laden: %s", getattr(res, "error_msg", res))

    if use_rag:
        for question in QUESTIONS:
            total += 1
            label = LABEL_MAPPING[question]
            truth_answer = extract_label_text(annotation_entry, label, text)
            prompt = make_prompt_englich(text, question, no_think_option)
            _LOGGER.info("prompt: %s", prompt)
            qa_question = QAQuestion(
                question=prompt,
                search_strategy="similarity",
                max_sources=3,
                no_think=no_think_option,
            )
            qa_answer = TRANSPORTER.qa_query(qa_question)
            pred_answer = qa_answer.response
            # _LOGGER.info("RAG-Antwort: %s", pred_answer)

    else:
        for question in QUESTIONS:
            total += 1
            label = LABEL_MAPPING[question]
            truth_answer = extract_label_text(annotation_entry, label, text)
            prompt = make_prompt_englich(text, question, no_think_option)
        try:
            qa_res = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
        except Exception as e:
            _LOGGER.warning(
                "WARNUNG: LLM-Aufruf für %s ist fehlgeschlagen: %s", fname, e
            )
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

        # pred = clean_pred(pred)

        score, match = evaluate_prediction(truth_answer, pred, label)
        if match:
            correct += 1

        _LOGGER.info(
            "%s | %s | Q: %s | GT: %s | PRED: %s | score=%.2f | match=%s",
            fname,
            question,
            truth_answer,
            pred,
            score,
            match,
        )

        rows.append(
            {
                "file": fname,
                "question": question,
                "label": label,
                "ground_truth": truth_answer,
                "predicted": pred,
                "match_score": f"{score:.3f}",
                "match": match,
            }
        )

    acc = correct / total if total else 0
    _LOGGER.info("================================================")
    _LOGGER.info("FINAL SCORE: %d / %d  accuracy=%.3f", correct, total, acc)
    _LOGGER.info("================================================")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        with RESULT_CSV_WITHOUT_RAG_ONE_FILE.open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        _LOGGER.info("Ergebnisse gespeichert in: %s", RESULT_CSV_WITHOUT_RAG_ONE_FILE)


def profile(file_path: Path, use_rag: bool, no_think: bool) -> None:
    """Profile a single file and extract key information.

    This function processes a single file to extract key information
    such as patient name, birth date, recording date, and release date.
    It uses either a RAG approach or direct LLM querying
    based on the parameters provided.
    The extracted information is then formatted into a profile
    and saved to a text file.

    Parameter:
        file_path (Path): The path to the file to be profiled.
        use_rag (bool): Whether to use the RAG approach.
        no_think (bool): Whether to use the 'no_think' option.

    Returns:
        None
    """
    _LOGGER.info("********Profile*********")
    text = file_path.read_text(encoding="utf-8")
    _LOGGER.info("Datei: %s", file_path.name)
    _LOGGER.info("Model: %s", model_config.name)
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
            _LOGGER.exception("Fehler beim Zurücksetzen des Vectorstores")

        upload = QAFileUpload(data=text.encode("utf-8"), name=file_path.name)
        res = TRANSPORTER.add_file(upload)
        if getattr(res, "status", None) != 200:
            msg = "Upload in Vectorstore fehlgeschlagen"
            raise RuntimeError(msg)

    # =========================
    # FRAGEN-SCHLEIFE
    # =========================
    for question in QUESTIONS:
        if use_rag:
            prompt = make_prompt_englich(text, question, no_think)
            q = QAQuestion(
                question=prompt,
                search_strategy="similarity",
                max_sources=3,
                no_think=no_think,
            )

            res = TRANSPORTER.qa_query(q)
            raw_answer = res.response
            answers[question] = raw_answer

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

        # answers[question] = raw_answer
        _LOGGER.info("Frage: %s | Antwort: %r", question, raw_answer)

    # =========================
    # SORTIERTE TEXTAUSGABE
    # =========================
    _LOGGER.info(
        "Wie heißt der Patient?:",
        answers.get("Wie heißt der Patient?", "Nicht angegeben"),
    )
    _LOGGER.info(answers.get("Wann hat der Patient Geburtstag?", "Nicht angegeben"))
    _LOGGER.info(
        answers.get("Wann wurde der Patient bei uns aufgenommen?", "Nicht angegeben")
    )
    _LOGGER.info(
        answers.get("Wann wurde der Patient bei uns entlassen?", "Nicht angegeben")
    )

    lines = [
        "Kurzprofil Entlassungsbrief\n",
        f"Patient: {extract_name_from_text(
            answers.get('Wie heißt der Patient?'
                        , 'Nicht angegeben'))}",
        f"Geburtsdatum: {extract_birthday_from_text(
            answers.get('Wann hat der Patient Geburtstag?'
                        , 'Nicht angegeben'))}",
        f"Aufnahme: {extract_recording_date_from_text(
            answers.get('Wann wurde der Patient bei uns aufgenommen?'
                        , 'Nicht angegeben'))}",
        f"Entlassung: {extract_release_date_from_text(
            answers.get('Wann wurde der Patient bei uns entlassen?'
                        , 'Nicht angegeben'))}",
    ]

    summary = "\n".join(lines).strip()

    # =========================
    # SPEICHERN
    # =========================
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{file_path.stem}_profile.txt"
    out_path.write_text(summary, encoding="utf-8")
    _LOGGER.info("Profil gespeichert unter: %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
    )
    # run_bmk(max_files = 5, use_rag=True, no_think_option= True)
    # run_bmk_one_file(file_path=RAW_TEXT_DIR / "Cajal.txt"
    # , use_rag=True, no_think_option=False)
    profile(file_path=RAW_TEXT_DIR / "Cajal.txt", use_rag=True, no_think=True)
