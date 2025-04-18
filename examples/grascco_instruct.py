"""GRASCCO dataset preparation for instruction training.

This example demonstrates how to generate training data for the Instruct model using
the Grascco dataset.
"""
# %%
# import packages

import json
from pathlib import Path

from gerd.models.model import ChatMessage
from gerd.training.instruct import InstructTrainingData, InstructTrainingSample

project_dir = Path(__file__).parent.parent

# %%
# Load annotation


with (project_dir / "tests/data/grascco/grascco.json").open() as f:
    annotation = {j["file_upload"].split("-")[1]: j for j in json.load(f)}

# %%
# Generate training data
data = InstructTrainingData()

system_prompt = ChatMessage(
    role="system",
    content=(
        "Du hilfst Ärzten und Ärztinnen "
        "bei der Erstellung von Arztbriefen. Nutze dazu folgenden Informationen, "
        "um eine Einleitung zu schreiben: "
        "Das Aufnahmedatum (AufnahmeDatum), "
        "das Entlassdatum "
        "(EntlassDatum), das Geburtsdatum des Patienten (PatientGeburtsdatum), "
        "und den Namen des Patienten (PatientName)."
    ),
)

for file in (project_dir / "tests/data/grascco/raw").glob("*.txt"):
    fname = file.name.replace("ö", "o")
    if fname not in annotation:
        msg = f"Annotation for {fname} not found"
        raise RuntimeError(msg)
    txt = file.read_text()
    anno = annotation[fname]["annotations"][0]
    labels = {
        r["value"]["labels"][0]: [r["value"]["start"], r["value"]["end"]]
        for r in anno["result"]
    }
    if "Anrede" not in labels:
        continue
    s, e = labels["Anrede"]
    data.samples.append(
        InstructTrainingSample(
            messages=[
                system_prompt,
                ChatMessage(
                    role="user",
                    content="\n".join(
                        f"{k} = {txt[a:b].strip()}"
                        for k, (a, b) in sorted(labels.items())
                        if k
                        in [
                            "PatientName",
                            "PatientGeburtsdatum",
                            "AufnahmeDatum",
                            "EntlassDatum",
                            # "BehandelnderArzt",
                            # "Abteilung",
                            # "Einrichtung",
                        ]
                    ),
                ),
                ChatMessage(role="assistant", content=f"{txt[s:e].strip()}"),
            ]
        )
    )

# %%
# Save training data
with (project_dir / "tmp/grascco_instruct.json").open("w") as f:
    f.write(data.model_dump_json(indent=2))

# %%
