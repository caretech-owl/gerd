import json
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, TypeAdapter, computed_field


class LabelStudioLabel(Enum):
    Abteilung = "Abteilung"
    Anrede = "Anrede"
    AufnahmeDatum = "AufnahmeDatum"
    BehandelnderArzt = "BehandelnderArzt"
    Einrichtung = "Einrichtung"
    EntlassDatum = "EntlassDatum"
    Hausarzt = "Hausarzt"
    PatientGeburtsdatum = "PatientGeburtsdatum"
    PatientName = "PatientName"


class LabelStudioAnnotationValue(BaseModel):
    end: int
    labels: List[LabelStudioLabel]
    start: int

    model_config = ConfigDict(extra="forbid")


class LabelStudioAnnotationResult(BaseModel):
    from_name: str
    id: str
    origin: str
    to_name: str
    type: str
    value: LabelStudioAnnotationValue

    model_config = ConfigDict(extra="forbid")


class LabelStudioAnnotation(BaseModel):
    completed_by: int
    created_at: str
    draft_created_at: Optional[str]
    ground_truth: bool
    id: int
    import_id: Optional[str]
    last_action: Optional[str]
    last_created_by: Optional[int]
    lead_time: float
    parent_annotation: Optional[str]
    parent_prediction: Optional[str]
    prediction: Dict[str, str]
    project: int
    result_count: int
    result: List[LabelStudioAnnotationResult]
    task: int
    unique_id: str
    updated_at: str
    updated_by: int
    was_cancelled: bool

    model_config = ConfigDict(extra="forbid")


class LabelStudioTask(BaseModel):
    annotations: List[LabelStudioAnnotation]
    cancelled_annotations: int
    comment_authors: List[str]
    comment_count: int
    created_at: str
    data: Optional[Dict[str, str]]
    drafts: List[str]
    file_upload: str
    id: int
    inner_id: int
    last_comment_updated_at: Optional[str]
    meta: Optional[Dict[str, str]]
    predictions: List[str]
    project: int
    total_annotations: int
    total_predictions: int
    unresolved_comment_count: int
    updated_at: str
    updated_by: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def file_name(self) -> str:
        return self.file_upload.split("-", 1)[-1]

    model_config = ConfigDict(extra="forbid")


def load_label_studio_tasks(file_path: str) -> List[LabelStudioTask]:
    with open(file_path, "r") as f:
        obj = json.load(f)

    tasks = TypeAdapter(List[LabelStudioTask]).validate_python(obj)
    return tasks
