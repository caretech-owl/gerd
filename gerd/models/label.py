"""Data definitions for [Label Studio](https://labelstud.io/) tasks.

The defined models and enums are used to parse and work with
Label Studio data exported as JSON.
"""

import json
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, TypeAdapter, computed_field


class LabelStudioLabel(Enum):
    """Labels for the GRASCCO Label Studio annotations."""

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
    """Value of a Label Studio annotation."""

    end: int
    """The end of the annotation."""
    labels: List[LabelStudioLabel]
    """The labels of the annotation."""
    start: int
    """The start of the annotation."""

    model_config = ConfigDict(extra="forbid")


class LabelStudioAnnotationResult(BaseModel):
    """Result of a Label Studio annotation."""

    from_name: str
    """The name of the source."""
    id: str
    """The ID of the result."""
    origin: str
    """The origin of the result."""
    to_name: str
    """The name of the target."""
    type: str
    """The type of the result."""
    value: LabelStudioAnnotationValue
    """The value of the result."""

    model_config = ConfigDict(extra="forbid")


class LabelStudioAnnotation(BaseModel):
    """Annotation of a Label Studio task.

    A collection of annotations is associated with a task.
    """

    completed_by: int
    """The user ID of the user who completed the annotation."""
    created_at: str
    """The creation date of the annotation."""
    draft_created_at: Optional[str]
    """The creation date of the draft."""
    ground_truth: bool
    """Whether the annotation is ground truth."""
    id: int
    """The ID of the annotation."""
    import_id: Optional[str]
    """The import ID of the annotation."""
    last_action: Optional[str]
    """The last action of the annotation."""
    last_created_by: Optional[int]
    """The user ID of the user who last created the annotation."""
    lead_time: float
    """The lead time of the annotation."""
    parent_annotation: Optional[str]
    """The parent annotation."""
    parent_prediction: Optional[str]
    """The parent prediction."""
    prediction: Dict[str, str]
    """The prediction of the annotation."""
    project: int
    """The project ID of the annotation."""
    result_count: int
    """The number of results."""
    result: List[LabelStudioAnnotationResult]
    """The results of the annotation."""
    task: int
    """The task ID of the annotation."""
    unique_id: str
    """The unique ID of the annotation."""
    updated_at: str
    """The update date of the annotation."""
    updated_by: int
    """The user ID of the user who updated the annotation."""
    was_cancelled: bool
    """Whether the annotation was cancelled."""

    model_config = ConfigDict(extra="forbid")


class LabelStudioTask(BaseModel):
    """Task of a Label Studio project.

    A task is a single unit of work that can be annotated by a user.
    Tasks can be used to train an auto labeler or to evaluate
    the performance of a model.
    """

    annotations: List[LabelStudioAnnotation]
    """The annotations of the task."""
    cancelled_annotations: int
    """The number of cancelled annotations."""
    comment_authors: List[str]
    """The authors of the comments."""
    comment_count: int
    """The number of comments."""
    created_at: str
    """The creation date of the task."""
    data: Optional[Dict[str, str]]
    """The data of the task."""
    drafts: List[str]
    """The drafts of the task."""
    file_upload: str
    """The file upload of the task."""
    id: int
    """The ID of the task."""
    inner_id: int
    """The inner ID of the task."""
    last_comment_updated_at: Optional[str]
    """The update date of the last comment."""
    meta: Optional[Dict[str, str]]
    """The meta data of the task."""
    predictions: List[str]
    """The predictions of the task."""
    project: int
    """The project ID of the task."""
    total_annotations: int
    """The total number of annotations."""
    total_predictions: int
    """The total number of predictions."""
    unresolved_comment_count: int
    """The number of unresolved comments."""
    updated_at: str
    """The update date of the task."""
    updated_by: int
    """The user ID of the user who updated the task."""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def file_name(self) -> str:
        """Extracts the original file name from the file upload.

        File uploads are stored as `project-id-filename` format to be unique.
        """
        return self.file_upload.split("-", 1)[-1]

    model_config = ConfigDict(extra="forbid")


def load_label_studio_tasks(file_path: str) -> List[LabelStudioTask]:
    """Load Label Studio tasks from a JSON file.

    Parameters:
        file_path: The path to the JSON file.

    Returns:
        The loaded Label Studio tasks
    """
    with open(file_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    tasks = TypeAdapter(List[LabelStudioTask]).validate_python(obj)
    return tasks
