from __future__ import annotations

from pydantic import BaseModel, Field


class AnnotationPayload(BaseModel):
    sequence_id: str
    plate_id: str
    well: str
    timepoint: str
    object_id: str
    canonical_target_id: str
    x_px: float
    y_px: float
    object_type: str
    viability: str = "unknown"
    division_state: str = "unknown"
    duplicate_of: str | None = None
    reviewer: str = "local_user"
    confidence: float | None = None
    notes: str = ""


class PointSelection(BaseModel):
    x_px: float | None = None
    y_px: float | None = None
    candidate_id: str | None = None
    area_px: float | None = None
    object_label: str = "uncertain"
    marker_diameter_px: float | None = None
    orientation_rad: float | None = None


class TimepointSelection(PointSelection):
    present: bool = True
    additional_points: list[PointSelection] = Field(default_factory=list)


class LinkReviewPayload(BaseModel):
    link_id: str
    parent_timepoint: str
    child_timepoint: str
    link_label: str = "uncertain"


class LineageReviewPayload(BaseModel):
    sequence_id: str
    plate_id: str
    well: str
    canonical_target_id: str
    object_type: str
    viability: str
    division_state: str = "unknown"
    morphology: str = "uncertain"
    points: dict[str, TimepointSelection]
    lineage_status: str = "needs_review"
    review_confidence: str = "medium"
    issue_tags: list[str] = Field(default_factory=list)
    links: list[LinkReviewPayload] = Field(default_factory=list)
    reviewer: str = "local_user"
    notes: str = ""


class TeachingLabelItem(BaseModel):
    candidate_id: str
    well: str
    timepoint: str = "T0"
    x_px: float
    y_px: float
    label: str
    source: str = "quick_teaching"


class TeachingLabelsPayload(BaseModel):
    items: list[TeachingLabelItem]
    reviewer: str = "local_user"


class MultiplicityLabelItem(BaseModel):
    candidate_id: str
    well: str
    timepoint: str
    x_px: float
    y_px: float
    label: str
    source: str = "quick_multiplicity"


class MultiplicityLabelsPayload(BaseModel):
    items: list[MultiplicityLabelItem]
    reviewer: str = "local_user"


class AutoReviewItem(BaseModel):
    candidate_id: str
    predicted_label: str
    reviewed_label: str


class AutoReviewsPayload(BaseModel):
    round_id: str
    items: list[AutoReviewItem]
    reviewer: str = "local_user"


class IntegratedReviewItem(BaseModel):
    candidate_id: str
    predicted_label: str
    reviewed_label: str


class IntegratedReviewsPayload(BaseModel):
    round_id: str
    items: list[IntegratedReviewItem]
    reviewer: str = "local_user"


class QuickReviewObjectItem(BaseModel):
    candidate_id: str
    predicted_label: str
    reviewed_label: str
    well: str
    timepoint: str
    x_px: float
    y_px: float
    diameter_px: float = 12.0
    is_new: bool = False


class V3TrackReviewItem(BaseModel):
    track_id: str
    well: str
    label: str
    behavior: str = ""


class QuickReviewWellPayload(BaseModel):
    round_id: str
    items: list[QuickReviewObjectItem]
    well: str | None = None
    reviewer: str = "local_user"
    duration_ms: int | None = None
    v3_track_reviews: list[V3TrackReviewItem] = Field(default_factory=list)


class QuickReviewUndoPayload(BaseModel):
    action_id: int | None = None
    reviewer: str = "local_user"


class WellScreeningReviewPayload(BaseModel):
    well: str
    decision: str
    reviewer: str = "local_user"
    notes: str = ""


class LateGrowthReviewPayload(BaseModel):
    well: str
    timepoint: str
    decision: str
    reviewer: str = "local_user"
    notes: str = ""


class MaskReviewSavePayload(BaseModel):
    round_id: str
    candidate_id: str
    decision: str
    reviewed_mask_rle: str | None = None
    reviewer: str = "local_user"
    notes: str = ""


class MaskComparisonPayload(BaseModel):
    old_checkpoint: str | None = None
    new_checkpoint: str | None = None
    source_configs: list[str] = Field(default_factory=list)
    round_id: str | None = None

