from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

class OriginType(str, Enum):
    native_digital = "native_digital"
    scanned_image = "scanned_image"
    mixed = "mixed"
    form_fillable = "form_fillable"

class LayoutComplexity(str, Enum):
    single_column = "single_column"
    multi_column = "multi_column"
    table_heavy = "table_heavy"
    figure_heavy = "figure_heavy"
    mixed = "mixed"

class DomainHint(str, Enum):
    financial = "financial"
    legal = "legal"
    technical = "technical"
    medical = "medical"
    general = "general"

class ExtractionStrategy(str, Enum):
    fast_text = "fast_text"
    layout_aware = "layout_aware"
    vision_augmented = "vision_augmented"

class DocumentProfile(BaseModel):
    doc_id: str
    origin_type: OriginType
    layout_complexity: LayoutComplexity
    language: str = "en"
    language_confidence: float = 1.0
    domain_hint: DomainHint = DomainHint.general
    estimated_cost_tier: ExtractionStrategy
    metadata: dict = Field(default_factory=dict)