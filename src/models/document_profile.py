"""
Pydantic models for document profiling and classification.
This module defines the DocumentProfile schema used throughout the pipeline.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class OriginType(str, Enum):
    """Document origin classification"""
    NATIVE_DIGITAL = "native_digital"
    SCANNED_IMAGE = "scanned_image"
    MIXED = "mixed"
    FORM_FILLABLE = "form_fillable"


class LayoutComplexity(str, Enum):
    """Document layout complexity classification"""
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TABLE_HEAVY = "table_heavy"
    FIGURE_HEAVY = "figure_heavy"
    MIXED = "mixed"


class DomainHint(str, Enum):
    """Document domain classification"""
    FINANCIAL = "financial"
    LEGAL = "legal"
    TECHNICAL = "technical"
    MEDICAL = "medical"
    GENERAL = "general"


class ExtractionStrategy(str, Enum):
    """Extraction strategy selection"""
    FAST_TEXT = "fast_text"
    LAYOUT_AWARE = "layout_aware"
    VISION_AUGMENTED = "vision_augmented"


class DocumentProfile(BaseModel):
    """
    Comprehensive document profile that governs extraction strategy selection.
    """
    doc_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
    page_count: int = Field(..., ge=1, description="Total number of pages")
    
    # Classification dimensions
    origin_type: OriginType = Field(..., description="Document origin classification")
    layout_complexity: LayoutComplexity = Field(..., description="Layout complexity assessment")
    language: Dict[str, float] = Field(
        default_factory=lambda: {"en": 1.0},
        description="Detected language codes with confidence scores"
    )
    domain_hint: DomainHint = Field(
        default=DomainHint.GENERAL,
        description="Document domain for prompt selection"
    )
    
    # Extraction strategy recommendation
    recommended_strategy: ExtractionStrategy = Field(
        ..., description="Recommended extraction strategy"
    )
    confidence_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "character_density": 0.7,
            "table_completeness": 0.8,
            "layout_preservation": 0.75
        },
        description="Confidence thresholds for extraction quality"
    )
    
    # Metadata for strategy selection
    character_density_stats: Dict[str, float] = Field(
        default_factory=dict,
        description="Character density statistics across pages"
    )
    image_area_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Ratio of image area to total page area"
    )
    has_embedded_fonts: bool = Field(
        default=False,
        description="Whether document has embedded fonts (indicates digital origin)"
    )
    table_count_estimate: int = Field(
        default=0, ge=0,
        description="Estimated number of tables in document"
    )
    figure_count_estimate: int = Field(
        default=0, ge=0,
        description="Estimated number of figures in document"
    )
    
    # Processing metadata
    extraction_cost_estimate: float = Field(
        default=0.0,
        description="Estimated cost in USD for extraction"
    )
    processing_time_estimate: float = Field(
        default=0.0,
        description="Estimated processing time in seconds"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('character_density_stats')
    def validate_character_density(cls, v):
        """Ensure character density stats have required fields"""
        required = ['mean', 'min', 'max', 'std']
        if v and not all(key in v for key in required):
            raise ValueError(f"Character density stats must contain: {required}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": "cbe_annual_report_2023",
                "filename": "CBE_ANNUAL_REPORT_2023-24.pdf",
                "file_size_bytes": 5242880,
                "page_count": 120,
                "origin_type": "native_digital",
                "layout_complexity": "multi_column",
                "language": {"en": 0.95, "am": 0.05},
                "domain_hint": "financial",
                "recommended_strategy": "layout_aware",
                "character_density_stats": {
                    "mean": 2500.5,
                    "min": 1800.0,
                    "max": 3200.0,
                    "std": 450.2
                },
                "image_area_ratio": 0.15,
                "has_embedded_fonts": True,
                "table_count_estimate": 45,
                "figure_count_estimate": 12
            }
        }