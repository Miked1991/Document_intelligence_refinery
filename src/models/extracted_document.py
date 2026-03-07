"""
Models for normalized document representation after extraction.
All extraction strategies must output this schema.
FIXED: Made bbox optional in ContentBlock to handle vision extraction.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime


class BlockType(str, Enum):
    """Types of content blocks in a document"""
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    HEADER = "header"
    FOOTER = "footer"
    CAPTION = "caption"
    LIST = "list"
    EQUATION = "equation"


class BoundingBox(BaseModel):
    """
    Spatial coordinates for provenance tracking.
    Uses pdfplumber's coordinate system (points from top-left).
    """
    x0: float = Field(..., description="Left coordinate")
    y0: float = Field(..., description="Top coordinate")
    x1: float = Field(..., description="Right coordinate")
    y1: float = Field(..., description="Bottom coordinate")
    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization"""
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "page": self.page_number
        }


class TableCell(BaseModel):
    """Represents a single cell in a table"""
    content: str = Field(..., description="Cell content")
    row_index: int = Field(..., ge=0, description="Row index (0-based)")
    col_index: int = Field(..., ge=0, description="Column index (0-based)")
    row_span: int = Field(1, ge=1, description="Number of rows spanned")
    col_span: int = Field(1, ge=1, description="Number of columns spanned")
    is_header: bool = Field(False, description="Whether this is a header cell")
    bbox: Optional[BoundingBox] = Field(None, description="Cell bounding box")


class Table(BaseModel):
    """Structured table representation"""
    headers: List[str] = Field(..., description="Column headers")
    rows: List[List[str]] = Field(..., description="Table rows as lists of strings")
    caption: Optional[str] = Field(None, description="Table caption")
    bbox: BoundingBox = Field(..., description="Table bounding box")
    cells: Optional[List[TableCell]] = Field(None, description="Detailed cell information")
    
    def to_markdown(self) -> str:
        """Convert table to markdown format"""
        if not self.rows:
            return ""
        
        # Create header row
        header_row = "| " + " | ".join(self.headers) + " |"
        separator = "|" + "|".join([" --- " for _ in self.headers]) + "|"
        
        # Create data rows
        data_rows = []
        for row in self.rows:
            # Ensure row has same number of columns as headers
            padded_row = row + [""] * (len(self.headers) - len(row))
            data_rows.append("| " + " | ".join(padded_row) + " |")
        
        return "\n".join([header_row, separator] + data_rows)


class Figure(BaseModel):
    """Represents a figure/image in the document"""
    caption: Optional[str] = Field(None, description="Figure caption")
    bbox: BoundingBox = Field(..., description="Figure bounding box")
    description: Optional[str] = Field(None, description="Description (for accessibility)")
    image_path: Optional[str] = Field(None, description="Path to extracted image")


class TextBlock(BaseModel):
    """Represents a contiguous text block"""
    content: str = Field(..., description="Text content")
    bbox: BoundingBox = Field(..., description="Block bounding box")
    block_type: BlockType = Field(BlockType.TEXT, description="Type of text block")
    font_info: Optional[Dict[str, Any]] = Field(None, description="Font information")


class ContentBlock(BaseModel):
    """
    Unified content block that can be any type.
    FIXED: Made bbox optional with default None to handle vision extraction
    where spatial coordinates aren't available.
    """
    block_id: str = Field(..., description="Unique block identifier")
    block_type: BlockType = Field(..., description="Type of block")
    content: Union[str, Table, Figure, Dict] = Field(..., description="Block content")
    bbox: Optional[BoundingBox] = Field(None, description="Block bounding box (optional for vision extraction)")
    parent_id: Optional[str] = Field(None, description="Parent block ID (for hierarchy)")
    children_ids: List[str] = Field(default_factory=list, description="Child block IDs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('block_id')
    def validate_block_id(cls, v):
        """Ensure block_id follows expected format"""
        if not v or len(v) < 5:
            raise ValueError("block_id must be at least 5 characters")
        return v


class ExtractedDocument(BaseModel):
    """
    Normalized document representation after extraction.
    All extraction strategies must output this schema.
    """
    doc_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    page_count: int = Field(..., ge=1, description="Number of pages")
    
    # Content blocks in reading order
    blocks: List[ContentBlock] = Field(..., description="Content blocks in reading order")
    
    # Extracted tables (for convenience)
    tables: List[Table] = Field(default_factory=list, description="Extracted tables")
    
    # Extracted figures
    figures: List[Figure] = Field(default_factory=list, description="Extracted figures")
    
    # Metadata
    extraction_strategy: str = Field(..., description="Strategy used for extraction")
    extraction_timestamp: datetime = Field(default_factory=datetime.now)
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    extraction_time_seconds: float = Field(..., ge=0, description="Extraction time")
    cost_estimate_usd: float = Field(0.0, ge=0, description="Estimated cost in USD")
    
    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": "cbe_annual_report_2023",
                "filename": "CBE_ANNUAL_REPORT_2023-24.pdf",
                "page_count": 120,
                "extraction_strategy": "layout_aware",
                "confidence_score": 0.95,
                "extraction_time_seconds": 45.2,
                "tables": [
                    {
                        "headers": ["Year", "Revenue", "Expenses"],
                        "rows": [["2023", "$4.2B", "$3.1B"]],
                        "caption": "Income Statement Summary"
                    }
                ]
            }
        }