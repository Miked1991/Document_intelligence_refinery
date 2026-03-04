"""Enhanced extracted document model with comprehensive validation."""

from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import hashlib
from datetime import datetime


class TextBlock(BaseModel):
    """Text block with full provenance."""
    text: str = Field(..., min_length=1)
    bbox: Tuple[float, float, float, float] = Field(..., description="x1,y1,x2,y2")
    font: Optional[str] = None
    size: Optional[float] = None
    confidence: float = Field(0.9, ge=0.0, le=1.0)
    language: str = "en"
    content_hash: Optional[str] = None
    
    @validator('bbox')
    def validate_bbox(cls, v):
        x1, y1, x2, y2 = v
        if x1 >= x2:
            raise ValueError(f"x1 ({x1}) must be less than x2 ({x2})")
        if y1 >= y2:
            raise ValueError(f"y1 ({y1}) must be less than y2 ({y2})")
        return v
    
    @validator('content_hash', always=True)
    def generate_hash(cls, v, values):
        if v is None and 'text' in values:
            return hashlib.sha256(values['text'].encode()).hexdigest()
        return v


class Table(BaseModel):
    """Structured table with headers and rows."""
    headers: List[str] = Field(..., min_items=1)
    rows: List[List[str]] = Field(..., min_items=1)
    bbox: Tuple[float, float, float, float]
    caption: Optional[str] = None
    confidence: float = Field(0.9, ge=0.0, le=1.0)
    
    @validator('rows')
    def rows_match_headers(cls, v, values):
        if 'headers' in values:
            expected_cols = len(values['headers'])
            for i, row in enumerate(v):
                if len(row) != expected_cols:
                    raise ValueError(
                        f"Row {i} has {len(row)} columns, expected {expected_cols}"
                    )
        return v
    
    def to_markdown(self) -> str:
        """Convert to markdown table format."""
        if not self.headers or not self.rows:
            return ""
        
        # Create header row
        lines = ["| " + " | ".join(self.headers) + " |"]
        
        # Create separator row
        lines.append("|" + "|".join([" --- " for _ in self.headers]) + "|")
        
        # Create data rows
        for row in self.rows:
            lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with structure preserved."""
        return {
            "headers": self.headers,
            "rows": self.rows,
            "caption": self.caption,
            "bbox": self.bbox
        }


class Figure(BaseModel):
    """Figure with caption and metadata."""
    caption: Optional[str] = None
    bbox: Tuple[float, float, float, float]
    image_data: Optional[bytes] = None
    image_hash: Optional[str] = None
    figure_type: str = "unknown"  # chart, diagram, photo, drawing
    confidence: float = Field(0.9, ge=0.0, le=1.0)
    
    @validator('image_hash', always=True)
    def generate_image_hash(cls, v, values):
        if v is None and 'image_data' in values and values['image_data']:
            return hashlib.sha256(values['image_data']).hexdigest()
        return v


class Page(BaseModel):
    """Single page with all content types."""
    page_num: int = Field(..., ge=1)
    text_blocks: List[TextBlock] = Field(default_factory=list)
    tables: List[Table] = Field(default_factory=list)
    figures: List[Figure] = Field(default_factory=list)
    reading_order: List[int] = Field(default_factory=list)
    width: Optional[float] = None
    height: Optional[float] = None
    
    @validator('reading_order')
    def validate_reading_order(cls, v, values):
        """Ensure reading order indices are valid."""
        if v:
            max_block_idx = len(values.get('text_blocks', [])) - 1
            max_table_idx = len(values.get('tables', [])) - 1 + max_block_idx + 1
            max_figure_idx = len(values.get('figures', [])) - 1 + max_table_idx + 1
            
            for idx in v:
                if idx < 0 or idx > max_figure_idx:
                    raise ValueError(f"Reading order index {idx} out of range")
        return v
    
    def get_all_content(self) -> List[Dict[str, Any]]:
        """Get all content in reading order with type information."""
        content = []
        
        # Create combined list
        all_items = []
        all_items.extend([("text", i, tb) for i, tb in enumerate(self.text_blocks)])
        all_items.extend([("table", i, t) for i, t in enumerate(self.tables)])
        all_items.extend([("figure", i, f) for i, f in enumerate(self.figures)])
        
        # Sort by reading order
        if self.reading_order:
            order_map = {idx: pos for pos, idx in enumerate(self.reading_order)}
            all_items.sort(key=lambda x: order_map.get(x[1], 999))
        
        for item_type, idx, item in all_items:
            content.append({
                "type": item_type,
                "index": idx,
                "content": item.dict(),
                "text": item.text if hasattr(item, 'text') else None
            })
        
        return content


class ExtractedDocument(BaseModel):
    """Complete extracted document with all pages."""
    document_id: Optional[str] = None
    pages: List[Page] = Field(..., min_items=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    extraction_strategy: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    
    @root_validator
    def validate_pages(cls, values):
        """Validate page consistency across document."""
        pages = values.get('pages', [])
        
        # Check page numbers are sequential
        page_nums = [p.page_num for p in pages]
        if page_nums != list(range(1, len(page_nums) + 1)):
            raise ValueError(f"Page numbers not sequential: {page_nums}")
        
        return values
    
    def get_table_of_contents(self) -> List[Dict[str, Any]]:
        """Generate simple table of contents based on headings."""
        toc = []
        for page in self.pages:
            for block in page.text_blocks:
                # Heuristic: large bold text might be headings
                if block.font and 'bold' in block.font.lower() and block.size and block.size > 14:
                    toc.append({
                        "title": block.text[:100],
                        "page": page.page_num,
                        "bbox": block.bbox
                    })
        return toc
    
    def extract_facts(self) -> List[Dict[str, Any]]:
        """Extract key-value facts from document."""
        facts = []
        
        # Simple pattern matching for financial facts
        import re
        patterns = {
            "revenue": r'revenue[:\s]*\$?([\d,]+(?:\.\d+)?[MB]?)',
            "profit": r'profit[:\s]*\$?([\d,]+(?:\.\d+)?[MB]?)',
            "date": r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
        }
        
        for page in self.pages:
            for block in page.text_blocks:
                for fact_type, pattern in patterns.items():
                    matches = re.findall(pattern, block.text.lower())
                    for match in matches:
                        facts.append({
                            "type": fact_type,
                            "value": match,
                            "page": page.page_num,
                            "bbox": block.bbox,
                            "confidence": block.confidence
                        })
        
        return facts
    
    def get_provenance_chain(self, claim: str) -> List[Dict[str, Any]]:
        """Get provenance chain for a specific claim."""
        provenance = []
        
        for page in self.pages:
            for block in page.text_blocks:
                if claim.lower() in block.text.lower():
                    provenance.append({
                        "document_id": self.document_id,
                        "page": page.page_num,
                        "bbox": block.bbox,
                        "text": block.text,
                        "content_hash": block.content_hash,
                        "confidence": block.confidence
                    })
        
        return provenance