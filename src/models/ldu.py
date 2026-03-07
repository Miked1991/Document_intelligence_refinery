"""
Logical Document Unit (LDU) models for semantic chunking.
These represent semantically coherent, self-contained units for RAG.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
import hashlib
import json


class ChunkType(str, Enum):
    """Types of logical document units"""
    SECTION = "section"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    LIST = "list"
    EQUATION = "equation"
    HEADER = "header"
    FOOTNOTE = "footnote"
    CAPTION = "caption"


class LDU(BaseModel):
    """
    Logical Document Unit - semantically coherent, self-contained unit.
    """
    chunk_id: str = Field(..., description="Unique chunk identifier")
    doc_id: str = Field(..., description="Source document identifier")
    chunk_type: ChunkType = Field(..., description="Type of chunk")
    content: str = Field(..., description="Text content of the chunk")
    
    # Structural context
    section_hierarchy: List[str] = Field(
        default_factory=list,
        description="Hierarchical section path (e.g., ['Executive Summary', 'Financial Highlights'])"
    )
    parent_section: Optional[str] = Field(
        None,
        description="Immediate parent section title"
    )
    
    # Spatial provenance
    page_refs: List[int] = Field(..., description="Page numbers containing this chunk")
    bounding_boxes: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Bounding boxes with page numbers"
    )
    
    # Metadata
    token_count: int = Field(..., ge=1, description="Number of tokens")
    content_hash: str = Field(..., description="Hash for verification")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., table headers, figure caption)"
    )
    
    # Relationships
    related_chunks: List[str] = Field(
        default_factory=list,
        description="IDs of related chunks (e.g., figure-caption pairs)"
    )
    
    @validator('content_hash', pre=True, always=True)
    def generate_content_hash(cls, v, values):
        """Generate or validate content hash"""
        if v is not None:
            return v
        
        # Generate hash from content and metadata
        content_str = json.dumps({
            'doc_id': values.get('doc_id'),
            'content': values.get('content'),
            'chunk_type': values.get('chunk_type'),
            'page_refs': values.get('page_refs')
        }, sort_keys=True)
        
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "chunk_001",
                "doc_id": "cbe_annual_report_2023",
                "chunk_type": "table",
                "content": "| Year | Revenue |\n| 2023 | $4.2B |",
                "section_hierarchy": ["Financial Statements", "Income Statement"],
                "parent_section": "Income Statement",
                "page_refs": [42],
                "token_count": 25,
                "content_hash": "a1b2c3d4e5f6g7h8",
                "metadata": {
                    "headers": ["Year", "Revenue"],
                    "caption": "Annual Revenue Summary"
                }
            }
        }


class ChunkCollection(BaseModel):
    """Collection of LDUs from a document"""
    doc_id: str = Field(..., description="Document identifier")
    chunks: List[LDU] = Field(..., description="List of chunks")
    chunk_count: int = Field(..., ge=0, description="Number of chunks")
    total_tokens: int = Field(..., ge=0, description="Total tokens")
    
    @validator('chunk_count', always=True)
    def validate_chunk_count(cls, v, values):
        """Ensure chunk_count matches actual chunks"""
        if 'chunks' in values:
            return len(values['chunks'])
        return v
    
    @validator('total_tokens', always=True)
    def validate_total_tokens(cls, v, values):
        """Calculate total tokens from chunks"""
        if 'chunks' in values:
            return sum(chunk.token_count for chunk in values['chunks'])
        return v