"""Enhanced provenance models with spatial indexing and cryptographic verification."""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import hashlib
import json


class BoundingBox(BaseModel):
    """Spatial coordinates with document-relative positioning."""
    x1: float = Field(..., ge=0, description="Left coordinate")
    y1: float = Field(..., ge=0, description="Top coordinate") 
    x2: float = Field(..., ge=0, description="Right coordinate")
    y2: float = Field(..., ge=0, description="Bottom coordinate")
    page_width: Optional[float] = Field(None, description="Page width for normalization")
    page_height: Optional[float] = Field(None, description="Page height for normalization")
    
    @validator('x2')
    def x2_must_be_greater(cls, v, values):
        if 'x1' in values and v <= values['x1']:
            raise ValueError(f'x2 ({v}) must be greater than x1 ({values["x1"]})')
        return v
    
    @validator('y2')
    def y2_must_be_greater(cls, v, values):
        if 'y1' in values and v <= values['y1']:
            raise ValueError(f'y2 ({v}) must be greater than y1 ({values["y1"]})')
        return v
    
    def to_normalized(self) -> Tuple[float, float, float, float]:
        """Return normalized coordinates (0-1000 range) for cross-document consistency."""
        if self.page_width and self.page_height:
            return (
                (self.x1 / self.page_width) * 1000,
                (self.y1 / self.page_height) * 1000,
                (self.x2 / self.page_width) * 1000,
                (self.y2 / self.page_height) * 1000
            )
        return (self.x1, self.y1, self.x2, self.y2)
    
    def area(self) -> float:
        """Calculate bounding box area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def intersection_over_union(self, other: 'BoundingBox') -> float:
        """Calculate IoU with another bounding box."""
        x_left = max(self.x1, other.x1)
        y_top = max(self.y1, other.y1)
        x_right = min(self.x2, other.x2)
        y_bottom = min(self.y2, other.y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = self.area() + other.area() - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0


class ContentHash(BaseModel):
    """Cryptographic hash for content verification."""
    algorithm: str = Field("sha256", regex="^(sha256|sha512|md5)$")
    hash_value: str = Field(..., min_length=32, max_length=128)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @classmethod
    def from_content(cls, content: str, algorithm: str = "sha256") -> 'ContentHash':
        """Generate hash from content string."""
        if algorithm == "sha256":
            hash_obj = hashlib.sha256(content.encode())
        elif algorithm == "sha512":
            hash_obj = hashlib.sha512(content.encode())
        else:
            hash_obj = hashlib.md5(content.encode())
        
        return cls(
            algorithm=algorithm,
            hash_value=hash_obj.hexdigest()
        )
    
    def verify(self, content: str) -> bool:
        """Verify content matches hash."""
        return self.hash_value == hashlib.new(
            self.algorithm, content.encode()
        ).hexdigest()


class ProvenanceItem(BaseModel):
    """Single provenance item with full spatial and content tracking."""
    document_id: str
    page_number: int = Field(..., ge=1)
    bbox: BoundingBox
    content_hash: ContentHash
    extracted_text: str = Field(..., max_length=10000)
    confidence: float = Field(..., ge=0.0, le=1.0)
    extraction_strategy: str
    chunk_id: Optional[str] = None
    parent_section: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_citation(self) -> str:
        """Generate human-readable citation."""
        return f"{self.document_id}, page {self.page_number}, {self.bbox.to_normalized()}"
    
    def to_markdown(self) -> str:
        """Generate markdown representation with citation."""
        return f"> {self.extracted_text}\n>\n> — *{self.to_citation()}*"


class ProvenanceChain(BaseModel):
    """Chain of provenance items supporting a claim."""
    claim: str
    items: List[ProvenanceItem] = Field(..., min_items=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    verification_status: str = Field("unverified", regex="^(unverified|verified|contradicted)$")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def verify(self) -> bool:
        """Verify all items in the chain."""
        all_verified = all(item.content_hash.verify(item.extracted_text) for item in self.items)
        self.verification_status = "verified" if all_verified else "contradicted"
        return all_verified
    
    def merge(self, other: 'ProvenanceChain') -> 'ProvenanceChain':
        """Merge two provenance chains."""
        return ProvenanceChain(
            claim=f"{self.claim} [AND] {other.claim}",
            items=self.items + other.items,
            confidence=min(self.confidence, other.confidence),
            verification_status="unverified"
        )