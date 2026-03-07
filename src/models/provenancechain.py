"""
Provenance models for audit trail and claim verification.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class SourceCitation(BaseModel):
    """
    Single source citation with spatial provenance.
    """
    document_name: str = Field(..., description="Source document filename")
    page_number: int = Field(..., ge=1, description="Page number")
    bbox: Optional[Dict[str, float]] = Field(
        None,
        description="Bounding box coordinates (x0, y0, x1, y1)"
    )
    content_hash: str = Field(..., description="Hash of source content")
    extracted_text: Optional[str] = Field(
        None,
        description="Extracted text snippet"
    )
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Extraction confidence")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_name": "CBE_ANNUAL_REPORT_2023-24.pdf",
                "page_number": 42,
                "bbox": {"x0": 100, "y0": 200, "x1": 500, "y1": 250},
                "content_hash": "a1b2c3d4e5f6g7h8",
                "extracted_text": "Revenue: $4.2 billion",
                "confidence": 0.98
            }
        }


class ProvenanceChain(BaseModel):
    """
    Chain of source citations supporting an answer.
    """
    citations: List[SourceCitation] = Field(
        ...,
        description="Source citations in order of relevance"
    )
    answer_text: str = Field(..., description="The generated answer")
    confidence_score: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Overall confidence in answer"
    )
    verification_status: str = Field(
        "unverified",
        pattern="^(verified|partial|unverified|contradicted)$",
        description="Verification status"
    )
    
    def add_citation(self, citation: SourceCitation):
        """Add a citation to the chain"""
        self.citations.append(citation)
    
    def verify_against_source(self, source_text: str) -> bool:
        """
        Verify a claim against source text.
        Simplified verification - in production would use semantic similarity.
        """
        if not self.citations:
            return False
        
        # Check if answer appears in source text
        answer_keywords = set(self.answer_text.lower().split())
        source_keywords = set(source_text.lower().split())
        
        # Calculate overlap ratio
        overlap = answer_keywords.intersection(source_keywords)
        overlap_ratio = len(overlap) / len(answer_keywords) if answer_keywords else 0
        
        self.verification_status = "verified" if overlap_ratio > 0.7 else "partial"
        return overlap_ratio > 0.5
    
    class Config:
        json_schema_extra = {
            "example": {
                "citations": [
                    {
                        "document_name": "CBE_ANNUAL_REPORT_2023-24.pdf",
                        "page_number": 42,
                        "bbox": {"x0": 100, "y0": 200, "x1": 500, "y1": 250},
                        "content_hash": "a1b2c3d4e5f6g7h8"
                    }
                ],
                "answer_text": "Revenue was $4.2 billion in 2023",
                "confidence_score": 0.95,
                "verification_status": "verified"
            }
        }


class AuditEntry(BaseModel):
    """
    Entry in the extraction ledger for audit trail.
    """
    doc_id: str = Field(..., description="Document identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    strategy_used: str = Field(..., description="Extraction strategy")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    cost_estimate_usd: float = Field(..., ge=0.0)
    processing_time_seconds: float = Field(..., ge=0.0)
    page_count: int = Field(..., ge=1)
    error_count: int = Field(0, ge=0)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)