"""
PageIndex model for hierarchical document navigation.
Inspired by VectifyAI's PageIndex concept.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class DataType(str, Enum):
    """Types of data present in a section"""
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"
    LIST = "list"
    CODE = "code"


class Section(BaseModel):
    """
    Represents a section in the document hierarchy.
    """
    section_id: str = Field(..., description="Unique section identifier")
    title: str = Field(..., description="Section title")
    level: int = Field(..., ge=1, description="Heading level (1 for top-level)")
    
    # Page range
    page_start: int = Field(..., ge=1, description="Starting page")
    page_end: Optional[int] = Field(None, ge=1, description="Ending page")
    
    # Hierarchy
    parent_id: Optional[str] = Field(None, description="Parent section ID")
    child_sections: List[str] = Field(
        default_factory=list,
        description="Child section IDs"
    )
    
    # Content summary
    summary: Optional[str] = Field(
        None,
        description="LLM-generated summary (2-3 sentences)"
    )
    key_entities: List[str] = Field(
        default_factory=list,
        description="Extracted named entities"
    )
    data_types_present: List[DataType] = Field(
        default_factory=list,
        description="Types of data in this section"
    )
    
    # Statistics
    chunk_count: int = Field(0, ge=0, description="Number of chunks in this section")
    table_count: int = Field(0, ge=0, description="Number of tables")
    figure_count: int = Field(0, ge=0, description="Number of figures")
    
    # Metadata
    embedding: Optional[List[float]] = Field(
        None,
        description="Section embedding for semantic retrieval"
    )


class PageIndex(BaseModel):
    """
    Hierarchical navigation structure for document.
    Enables section-aware retrieval without full document scan.
    """
    doc_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    total_pages: int = Field(..., ge=1, description="Total pages")
    
    # Root sections (top-level)
    root_sections: List[str] = Field(
        default_factory=list,
        description="Top-level section IDs"
    )
    
    # All sections mapped by ID
    sections: Dict[str, Section] = Field(
        default_factory=dict,
        description="All sections mapped by ID"
    )
    
    # Flat list of all section titles for quick lookup
    section_titles: List[str] = Field(
        default_factory=list,
        description="All section titles"
    )
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    build_time_seconds: float = Field(0.0, ge=0, description="Time to build index")
    
    def get_section_path(self, section_id: str) -> List[Section]:
        """Get the full path from root to a section"""
        path = []
        current_id = section_id
        
        while current_id:
            if current_id not in self.sections:
                break
            section = self.sections[current_id]
            path.insert(0, section)
            current_id = section.parent_id
        
        return path
    
    def find_sections_by_title(self, title_substring: str) -> List[Section]:
        """Find sections containing title substring"""
        results = []
        title_lower = title_substring.lower()
        
        for section in self.sections.values():
            if title_lower in section.title.lower():
                results.append(section)
        
        return results
    
    def get_sections_by_page(self, page_num: int) -> List[Section]:
        """Get all sections covering a given page"""
        results = []
        
        for section in self.sections.values():
            if section.page_start <= page_num:
                if section.page_end is None or page_num <= section.page_end:
                    results.append(section)
        
        return results
    
    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": "cbe_annual_report_2023",
                "filename": "CBE_ANNUAL_REPORT_2023-24.pdf",
                "total_pages": 120,
                "root_sections": ["sec_001", "sec_005"],
                "sections": {
                    "sec_001": {
                        "section_id": "sec_001",
                        "title": "Executive Summary",
                        "level": 1,
                        "page_start": 1,
                        "page_end": 3,
                        "summary": "Overview of financial performance for FY2023-24",
                        "key_entities": ["CBE", "$4.2B revenue"],
                        "data_types_present": ["text", "table"]
                    }
                }
            }
        }