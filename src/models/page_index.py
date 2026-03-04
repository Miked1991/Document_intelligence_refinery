"""Enhanced PageIndex models with comprehensive navigation and entity tracking."""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Set
from datetime import datetime
from enum import Enum


class ContentType(str, Enum):
    """Types of content that can appear in sections."""
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"
    CODE = "code"
    LIST = "list"
    QUOTE = "quote"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"


class EntityType(str, Enum):
    """Types of entities that can be extracted."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    MONEY = "money"
    PERCENTAGE = "percentage"
    PRODUCT = "product"
    LAW = "law"
    REGULATION = "regulation"
    CASE_CITATION = "case_citation"
    FINANCIAL_METRIC = "financial_metric"
    TECHNICAL_TERM = "technical_term"


class ExtractedEntity(BaseModel):
    """Named entity with context."""
    text: str
    entity_type: EntityType
    confidence: float = Field(..., ge=0.0, le=1.0)
    mentions: List[int] = Field(default_factory=list)  # Page numbers where entity appears
    first_mention_page: Optional[int] = None
    last_mention_page: Optional[int] = None
    context_snippets: List[str] = Field(default_factory=list, max_items=5)


class SectionNode(BaseModel):
    """Enhanced section node with comprehensive metadata."""
    id: str = Field(..., description="Unique section identifier")
    title: str
    level: int = Field(..., ge=0, le=10)
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)
    parent_id: Optional[str] = None
    children: List['SectionNode'] = Field(default_factory=list)
    
    # Content metadata
    content_types: Set[ContentType] = Field(default_factory=set)
    word_count: int = Field(0, ge=0)
    character_count: int = Field(0, ge=0)
    table_count: int = Field(0, ge=0)
    figure_count: int = Field(0, ge=0)
    equation_count: int = Field(0, ge=0)
    
    # Entities
    entities: Dict[EntityType, List[ExtractedEntity]] = Field(default_factory=dict)
    key_terms: List[str] = Field(default_factory=list, max_items=20)
    
    # Summary
    summary: Optional[str] = Field(None, max_length=500)
    summary_embedding: Optional[List[float]] = None
    summary_confidence: float = Field(0.0, ge=0.0, le=1.0)
    
    # Navigation
    prev_section_id: Optional[str] = None
    next_section_id: Optional[str] = None
    toc_entry: bool = Field(True, description="Whether this section appears in TOC")
    
    # Provenance
    content_hash: Optional[str] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('page_end')
    def page_end_must_be_greater(cls, v, values):
        if 'page_start' in values and v < values['page_start']:
            raise ValueError(f'page_end ({v}) must be >= page_start ({values["page_start"]})')
        return v
    
    def add_child(self, child: 'SectionNode'):
        """Add child section with relationship maintenance."""
        child.parent_id = self.id
        child.level = self.level + 1
        self.children.append(child)
        
        # Update content counts
        self.word_count += child.word_count
        self.character_count += child.character_count
        self.table_count += child.table_count
        self.figure_count += child.figure_count
        self.equation_count += child.equation_count
        self.content_types.update(child.content_types)
    
    def get_all_descendants(self) -> List['SectionNode']:
        """Get all descendant sections."""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_all_descendants())
        return descendants
    
    def find_by_title(self, title: str, partial_match: bool = False) -> List['SectionNode']:
        """Find sections by title."""
        results = []
        if partial_match:
            if title.lower() in self.title.lower():
                results.append(self)
        else:
            if title == self.title:
                results.append(self)
        
        for child in self.children:
            results.extend(child.find_by_title(title, partial_match))
        
        return results
    
    def get_table_of_contents(self) -> List[Dict[str, Any]]:
        """Generate TOC entry for this section and children."""
        toc = [{
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "page": self.page_start,
            "has_children": len(self.children) > 0
        }]
        
        for child in sorted(self.children, key=lambda x: x.page_start):
            toc.extend(child.get_table_of_contents())
        
        return toc


class DocumentNavigationIndex(BaseModel):
    """Enhanced PageIndex with navigation and search capabilities."""
    document_id: str
    root: SectionNode
    total_pages: int = Field(..., ge=1)
    total_sections: int = Field(..., ge=1)
    
    # Indexes
    page_to_section: Dict[int, List[str]] = Field(default_factory=dict)  # page -> section_ids
    title_to_section: Dict[str, List[str]] = Field(default_factory=dict)
    entity_index: Dict[EntityType, Dict[str, List[str]]] = Field(default_factory=dict)  # entity text -> section_ids
    
    # Statistics
    content_type_stats: Dict[ContentType, int] = Field(default_factory=dict)
    entity_type_stats: Dict[EntityType, int] = Field(default_factory=dict)
    average_section_length: float = Field(0.0)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field("1.0")
    
    def build_indexes(self):
        """Build all indexes after tree construction."""
        self._build_page_index(self.root)
        self._build_title_index(self.root)
        self._build_entity_index(self.root)
        self._compute_statistics()
    
    def _build_page_index(self, node: SectionNode):
        """Build page to section mapping."""
        for page in range(node.page_start, node.page_end + 1):
            if page not in self.page_to_section:
                self.page_to_section[page] = []
            self.page_to_section[page].append(node.id)
        
        for child in node.children:
            self._build_page_index(child)
    
    def _build_title_index(self, node: SectionNode):
        """Build title to section mapping."""
        title_lower = node.title.lower()
        if title_lower not in self.title_to_section:
            self.title_to_section[title_lower] = []
        self.title_to_section[title_lower].append(node.id)
        
        for child in node.children:
            self._build_title_index(child)
    
    def _build_entity_index(self, node: SectionNode):
        """Build entity index."""
        for entity_type, entities in node.entities.items():
            if entity_type not in self.entity_index:
                self.entity_index[entity_type] = {}
            
            for entity in entities:
                if entity.text not in self.entity_index[entity_type]:
                    self.entity_index[entity_type][entity.text] = []
                self.entity_index[entity_type][entity.text].append(node.id)
        
        for child in node.children:
            self._build_entity_index(child)
    
    def _compute_statistics(self):
        """Compute document statistics."""
        all_sections = self.root.get_all_descendants() + [self.root]
        
        # Content type stats
        for section in all_sections:
            for content_type in section.content_types:
                self.content_type_stats[content_type] = self.content_type_stats.get(content_type, 0) + 1
        
        # Entity type stats
        for entity_type, entities in self.entity_index.items():
            self.entity_type_stats[entity_type] = sum(len(locs) for locs in entities.values())
        
        # Average section length
        if all_sections:
            self.average_section_length = sum(s.word_count for s in all_sections) / len(all_sections)
    
    def find_sections_by_page(self, page: int) -> List[SectionNode]:
        """Find all sections containing a specific page."""
        section_ids = self.page_to_section.get(page, [])
        return [self._find_node_by_id(self.root, sid) for sid in section_ids if self._find_node_by_id(self.root, sid)]
    
    def find_sections_by_entity(self, entity_type: EntityType, entity_text: str) -> List[SectionNode]:
        """Find sections containing a specific entity."""
        section_ids = self.entity_index.get(entity_type, {}).get(entity_text, [])
        return [self._find_node_by_id(self.root, sid) for sid in section_ids if self._find_node_by_id(self.root, sid)]
    
    def _find_node_by_id(self, node: SectionNode, node_id: str) -> Optional[SectionNode]:
        """Find a node by ID."""
        if node.id == node_id:
            return node
        for child in node.children:
            result = self._find_node_by_id(child, node_id)
            if result:
                return result
        return None


PageIndex = DocumentNavigationIndex  # Alias for backward compatibility