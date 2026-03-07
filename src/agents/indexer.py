"""
PageIndex builder for hierarchical document navigation.
"""

from typing import List, Dict, Optional, Any
import hashlib
import json
from pathlib import Path
import numpy as np

from ..models.page_index import PageIndex, Section, DataType
from ..models.ldu import LDU, ChunkCollection


class PageIndexBuilder:
    """
    Builds hierarchical navigation structure for documents.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",  # For summaries (simulated)
        embedding_model: Optional[str] = None
    ):
        """
        Initialize page index builder.
        
        Args:
            model_name: Model for generating summaries
            embedding_model: Model for section embeddings
        """
        self.model_name = model_name
        self.embedding_model = embedding_model
    
    def build_index(self, chunks: ChunkCollection) -> PageIndex:
        """
        Build PageIndex from chunks.
        
        Args:
            chunks: Chunk collection
            
        Returns:
            PageIndex for document
        """
        sections: Dict[str, Section] = {}
        root_sections = []
        
        # Group chunks by section
        section_chunks = self._group_by_section(chunks.chunks)
        
        # Build section tree
        for section_path, section_chunks_list in section_chunks.items():
            if not section_path:  # No section info
                continue
            
            # Create section for each level in path
            parent_id = None
            for level, section_title in enumerate(section_path, 1):
                section_id = self._generate_section_id(section_title, level)
                
                if section_id not in sections:
                    # Determine page range
                    page_nums = []
                    data_types = set()
                    for c in section_chunks_list:
                        page_nums.extend(c.page_refs)
                        data_types.add(c.chunk_type.value)
                    
                    page_start = min(page_nums) if page_nums else 1
                    page_end = max(page_nums) if page_nums else None
                    
                    # Create section
                    sections[section_id] = Section(
                        section_id=section_id,
                        title=section_title,
                        level=level,
                        page_start=page_start,
                        page_end=page_end,
                        parent_id=parent_id,
                        child_sections=[],
                        data_types_present=[
                            DataType(dt) for dt in data_types
                            if dt in [d.value for d in DataType]
                        ],
                        chunk_count=len(section_chunks_list),
                        table_count=sum(
                            1 for c in section_chunks_list
                            if c.chunk_type.value == "table"
                        ),
                        figure_count=sum(
                            1 for c in section_chunks_list
                            if c.chunk_type.value == "figure"
                        )
                    )
                    
                    if parent_id:
                        if parent_id in sections:
                            sections[parent_id].child_sections.append(section_id)
                    else:
                        root_sections.append(section_id)
                
                parent_id = section_id
        
        # Generate summaries for sections
        sections = self._generate_summaries(sections, section_chunks)
        
        # Extract key entities
        sections = self._extract_entities(sections, section_chunks)
        
        return PageIndex(
            doc_id=chunks.doc_id,
            filename=f"{chunks.doc_id}.pdf",  # Would need original filename
            total_pages=self._get_total_pages(chunks.chunks),
            root_sections=root_sections,
            sections=sections,
            section_titles=[s.title for s in sections.values()]
        )
    
    def _group_by_section(self, chunks: List[LDU]) -> Dict[tuple, List[LDU]]:
        """Group chunks by their section hierarchy"""
        groups = {}
        
        for chunk in chunks:
            # Convert hierarchy to tuple for dict key
            section_path = tuple(chunk.section_hierarchy)
            
            if section_path not in groups:
                groups[section_path] = []
            groups[section_path].append(chunk)
        
        return groups
    
    def _generate_section_id(self, title: str, level: int) -> str:
        """Generate unique section ID"""
        hash_input = f"{title}_{level}"
        return f"sec_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"
    
    def _generate_summaries(
        self,
        sections: Dict[str, Section],
        section_chunks: Dict[tuple, List[LDU]]
    ) -> Dict[str, Section]:
        """Generate LLM summaries for sections"""
        for section_path, chunks in section_chunks.items():
            if not section_path:
                continue
            
            # Get the deepest section
            section_title = section_path[-1]
            section_id = self._generate_section_id(section_title, len(section_path))
            
            if section_id in sections:
                # Combine chunk content
                content = "\n\n".join([
                    c.content for c in chunks
                    if c.chunk_type.value not in ["table", "figure"]
                ][:1000])  # Limit for summary
                
                # Generate summary (simulated)
                summary = self._simulate_summary(section_title, content)
                sections[section_id].summary = summary
        
        return sections
    
    def _simulate_summary(self, title: str, content: str) -> str:
        """Simulate LLM summary generation"""
        # In production, would call actual LLM
        # This is a simplified placeholder
        words = content.split()[:30]
        preview = " ".join(words)
        
        if len(words) < 10:
            return f"This section covers {title.lower()}."
        else:
            return f"Overview of {title.lower()}. {preview}..."
    
    def _extract_entities(
        self,
        sections: Dict[str, Section],
        section_chunks: Dict[tuple, List[LDU]]
    ) -> Dict[str, Section]:
        """Extract key entities from sections"""
        import re
        
        # Common entity patterns
        patterns = {
            "money": r'\$\d+(?:\.\d+)?(?:\s?(?:billion|million|thousand|B|M|K))?',
            "date": r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}',
            "percentage": r'\d+(?:\.\d+)?%',
            "number": r'\d+(?:,\d{3})*(?:\.\d+)?'
        }
        
        for section_path, chunks in section_chunks.items():
            if not section_path:
                continue
            
            section_title = section_path[-1]
            section_id = self._generate_section_id(section_title, len(section_path))
            
            if section_id in sections:
                # Combine content
                content = " ".join([c.content for c in chunks])
                
                # Extract entities
                entities = []
                for pattern_name, pattern in patterns.items():
                    matches = re.findall(pattern, content)
                    entities.extend(matches[:3])  # Limit per pattern
                
                sections[section_id].key_entities = list(set(entities))[:10]  # Unique, limited
        
        return sections
    
    def _get_total_pages(self, chunks: List[LDU]) -> int:
        """Get total pages from chunks"""
        all_pages = set()
        for chunk in chunks:
            all_pages.update(chunk.page_refs)
        return max(all_pages) if all_pages else 1
    
    async def save_index(self, index: PageIndex, output_dir: str = ".refinery/pageindex"):
        """Save PageIndex to JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / f"{index.doc_id}_pageindex.json"
        
        # Convert to dict for serialization
        index_dict = {
            "doc_id": index.doc_id,
            "filename": index.filename,
            "total_pages": index.total_pages,
            "root_sections": index.root_sections,
            "sections": {
                sid: {
                    "section_id": s.section_id,
                    "title": s.title,
                    "level": s.level,
                    "page_start": s.page_start,
                    "page_end": s.page_end,
                    "parent_id": s.parent_id,
                    "child_sections": s.child_sections,
                    "summary": s.summary,
                    "key_entities": s.key_entities,
                    "data_types_present": [dt.value for dt in s.data_types_present],
                    "chunk_count": s.chunk_count,
                    "table_count": s.table_count,
                    "figure_count": s.figure_count
                }
                for sid, s in index.sections.items()
            },
            "section_titles": index.section_titles,
            "created_at": index.created_at.isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(index_dict, f, indent=2)
        
        return filepath