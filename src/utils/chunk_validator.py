"""
Chunk validator enforcing the chunking constitution rules.
"""

from typing import List, Dict, Any, Optional
from ..models.ldu import LDU, ChunkType
from ..models.extracted_document import Table


class ChunkValidator:
    """
    Validates chunks against the chunking constitution rules.
    """
    
    def __init__(self, rules: Optional[Dict[str, bool]] = None):
        """
        Initialize validator with configurable rules.
        
        Args:
            rules: Dictionary of rule names to boolean (enabled/disabled)
        """
        self.rules = rules or {
            "no_split_table_cells": True,
            "caption_with_figure": True,
            "list_integrity": True,
            "section_hierarchy": True,
            "cross_reference_resolution": True,
            "max_token_limit": 2048,
            "min_token_limit": 10
        }
    
    def validate_chunk(self, chunk: LDU, all_chunks: List[LDU]) -> List[str]:
        """
        Validate a single chunk against all rules.
        
        Args:
            chunk: Chunk to validate
            all_chunks: All chunks from the document
            
        Returns:
            List of violation messages (empty if valid)
        """
        violations = []
        
        # Rule 1: No table cell splitting
        if self.rules.get("no_split_table_cells", True):
            if chunk.chunk_type == ChunkType.TABLE:
                table_violations = self._check_table_integrity(chunk)
                violations.extend(table_violations)
        
        # Rule 2: Caption with figure
        if self.rules.get("caption_with_figure", True):
            if chunk.chunk_type == ChunkType.FIGURE:
                caption_violations = self._check_figure_caption(chunk, all_chunks)
                violations.extend(caption_violations)
            elif chunk.chunk_type == ChunkType.CAPTION:
                figure_violations = self._check_caption_figure(chunk, all_chunks)
                violations.extend(figure_violations)
        
        # Rule 3: List integrity
        if self.rules.get("list_integrity", True):
            if chunk.chunk_type == ChunkType.LIST:
                list_violations = self._check_list_integrity(chunk)
                violations.extend(list_violations)
        
        # Rule 4: Section hierarchy
        if self.rules.get("section_hierarchy", True):
            hierarchy_violations = self._check_section_hierarchy(chunk)
            violations.extend(hierarchy_violations)
        
        # Rule 5: Cross-reference resolution
        if self.rules.get("cross_reference_resolution", True):
            ref_violations = self._check_cross_references(chunk, all_chunks)
            violations.extend(ref_violations)
        
        # Token limits
        if "max_token_limit" in self.rules:
            max_tokens = self.rules["max_token_limit"]
            if chunk.token_count > max_tokens:
                violations.append(
                    f"Chunk exceeds max token limit: {chunk.token_count} > {max_tokens}"
                )
        
        if "min_token_limit" in self.rules:
            min_tokens = self.rules["min_token_limit"]
            if chunk.token_count < min_tokens:
                violations.append(
                    f"Chunk below min token limit: {chunk.token_count} < {min_tokens}"
                )
        
        return violations
    
    def _check_table_integrity(self, chunk: LDU) -> List[str]:
        """Check if table chunk preserves cell-header relationships"""
        violations = []
        
        # Check if table has proper structure
        if "headers" not in chunk.metadata:
            violations.append("Table chunk missing headers metadata")
        
        if "row_count" not in chunk.metadata:
            violations.append("Table chunk missing row count")
        
        # Check if table appears complete
        content_lines = chunk.content.split('\n')
        if len(content_lines) < 3:  # At least header + separator + 1 row
            violations.append("Table chunk appears truncated")
        
        return violations
    
    def _check_figure_caption(self, figure_chunk: LDU, all_chunks: List[LDU]) -> List[str]:
        """Check if figure has associated caption"""
        # Look for caption chunks on same page
        figure_page = figure_chunk.page_refs[0] if figure_chunk.page_refs else -1
        
        captions = [
            c for c in all_chunks
            if c.chunk_type == ChunkType.CAPTION
            and figure_page in c.page_refs
        ]
        
        if not captions:
            return ["Figure has no associated caption on same page"]
        
        # Check if caption is linked
        if figure_chunk.chunk_id not in [c.related_chunks for c in captions]:
            return ["Figure not linked to its caption via related_chunks"]
        
        return []
    
    def _check_caption_figure(self, caption_chunk: LDU, all_chunks: List[LDU]) -> List[str]:
        """Check if caption has associated figure"""
        caption_page = caption_chunk.page_refs[0] if caption_chunk.page_refs else -1
        
        figures = [
            f for f in all_chunks
            if f.chunk_type == ChunkType.FIGURE
            and caption_page in f.page_refs
        ]
        
        if not figures:
            return ["Caption has no associated figure on same page"]
        
        return []
    
    def _check_list_integrity(self, list_chunk: LDU) -> List[str]:
        """Check if list maintains its structure"""
        content = list_chunk.content
        
        # Check for list markers
        lines = content.split('\n')
        list_markers = 0
        expected_markers = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            expected_markers += 1
            
            # Check for common list markers
            if (line.startswith(('•', '-', '*')) or
                line[0].isdigit() and line[1:3] in ('. ', ') ')):
                list_markers += 1
        
        if expected_markers > 0 and list_markers < expected_markers * 0.7:
            return ["List appears to have missing or incorrect markers"]
        
        return []
    
    def _check_section_hierarchy(self, chunk: LDU) -> List[str]:
        """Check if section hierarchy is consistent"""
        violations = []
        
        # Check if section hierarchy is present for non-standalone chunks
        if chunk.chunk_type not in [ChunkType.HEADER, ChunkType.SECTION]:
            if not chunk.section_hierarchy and not chunk.parent_section:
                violations.append(
                    f"Chunk missing section hierarchy information"
                )
        
        # Check hierarchy consistency
        if chunk.section_hierarchy:
            if chunk.parent_section:
                if chunk.parent_section != chunk.section_hierarchy[-1]:
                    violations.append(
                        f"Parent section '{chunk.parent_section}' does not match "
                        f"last in hierarchy '{chunk.section_hierarchy[-1]}'"
                    )
        
        return violations
    
    def _check_cross_references(self, chunk: LDU, all_chunks: List[LDU]) -> List[str]:
        """Check if cross-references are resolved"""
        content = chunk.content.lower()
        violations = []
        
        # Look for common reference patterns
        import re
        
        # Pattern for "see Table X", "as shown in Figure Y", etc.
        ref_patterns = [
            r'see\s+(?:table|figure|section)\s+(\d+(?:\.\d+)?)',
            r'as\s+shown\s+in\s+(?:table|figure|section)\s+(\d+(?:\.\d+)?)',
            r'refer\s+to\s+(?:table|figure|section)\s+(\d+(?:\.\d+)?)',
            r'in\s+(?:table|figure|section)\s+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in ref_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # Check if referenced chunk exists
                referenced = False
                for other in all_chunks:
                    if other.chunk_id.endswith(match) or match in other.chunk_id:
                        referenced = True
                        break
                
                if not referenced:
                    violations.append(
                        f"Unresolved cross-reference to '{match}' in content"
                    )
        
        return violations
    
    def validate_all(self, chunks: List[LDU]) -> Dict[str, List[str]]:
        """
        Validate all chunks in a collection.
        
        Returns:
            Dictionary mapping chunk_id to list of violations
        """
        all_violations = {}
        
        for chunk in chunks:
            violations = self.validate_chunk(chunk, chunks)
            if violations:
                all_violations[chunk.chunk_id] = violations
        
        return all_violations