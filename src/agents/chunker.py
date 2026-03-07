"""
Semantic Chunking Engine for converting extracted documents to LDUs.
"""

from typing import List, Dict, Optional
import hashlib
import tiktoken

from ..models.extracted_document import (
    ExtractedDocument, ContentBlock, BlockType, Table
)
from ..models.ldu import LDU, ChunkType, ChunkCollection
from ..utils.chunk_validator import ChunkValidator


class ChunkingEngine:
    """
    Converts extracted documents to Logical Document Units (LDUs).
    """
    
    def __init__(
        self,
        max_tokens: int = 512,
        respect_semantic_boundaries: bool = True,
        rules: Optional[Dict] = None
    ):
        """
        Initialize chunking engine.
        
        Args:
            max_tokens: Maximum tokens per chunk
            respect_semantic_boundaries: Whether to respect paragraph/table boundaries
            rules: Chunking rules configuration
        """
        self.max_tokens = max_tokens
        self.respect_semantic_boundaries = respect_semantic_boundaries
        self.validator = ChunkValidator(rules or {})
        
        # Tokenizer for counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def chunk(self, doc: ExtractedDocument) -> ChunkCollection:
        """
        Convert extracted document to LDUs.
        
        Args:
            doc: Extracted document
            
        Returns:
            Collection of LDUs
        """
        chunks: List[LDU] = []
        
        # First pass: create chunks for special elements
        special_chunks = self._create_special_chunks(doc)
        chunks.extend(special_chunks)
        
        # Second pass: chunk text blocks
        text_chunks = self._chunk_text_blocks(doc)
        chunks.extend(text_chunks)
        
        # Build section hierarchy
        chunks = self._assign_section_hierarchy(chunks)
        
        # Resolve relationships
        chunks = self._resolve_relationships(chunks)
        
        # Sort by page and position
        chunks = self._sort_chunks(chunks)
        
        # Validate all chunks
        violations = self.validator.validate_all(chunks)
        if violations:
            print(f"Warning: Found {len(violations)} chunks with validation issues")
            for chunk_id, issues in list(violations.items())[:5]:
                print(f"  {chunk_id}: {issues[0]}")
        
        return ChunkCollection(
            doc_id=doc.doc_id,
            chunks=chunks,
            chunk_count=len(chunks),
            total_tokens=sum(c.token_count for c in chunks)
        )
    
    def _create_special_chunks(self, doc: ExtractedDocument) -> List[LDU]:
        """Create chunks for tables, figures, etc."""
        chunks = []
        
        # Create table chunks
        for i, table in enumerate(doc.tables):
            chunk = self._table_to_chunk(table, doc.doc_id, i)
            if chunk:
                chunks.append(chunk)
        
        # Create figure chunks
        for i, figure in enumerate(doc.figures):
            chunk = self._figure_to_chunk(figure, doc.doc_id, i)
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _table_to_chunk(self, table: Table, doc_id: str, index: int) -> Optional[LDU]:
        """Convert table to LDU"""
        content = table.to_markdown()
        token_count = len(self.tokenizer.encode(content))
        
        # Check if table exceeds max tokens (unlikely for tables)
        if token_count > self.max_tokens * 2:
            # Table too large - split by rows? Better to keep intact
            # but warn
            print(f"Warning: Large table ({token_count} tokens) may cause issues")
        
        return LDU(
            chunk_id=f"{doc_id}_table_{index}",
            doc_id=doc_id,
            chunk_type=ChunkType.TABLE,
            content=content,
            page_refs=[table.bbox.page_number] if table.bbox else [1],
            bounding_boxes=[table.bbox.to_dict()] if table.bbox else [],
            token_count=token_count,
            content_hash="",  # Will be auto-generated
            metadata={
                "headers": table.headers,
                "caption": table.caption,
                "row_count": len(table.rows)
            }
        )
    
    def _figure_to_chunk(self, figure, doc_id: str, index: int) -> Optional[LDU]:
        """Convert figure to LDU"""
        content = figure.description or f"Figure {index + 1}"
        if figure.caption:
            content = f"{content}\n\nCaption: {figure.caption}"
        
        token_count = len(self.tokenizer.encode(content))
        
        return LDU(
            chunk_id=f"{doc_id}_figure_{index}",
            doc_id=doc_id,
            chunk_type=ChunkType.FIGURE,
            content=content,
            page_refs=[figure.bbox.page_number] if figure.bbox else [1],
            bounding_boxes=[figure.bbox.to_dict()] if figure.bbox else [],
            token_count=token_count,
            content_hash="",  # Will be auto-generated
            metadata={"caption": figure.caption}
        )
    
    def _chunk_text_blocks(self, doc: ExtractedDocument) -> List[LDU]:
        """Chunk text blocks into LDUs"""
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        
        # Get text blocks in reading order
        text_blocks = [
            b for b in doc.blocks
            if b.block_type in [BlockType.TEXT, BlockType.HEADER, BlockType.LIST]
        ]
        
        for block in text_blocks:
            block_tokens = len(self.tokenizer.encode(block.content))
            
            # Check if this is a semantic boundary
            is_boundary = (
                block.block_type == BlockType.HEADER or
                block.content.startswith(('#', 'Chapter', 'Section')) or
                block.content.strip().endswith(('.', '!', '?'))
            )
            
            # If adding this block would exceed max tokens and we have content,
            # or if we hit a semantic boundary with enough content
            if (current_tokens + block_tokens > self.max_tokens and current_chunk) or \
               (is_boundary and current_tokens > self.max_tokens // 2 and current_chunk):
                # Create chunk from current blocks
                chunk = self._create_text_chunk(
                    current_chunk, doc.doc_id, chunk_index
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk
                current_chunk = [block]
                current_tokens = block_tokens
            else:
                # Add to current chunk
                current_chunk.append(block)
                current_tokens += block_tokens
        
        # Add remaining blocks
        if current_chunk:
            chunk = self._create_text_chunk(
                current_chunk, doc.doc_id, chunk_index
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _create_text_chunk(
        self,
        blocks: List[ContentBlock],
        doc_id: str,
        index: int
    ) -> Optional[LDU]:
        """Create text chunk from list of blocks"""
        if not blocks:
            return None
        
        # Combine content
        content = "\n\n".join(b.content for b in blocks if b.content)
        
        # Determine chunk type
        if any(b.block_type == BlockType.HEADER for b in blocks):
            chunk_type = ChunkType.SECTION
        elif any(b.block_type == BlockType.LIST for b in blocks):
            chunk_type = ChunkType.LIST
        else:
            chunk_type = ChunkType.PARAGRAPH
        
        # Collect page references
        page_refs = list(set(
            b.bbox.page_number for b in blocks if b.bbox
        ))
        if not page_refs and blocks:
            page_refs = [1]
        
        # Collect bounding boxes
        bounding_boxes = [
            b.bbox.to_dict() for b in blocks if b.bbox
        ]
        
        token_count = len(self.tokenizer.encode(content))
        
        return LDU(
            chunk_id=f"{doc_id}_text_{index}",
            doc_id=doc_id,
            chunk_type=chunk_type,
            content=content,
            page_refs=sorted(page_refs),
            bounding_boxes=bounding_boxes,
            token_count=token_count,
            content_hash="",  # Will be auto-generated
            metadata={
                "block_count": len(blocks),
                "has_header": any(b.block_type == BlockType.HEADER for b in blocks)
            }
        )
    
    def _assign_section_hierarchy(self, chunks: List[LDU]) -> List[LDU]:
        """Assign section hierarchy to chunks"""
        # Find header chunks
        header_chunks = [
            c for c in chunks
            if c.chunk_type == ChunkType.SECTION and
            c.metadata.get("has_header", False)
        ]
        
        # Sort by page and position (simplified)
        header_chunks.sort(key=lambda c: (c.page_refs[0] if c.page_refs else 0))
        
        # Build hierarchy
        hierarchy = []
        for chunk in chunks:
            if chunk.chunk_type == ChunkType.SECTION and chunk.metadata.get("has_header"):
                # This is a header, add to hierarchy
                hierarchy.append(chunk.content.split('\n')[0][:50])
                chunk.section_hierarchy = hierarchy.copy()
            else:
                # Content chunk, assign current hierarchy
                chunk.section_hierarchy = hierarchy.copy()
                if hierarchy:
                    chunk.parent_section = hierarchy[-1]
        
        return chunks
    
    def _resolve_relationships(self, chunks: List[LDU]) -> List[LDU]:
        """Resolve relationships between chunks (e.g., figure-caption)"""
        # Group by page
        page_groups = {}
        for chunk in chunks:
            for page in chunk.page_refs:
                if page not in page_groups:
                    page_groups[page] = []
                page_groups[page].append(chunk)
        
        # Find figure-caption pairs
        for page, page_chunks in page_groups.items():
            figures = [c for c in page_chunks if c.chunk_type == ChunkType.FIGURE]
            captions = [c for c in page_chunks if c.chunk_type == ChunkType.CAPTION]
            
            # Simple matching: associate captions with nearest figure
            # In production, would use spatial proximity
            for i, figure in enumerate(figures):
                if i < len(captions):
                    figure.related_chunks.append(captions[i].chunk_id)
                    captions[i].related_chunks.append(figure.chunk_id)
        
        return chunks
    
    def _sort_chunks(self, chunks: List[LDU]) -> List[LDU]:
        """Sort chunks by page and position"""
        def chunk_sort_key(chunk):
            # Primary: first page
            first_page = chunk.page_refs[0] if chunk.page_refs else 9999
            
            # Secondary: approximate y-position from first bounding box
            first_y = 0
            if chunk.bounding_boxes:
                first_y = chunk.bounding_boxes[0].get('y0', 0) if chunk.bounding_boxes else 0
            
            return (first_page, first_y)
        
        return sorted(chunks, key=chunk_sort_key)