"""
Fast text extraction strategy using pdfplumber.
"""

import pdfplumber
from typing import Optional, List, Dict, Any
import hashlib
import time
from datetime import datetime

from .base import ExtractionStrategy
from ..models.extracted_document import (
    ExtractedDocument, ContentBlock, BlockType,
    BoundingBox, Table, Figure
)
from ..models.document_profile import DocumentProfile
from ..utils.confidence_scorer import ConfidenceScorer


class FastTextExtractor(ExtractionStrategy):
    """
    Fast text extraction using pdfplumber.
    Cost: Low ($0.001 per page)
    """
    
    def __init__(self):
        super().__init__("fast_text")
        self.confidence_scorer = ConfidenceScorer()
        self.page_cost = 0.001  # $0.001 per page
    
    async def extract(
        self,
        pdf_path: str,
        profile: Optional[DocumentProfile] = None
    ) -> ExtractedDocument:
        """
        Extract text content using pdfplumber.
        """
        start_time = time.time()
        doc_id = hashlib.md5(pdf_path.encode()).hexdigest()[:12]
        
        blocks: List[ContentBlock] = []
        tables: List[Table] = []
        figures: List[Figure] = []
        page_data = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Store page data for confidence scoring
                page_info = {
                    "page_number": page_num,
                    "width": page.width,
                    "height": page.height,
                    "chars": list(page.chars) if page.chars else [],
                    "images": list(page.images) if page.images else []
                }
                page_data.append(page_info)
                
                # Extract text with position information
                words = page.extract_words(
                    keep_blank_chars=False,
                    use_text_flow=True,
                    extra_attrs=['fontname', 'size']
                )
                
                # Group words into lines (simple heuristic)
                current_block = []
                current_y = None
                
                for word in words:
                    word_y = round(word['top'], 1)
                    
                    if current_y is None or abs(word_y - current_y) < 5:
                        # Same line
                        current_block.append(word)
                    else:
                        # New line
                        if current_block:
                            block = self._words_to_block(
                                current_block, page_num, doc_id
                            )
                            if block:
                                blocks.append(block)
                        current_block = [word]
                    
                    current_y = word_y
                
                # Add last block
                if current_block:
                    block = self._words_to_block(current_block, page_num, doc_id)
                    if block:
                        blocks.append(block)
                
                # Extract tables (basic)
                page_tables = page.extract_tables()
                for table_data in page_tables:
                    if table_data and len(table_data) > 1:
                        table = self._create_table(
                            table_data, page_num, doc_id
                        )
                        if table:
                            tables.append(table)
                
                # Extract figures (as images)
                for img_idx, img in enumerate(page.images):
                    figure = self._create_figure(img, page_num, img_idx, doc_id)
                    if figure:
                        figures.append(figure)
        
        # Create blocks for tables as well
        for table in tables:
            table_block = ContentBlock(
                block_id=f"{doc_id}_table_{len(blocks)}",
                block_type=BlockType.TABLE,
                content=table.to_markdown(),
                bbox=table.bbox,
                metadata={"table_headers": table.headers}
            )
            blocks.append(table_block)
        
        # Calculate confidence
        extracted_text = " ".join([b.content for b in blocks if b.block_type == BlockType.TEXT])
        confidence = self.confidence_scorer.score_fast_text_extraction(
            page_data, extracted_text
        )
        
        extraction_time = time.time() - start_time
        
        return ExtractedDocument(
            doc_id=doc_id,
            filename=pdf_path.split('/')[-1],
            page_count=len(pdf.pages),
            blocks=blocks,
            tables=tables,
            figures=figures,
            extraction_strategy=self.name,
            extraction_timestamp=datetime.now(),
            confidence_score=confidence,
            extraction_time_seconds=extraction_time,
            cost_estimate_usd=self.estimate_cost(pdf_path)
        )
    
    def _words_to_block(
        self,
        words: List[Dict],
        page_num: int,
        doc_id: str
    ) -> Optional[ContentBlock]:
        """Convert word list to content block"""
        if not words:
            return None
        
        # Calculate bounding box
        x0 = min(w['x0'] for w in words)
        y0 = min(w['top'] for w in words)
        x1 = max(w['x1'] for w in words)
        y1 = max(w['bottom'] for w in words)
        
        # Combine text
        text = " ".join(w['text'] for w in words)
        
        # Create block
        return ContentBlock(
            block_id=f"{doc_id}_block_{page_num}_{int(y0)}",
            block_type=BlockType.TEXT,
            content=text,
            bbox=BoundingBox(
                x0=x0, y0=y0, x1=x1, y1=y1,
                page_number=page_num
            ),
            metadata={
                "font_names": list(set(w.get('fontname', '') for w in words)),
                "word_count": len(words)
            }
        )
    
    def _create_table(
        self,
        table_data: List[List],
        page_num: int,
        doc_id: str
    ) -> Optional[Table]:
        """Create table from extracted data"""
        if not table_data or len(table_data) < 2:
            return None
        
        # Assume first row is headers
        headers = [str(cell) if cell else "" for cell in table_data[0]]
        rows = [
            [str(cell) if cell else "" for cell in row]
            for row in table_data[1:]
        ]
        
        # Create bounding box (approximate)
        bbox = BoundingBox(
            x0=50, y0=100, x1=550, y1=500,  # Approximate
            page_number=page_num
        )
        
        return Table(
            headers=headers,
            rows=rows,
            bbox=bbox
        )
    
    def _create_figure(
        self,
        img_data: Dict,
        page_num: int,
        img_idx: int,
        doc_id: str
    ) -> Optional[Figure]:
        """Create figure from image data"""
        bbox = BoundingBox(
            x0=img_data.get('x0', 0),
            y0=img_data.get('top', 0),
            x1=img_data.get('x1', img_data.get('width', 0)),
            y1=img_data.get('bottom', img_data.get('height', 0)),
            page_number=page_num
        )
        
        return Figure(
            bbox=bbox,
            description=f"Figure {img_idx + 1} on page {page_num}"
        )
    
    def estimate_cost(self, pdf_path: str) -> float:
        """Estimate cost based on page count"""
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
        return page_count * self.page_cost