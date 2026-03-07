"""
Confidence scoring utilities for extraction quality assessment.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from ..models.extracted_document import ExtractedDocument, Table
from ..models.document_profile import DocumentProfile


class ConfidenceScorer:
    """
    Multi-signal confidence scorer for extraction quality.
    """
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize confidence scorer with configurable thresholds.
        
        Args:
            thresholds: Dictionary of threshold values for different signals
        """
        self.thresholds = thresholds or {
            "character_density_min": 100,  # characters per page
            "character_density_ratio": 0.7,  # ratio of expected density
            "table_completeness": 0.8,  # minimum table cell coverage
            "layout_preservation": 0.75,  # layout structure preservation
            "text_extraction_ratio": 0.9,  # ratio of text successfully extracted
        }
    
    def score_fast_text_extraction(
        self,
        pdf_page_data: List[Dict[str, Any]],
        extracted_text: str
    ) -> float:
        """
        Score confidence of fast text extraction.
        
        Args:
            pdf_page_data: List of page data from pdfplumber
            extracted_text: Extracted text content
            
        Returns:
            Confidence score between 0 and 1
        """
        signals = []
        
        # Signal 1: Character density
        total_chars = len(extracted_text)
        total_pages = len(pdf_page_data)
        avg_chars_per_page = total_chars / total_pages if total_pages > 0 else 0
        
        char_density_score = min(1.0, avg_chars_per_page / self.thresholds["character_density_min"])
        signals.append(char_density_score)
        
        # Signal 2: Image-to-page ratio (high image ratio suggests scanned document)
        image_ratios = []
        for page in pdf_page_data:
            if 'images' in page and page['images']:
                page_area = page.get('width', 612) * page.get('height', 792)  # Default letter size
                image_area = sum(img.get('width', 0) * img.get('height', 0) for img in page['images'])
                image_ratios.append(min(1.0, image_area / page_area if page_area > 0 else 0))
        
        if image_ratios:
            avg_image_ratio = np.mean(image_ratios)
            # Lower image ratio is better for text extraction
            image_score = 1.0 - min(1.0, avg_image_ratio)
            signals.append(image_score)
        
        # Signal 3: Font metadata presence
        font_present = any(
            page.get('chars') and any('fontname' in char for char in page['chars'])
            for page in pdf_page_data
        )
        signals.append(1.0 if font_present else 0.3)
        
        # Signal 4: Text extraction consistency
        if total_pages > 0 and pdf_page_data:
            expected_chars_per_page = []
            for page in pdf_page_data:
                if 'chars' in page:
                    expected_chars_per_page.append(len(page['chars']))
            
            if expected_chars_per_page:
                expected_avg = np.mean(expected_chars_per_page)
                if expected_avg > 0:
                    consistency_score = min(1.0, avg_chars_per_page / expected_avg)
                    signals.append(consistency_score)
        
        # Weighted average (simple mean for now)
        confidence = float(np.mean(signals)) if signals else 0.5
        
        return min(1.0, max(0.0, confidence))
    
    def score_table_extraction(self, table: Table, original_text_hint: str = "") -> float:
        """
        Score confidence of table extraction.
        
        Args:
            table: Extracted table
            original_text_hint: Optional original text for comparison
            
        Returns:
            Confidence score between 0 and 1
        """
        signals = []
        
        # Signal 1: Table structure completeness
        if table.rows and table.headers:
            # Check if all rows have the same number of columns as headers
            row_completeness = sum(
                1 for row in table.rows if len(row) == len(table.headers)
            ) / len(table.rows) if table.rows else 0
            signals.append(row_completeness)
        
        # Signal 2: Cell content quality
        if table.cells:
            non_empty_cells = sum(1 for cell in table.cells if cell.content.strip())
            cell_coverage = non_empty_cells / len(table.cells) if table.cells else 0
            signals.append(cell_coverage)
        elif table.rows:
            # Estimate from rows
            total_cells = len(table.rows) * len(table.headers) if table.headers else 0
            filled_cells = sum(1 for row in table.rows for cell in row if cell.strip())
            cell_coverage = filled_cells / total_cells if total_cells > 0 else 0
            signals.append(cell_coverage)
        
        # Signal 3: Caption presence (good indicator of proper extraction)
        signals.append(1.0 if table.caption else 0.5)
        
        # Signal 4: Numerical consistency (for financial tables)
        if table.headers and any(h in str(table.headers).lower() for h in ['revenue', 'amount', 'total']):
            # Check if numbers are present and formatted consistently
            number_patterns = []
            for row in table.rows:
                for cell in row:
                    if any(c.isdigit() for c in cell):
                        number_patterns.append(1)
                        break
            
            if number_patterns:
                numerical_presence = len(number_patterns) / len(table.rows)
                signals.append(numerical_presence)
        
        confidence = float(np.mean(signals)) if signals else 0.7
        
        return min(1.0, max(0.0, confidence))
    
    def score_layout_extraction(self, doc: ExtractedDocument) -> float:
        """
        Score confidence of layout-aware extraction.
        
        Args:
            doc: Extracted document
            
        Returns:
            Confidence score between 0 and 1
        """
        signals = []
        
        # Signal 1: Block structure
        if doc.blocks:
            # Check if blocks have proper bounding boxes
            blocks_with_bbox = sum(1 for block in doc.blocks if block.bbox)
            bbox_coverage = blocks_with_bbox / len(doc.blocks)
            signals.append(bbox_coverage)
        
        # Signal 2: Table extraction quality
        if doc.tables:
            table_scores = [self.score_table_extraction(table) for table in doc.tables]
            signals.append(float(np.mean(table_scores)))
        
        # Signal 3: Reading order preservation
        if len(doc.blocks) > 1:
            # Check if blocks are ordered by page and y-coordinate
            ordered_pairs = 0
            total_pairs = len(doc.blocks) - 1
            
            for i in range(total_pairs):
                curr = doc.blocks[i]
                next_block = doc.blocks[i + 1]
                
                if curr.bbox and next_block.bbox:
                    if curr.bbox.page_number < next_block.bbox.page_number:
                        ordered_pairs += 1
                    elif (curr.bbox.page_number == next_block.bbox.page_number and
                          curr.bbox.y0 <= next_block.bbox.y0):
                        ordered_pairs += 1
            
            reading_order_score = ordered_pairs / total_pairs if total_pairs > 0 else 1.0
            signals.append(reading_order_score)
        
        # Signal 4: Figure/caption association
        if doc.figures:
            figures_with_captions = sum(1 for fig in doc.figures if fig.caption)
            caption_coverage = figures_with_captions / len(doc.figures)
            signals.append(caption_coverage)
        
        confidence = float(np.mean(signals)) if signals else 0.8
        
        return min(1.0, max(0.0, confidence))
    
    def should_escalate(
        self,
        confidence: float,
        strategy: str,
        profile: DocumentProfile
    ) -> bool:
        """
        Determine if extraction should escalate to next strategy.
        
        Args:
            confidence: Current extraction confidence
            strategy: Current strategy used
            profile: Document profile
            
        Returns:
            True if should escalate to next strategy
        """
        thresholds = {
            "fast_text": 0.6,
            "layout_aware": 0.7,
            "vision_augmented": 0.8  # No escalation beyond vision
        }
        
        threshold = thresholds.get(strategy, 0.7)
        
        # Adjust threshold based on document complexity
        if profile.layout_complexity in ["multi_column", "table_heavy"]:
            threshold += 0.1
        if profile.origin_type == "scanned_image":
            threshold += 0.2
        
        return confidence < threshold