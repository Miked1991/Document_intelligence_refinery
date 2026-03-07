"""
Layout-aware extraction strategy using Docling.
Note: Docking in the document refers to Docling (IBM Research)
"""

import os
import tempfile
import subprocess
import json
from typing import Optional, List, Dict, Any
import time
from datetime import datetime
import hashlib

from .base import ExtractionStrategy
from ..models.extracted_document import (
    ExtractedDocument, ContentBlock, BlockType,
    BoundingBox, Table, TableCell, Figure
)
from ..models.document_profile import DocumentProfile
from ..utils.confidence_scorer import ConfidenceScorer


class LayoutExtractor(ExtractionStrategy):
    """
    Layout-aware extraction using Docling.
    Cost: Medium ($0.01 per page)
    """
    
    def __init__(self, docling_path: str = "docling"):
        """
        Initialize layout extractor.
        
        Args:
            docling_path: Path to docling executable
        """
        super().__init__("layout_aware")
        self.docling_path = docling_path
        self.confidence_scorer = ConfidenceScorer()
        self.page_cost = 0.01  # $0.01 per page
    
    async def extract(
        self,
        pdf_path: str,
        profile: Optional[DocumentProfile] = None
    ) -> ExtractedDocument:
        """
        Extract content using Docling.
        """
        start_time = time.time()
        doc_id = hashlib.md5(pdf_path.encode()).hexdigest()[:12]
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = os.path.join(tmpdir, "output.json")
            output_md = os.path.join(tmpdir, "output.md")
            
            # Run Docling
            try:
                cmd = [
                    self.docling_path,
                    pdf_path,
                    "--output", tmpdir,
                    "--json",
                    "--markdown"
                ]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"Docling error: {e.stderr}")
                # Fallback to basic extraction
                return await self._fallback_extraction(pdf_path, doc_id)
            except FileNotFoundError:
                print("Docling not found, using fallback")
                return await self._fallback_extraction(pdf_path, doc_id)
            
            # Parse Docling output
            blocks, tables, figures = self._parse_docling_output(
                output_json, output_md, doc_id
            )
        
        # Calculate confidence
        doc = ExtractedDocument(
            doc_id=doc_id,
            filename=pdf_path.split('/')[-1],
            page_count=self._get_page_count(pdf_path),
            blocks=blocks,
            tables=tables,
            figures=figures,
            extraction_strategy=self.name,
            extraction_timestamp=datetime.now(),
            confidence_score=0.0,  # Will be updated
            extraction_time_seconds=time.time() - start_time,
            cost_estimate_usd=self.estimate_cost(pdf_path)
        )
        
        # Score confidence
        confidence = self.confidence_scorer.score_layout_extraction(doc)
        doc.confidence_score = confidence
        
        return doc
    
    def _parse_docling_output(
        self,
        json_path: str,
        md_path: str,
        doc_id: str
    ) -> tuple:
        """Parse Docling JSON and markdown output"""
        blocks: List[ContentBlock] = []
        tables: List[Table] = []
        figures: List[Figure] = []
        
        # Parse JSON if available
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            
            # Extract document structure
            if "pages" in data:
                for page_data in data["pages"]:
                    page_num = page_data.get("page_number", 1)
                    
                    # Extract text blocks
                    for element in page_data.get("elements", []):
                        block = self._element_to_block(
                            element, page_num, doc_id
                        )
                        if block:
                            blocks.append(block)
                            
                            # Extract tables
                            if element.get("type") == "table":
                                table = self._element_to_table(
                                    element, page_num, doc_id
                                )
                                if table:
                                    tables.append(table)
                            
                            # Extract figures
                            elif element.get("type") == "figure":
                                figure = self._element_to_figure(
                                    element, page_num, doc_id
                                )
                                if figure:
                                    figures.append(figure)
        
        # Parse markdown for additional content
        if os.path.exists(md_path) and not blocks:
            blocks = self._parse_markdown(md_path, doc_id)
        
        return blocks, tables, figures
    
    def _element_to_block(
        self,
        element: Dict,
        page_num: int,
        doc_id: str
    ) -> Optional[ContentBlock]:
        """Convert Docling element to content block"""
        element_type = element.get("type", "text")
        text = element.get("text", "")
        
        if not text:
            return None
        
        # Map Docling types to BlockType
        type_map = {
            "paragraph": BlockType.TEXT,
            "heading": BlockType.HEADER,
            "list": BlockType.LIST,
            "table": BlockType.TABLE,
            "figure": BlockType.FIGURE,
            "caption": BlockType.CAPTION,
            "equation": BlockType.EQUATION,
            "footer": BlockType.FOOTER,
            "header": BlockType.HEADER
        }
        
        block_type = type_map.get(element_type, BlockType.TEXT)
        
        # Extract bounding box
        bbox = None
        if "bbox" in element:
            bbox_data = element["bbox"]
            bbox = BoundingBox(
                x0=bbox_data.get("x0", 0),
                y0=bbox_data.get("y0", 0),
                x1=bbox_data.get("x1", bbox_data.get("width", 0)),
                y1=bbox_data.get("y1", bbox_data.get("height", 0)),
                page_number=page_num
            )
        
        return ContentBlock(
            block_id=f"{doc_id}_{element_type}_{page_num}_{hash(text[:50])}",
            block_type=block_type,
            content=text,
            bbox=bbox,
            metadata=element.get("metadata", {})
        )
    
    def _element_to_table(
        self,
        element: Dict,
        page_num: int,
        doc_id: str
    ) -> Optional[Table]:
        """Convert Docling table element to Table model"""
        if "table_data" not in element:
            return None
        
        table_data = element["table_data"]
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])
        
        # Create bounding box
        bbox = None
        if "bbox" in element:
            bbox_data = element["bbox"]
            bbox = BoundingBox(
                x0=bbox_data.get("x0", 0),
                y0=bbox_data.get("y0", 0),
                x1=bbox_data.get("x1", bbox_data.get("width", 0)),
                y1=bbox_data.get("y1", bbox_data.get("height", 0)),
                page_number=page_num
            )
        else:
            bbox = BoundingBox(x0=0, y0=0, x1=0, y1=0, page_number=page_num)
        
        # Create cells
        cells = []
        for r_idx, row in enumerate(rows):
            for c_idx, cell in enumerate(row):
                if c_idx < len(headers):
                    cells.append(TableCell(
                        content=str(cell),
                        row_index=r_idx,
                        col_index=c_idx,
                        is_header=False
                    ))
        
        return Table(
            headers=headers,
            rows=[[str(cell) for cell in row] for row in rows],
            caption=element.get("caption"),
            bbox=bbox,
            cells=cells
        )
    
    def _element_to_figure(
        self,
        element: Dict,
        page_num: int,
        doc_id: str
    ) -> Optional[Figure]:
        """Convert Docling figure element to Figure model"""
        # Create bounding box
        bbox = None
        if "bbox" in element:
            bbox_data = element["bbox"]
            bbox = BoundingBox(
                x0=bbox_data.get("x0", 0),
                y0=bbox_data.get("y0", 0),
                x1=bbox_data.get("x1", bbox_data.get("width", 0)),
                y1=bbox_data.get("y1", bbox_data.get("height", 0)),
                page_number=page_num
            )
        else:
            bbox = BoundingBox(x0=0, y0=0, x1=0, y1=0, page_number=page_num)
        
        return Figure(
            caption=element.get("caption"),
            bbox=bbox,
            description=element.get("text", "")
        )
    
    def _parse_markdown(self, md_path: str, doc_id: str) -> List[ContentBlock]:
        """Fallback: parse markdown output"""
        blocks = []
        
        with open(md_path) as f:
            content = f.read()
        
        # Split by headers
        import re
        sections = re.split(r'(^#+ .+$)', content, flags=re.MULTILINE)
        
        current_section = "Root"
        for i, section in enumerate(sections):
            if section.startswith('#'):
                current_section = section.strip('# ')
            elif section.strip():
                block = ContentBlock(
                    block_id=f"{doc_id}_md_{i}",
                    block_type=BlockType.TEXT,
                    content=section.strip(),
                    bbox=None,  # No bbox from markdown
                    metadata={"section": current_section}
                )
                blocks.append(block)
        
        return blocks
    
    async def _fallback_extraction(
        self,
        pdf_path: str,
        doc_id: str
    ) -> ExtractedDocument:
        """Fallback to fast text if Docling fails"""
        from .fast_text import FastTextExtractor
        extractor = FastTextExtractor()
        return await extractor.extract(pdf_path)
    
    def _get_page_count(self, pdf_path: str) -> int:
        """Get page count using pdfplumber"""
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    
    def estimate_cost(self, pdf_path: str) -> float:
        """Estimate cost based on page count"""
        return self._get_page_count(pdf_path) * self.page_cost 