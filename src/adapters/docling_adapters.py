"""
Adapter for Docling (IBM Research) document understanding library.
Converts Docling output to the internal ExtractedDocument schema.
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..models.extracted_document import (
    ExtractedDocument, ContentBlock, BlockType,
    BoundingBox, Table, TableCell, Figure
)


class DoclingAdapter:
    """
    Adapter for converting Docling output to internal schema.
    """
    
    @classmethod
    def from_json(cls, json_path: str, doc_id: str, filename: str) -> ExtractedDocument:
        """
        Create ExtractedDocument from Docling JSON output.
        
        Args:
            json_path: Path to Docling JSON file
            doc_id: Document ID
            filename: Original filename
            
        Returns:
            ExtractedDocument in internal schema
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        blocks = []
        tables = []
        figures = []
        
        # Process each page
        for page_data in data.get('pages', []):
            page_num = page_data.get('page_number', 1)
            
            # Process elements
            for element in page_data.get('elements', []):
                block = cls._element_to_block(element, page_num, doc_id)
                if block:
                    blocks.append(block)
                
                # Extract tables
                if element.get('type') == 'table':
                    table = cls._element_to_table(element, page_num, doc_id)
                    if table:
                        tables.append(table)
                
                # Extract figures
                elif element.get('type') == 'figure':
                    figure = cls._element_to_figure(element, page_num, doc_id)
                    if figure:
                        figures.append(figure)
        
        return ExtractedDocument(
            doc_id=doc_id,
            filename=filename,
            page_count=len(data.get('pages', [])),
            blocks=blocks,
            tables=tables,
            figures=figures,
            extraction_strategy='layout_aware',
            confidence_score=0.95,  # Docling is generally high quality
            extraction_time_seconds=data.get('processing_time', 0),
            cost_estimate_usd=0.01 * len(data.get('pages', []))  # $0.01 per page
        )
    
    @classmethod
    def _element_to_block(cls, element: Dict, page_num: int, doc_id: str) -> Optional[ContentBlock]:
        """Convert Docling element to ContentBlock"""
        element_type = element.get('type', 'text')
        text = element.get('text', '')
        
        if not text:
            return None
        
        # Map Docling types to BlockType
        type_map = {
            'paragraph': BlockType.TEXT,
            'heading': BlockType.HEADER,
            'title': BlockType.HEADER,
            'list': BlockType.LIST,
            'table': BlockType.TABLE,
            'figure': BlockType.FIGURE,
            'caption': BlockType.CAPTION,
            'equation': BlockType.EQUATION,
            'footer': BlockType.FOOTER,
            'header': BlockType.HEADER,
            'footnote': BlockType.FOOTNOTE
        }
        
        block_type = type_map.get(element_type, BlockType.TEXT)
        
        # Extract bounding box
        bbox = None
        if 'bbox' in element:
            bbox_data = element['bbox']
            bbox = BoundingBox(
                x0=bbox_data.get('x0', 0),
                y0=bbox_data.get('y0', 0),
                x1=bbox_data.get('x1', bbox_data.get('width', 0)),
                y1=bbox_data.get('y1', bbox_data.get('height', 0)),
                page_number=page_num
            )
        
        return ContentBlock(
            block_id=f"{doc_id}_docling_{element_type}_{page_num}_{hash(text[:50])}",
            block_type=block_type,
            content=text,
            bbox=bbox,
            metadata=element.get('metadata', {})
        )
    
    @classmethod
    def _element_to_table(cls, element: Dict, page_num: int, doc_id: str) -> Optional[Table]:
        """Convert Docling table element to Table model"""
        if 'table_data' not in element:
            return None
        
        table_data = element['table_data']
        headers = table_data.get('headers', [])
        rows = table_data.get('rows', [])
        
        # Create bounding box
        bbox = None
        if 'bbox' in element:
            bbox_data = element['bbox']
            bbox = BoundingBox(
                x0=bbox_data.get('x0', 0),
                y0=bbox_data.get('y0', 0),
                x1=bbox_data.get('x1', bbox_data.get('width', 0)),
                y1=bbox_data.get('y1', bbox_data.get('height', 0)),
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
            caption=element.get('caption'),
            bbox=bbox,
            cells=cells
        )
    
    @classmethod
    def _element_to_figure(cls, element: Dict, page_num: int, doc_id: str) -> Optional[Figure]:
        """Convert Docling figure element to Figure model"""
        # Create bounding box
        bbox = None
        if 'bbox' in element:
            bbox_data = element['bbox']
            bbox = BoundingBox(
                x0=bbox_data.get('x0', 0),
                y0=bbox_data.get('y0', 0),
                x1=bbox_data.get('x1', bbox_data.get('width', 0)),
                y1=bbox_data.get('y1', bbox_data.get('height', 0)),
                page_number=page_num
            )
        else:
            bbox = BoundingBox(x0=0, y0=0, x1=0, y1=0, page_number=page_num)
        
        return Figure(
            caption=element.get('caption'),
            bbox=bbox,
            description=element.get('text', ''),
            image_path=element.get('image_path')
        )