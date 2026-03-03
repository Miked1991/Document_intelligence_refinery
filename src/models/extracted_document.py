from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum

class BlockType(str, Enum):
    text = "text"
    table = "table"
    figure = "figure"
    equation = "equation"

class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float
    page: int

class TextBlock(BaseModel):
    bbox: BBox
    text: str
    block_type: BlockType = BlockType.text
    metadata: Dict[str, Any] = {}

class TableCell(BaseModel):
    row: int
    col: int
    text: str
    bbox: Optional[BBox] = None

class Table(TextBlock):
    block_type: BlockType = BlockType.table
    headers: List[str]
    rows: List[List[TableCell]]
    num_rows: int
    num_cols: int

class Figure(TextBlock):
    block_type: BlockType = BlockType.figure
    caption: Optional[str] = None
    image_path: Optional[str] = None

class ExtractedDocument(BaseModel):
    doc_id: str
    pages: List[int]
    blocks: List[TextBlock]   # includes tables and figures
    reading_order: List[int]   # indices into blocks
    metadata: Dict[str, Any] = {}