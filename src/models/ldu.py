from pydantic import BaseModel
from typing import List, Optional
from .provenance import ProvenanceRef

class LDU(BaseModel):
    ldu_id: str
    content: str
    chunk_type: str          # e.g., "paragraph", "table", "list", "section_header"
    page_refs: List[int]
    bbox: Optional[tuple]    # (x0,y0,x1,y1) on first page
    parent_section: Optional[str]  # section title or ID
    token_count: int
    content_hash: str        # for provenance
    metadata: dict = {}
    provenance: List[ProvenanceRef] = []