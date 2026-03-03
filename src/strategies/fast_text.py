import pdfplumber
from .base import BaseExtractor
from ..models.extracted_document import ExtractedDocument, TextBlock, BBox
from ..models.document_profile import OriginType
import hashlib

class FastTextExtractor(BaseExtractor):
    def extract(self, pdf_path: str) -> ExtractedDocument:
        blocks = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                words = page.extract_words(keep_blank_chars=True, x_tolerance=2)
                # Group words into lines (simple heuristic)
                lines = {}
                for w in words:
                    y = round(w['top'], 1)
                    lines.setdefault(y, []).append(w)
                for y, line_words in sorted(lines.items()):
                    line_text = ' '.join(w['text'] for w in line_words)
                    x0 = min(w['x0'] for w in line_words)
                    x1 = max(w['x1'] for w in line_words)
                    y0 = min(w['top'] for w in line_words)
                    y1 = max(w['bottom'] for w in line_words)
                    block = TextBlock(
                        bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1, page=page_num),
                        text=line_text
                    )
                    blocks.append(block)
        # Simple reading order: top-to-bottom, left-to-right
        blocks.sort(key=lambda b: (b.bbox.page, b.bbox.y0, b.bbox.x0))
        reading_order = list(range(len(blocks)))
        return ExtractedDocument(
            doc_id=hashlib.md5(pdf_path.encode()).hexdigest()[:8],
            pages=list(range(1, len(pdf.pages)+1)),
            blocks=blocks,
            reading_order=reading_order
        )

    def confidence(self, doc: ExtractedDocument) -> float:
        # Use character density and image area ratio (from pdfplumber)
        # This is a simplified example; in reality we'd compute from pdfplumber page stats
        # For now, return a heuristic based on block count and text length
        total_chars = sum(len(b.text) for b in doc.blocks)
        if total_chars < 100:
            return 0.2
        return min(1.0, total_chars / 5000)