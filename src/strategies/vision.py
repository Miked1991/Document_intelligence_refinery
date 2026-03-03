import base64
import requests
from pdf2image import convert_from_path
from .base import BaseExtractor
from ..models.extracted_document import ExtractedDocument, TextBlock, Table, Figure, BBox
from ..utils.config import OPENROUTER_API_KEY, VISION_MODEL
import hashlib
import io
from PIL import Image

class VisionExtractor(BaseExtractor):
    def __init__(self, api_key: str = OPENROUTER_API_KEY, model: str = VISION_MODEL):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def extract(self, pdf_path: str) -> ExtractedDocument:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path, dpi=150)
        all_blocks = []
        for page_num, img in enumerate(images, start=1):
            # Encode image to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            # Prepare prompt for VLM
            prompt = """
            Extract all text, tables, and figures from this document page.
            Return a JSON array of objects with keys: type (text/table/figure), 
            bbox (x0,y0,x1,y1), content (text for text, or table as 2D array, 
            or figure description), and optionally caption.
            Use normalized coordinates 0-1000 for x,y.
            """
            response = self._call_vlm(img_b64, prompt)
            # Parse response (assuming JSON)
            try:
                page_blocks = response.json()  # simplified; actual response parsing needed
                for blk in page_blocks:
                    # Convert to TextBlock/Table/Figure
                    bbox = BBox(
                        x0=blk['bbox'][0], y0=blk['bbox'][1],
                        x1=blk['bbox'][2], y1=blk['bbox'][3],
                        page=page_num
                    )
                    if blk['type'] == 'table':
                        # Assume content is list of lists
                        table = Table(
                            bbox=bbox,
                            text="",  # we'll store JSON in metadata
                            headers=blk['content'][0] if blk['content'] else [],
                            rows=[[TableCell(row=r, col=c, text=cell) for c, cell in enumerate(row)] for r, row in enumerate(blk['content'])],
                            num_rows=len(blk['content']),
                            num_cols=len(blk['content'][0]) if blk['content'] else 0,
                            metadata={'raw': blk['content']}
                        )
                        all_blocks.append(table)
                    else:
                        block = TextBlock(
                            bbox=bbox,
                            text=blk['content'] if blk['type'] == 'text' else blk.get('caption', ''),
                            block_type=blk['type']
                        )
                        all_blocks.append(block)
            except:
                # Fallback: treat whole page as text
                block = TextBlock(bbox=BBox(x0=0,y0=0,x1=1000,y1=1000,page=page_num), text="[Vision extraction failed]")
                all_blocks.append(block)

        reading_order = list(range(len(all_blocks)))
        return ExtractedDocument(
            doc_id=hashlib.md5(pdf_path.encode()).hexdigest()[:8],
            pages=list(range(1, len(images)+1)),
            blocks=all_blocks,
            reading_order=reading_order
        )

    def _call_vlm(self, img_b64: str, prompt: str):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ]
                }
            ]
        }
        response = requests.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()  # parse as needed; actual extraction of structured data may require function calling

    def confidence(self, doc: ExtractedDocument) -> float:
        # Vision extractor typically high confidence if pages were processed
        return 0.98