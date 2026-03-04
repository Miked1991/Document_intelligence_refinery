import subprocess
import json
import hashlib
from .base import BaseExtractor
from ..models.extracted_document import ExtractedDocument, TextBlock, Table, Figure, BBox
from ..utils.adapters import MinerUAdapter

class LayoutExtractor(BaseExtractor):
    def __init__(self, minerU_path: str = "minerU"):
        self.minerU_path = minerU_path

    def extract(self, pdf_path: str) -> ExtractedDocument:
        # Call MinerU CLI to extract to JSON
        result = subprocess.run(
            [self.minerU_path, pdf_path, "--output-format", "json"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"MinerU failed: {result.stderr}")
        data = json.loads(result.stdout)
        # Use adapter to convert MinerU output to ExtractedDocument
        adapter = MinerUAdapter()
        return adapter.convert(data, pdf_path)

    def confidence(self, doc: ExtractedDocument) -> float:
        # Layout extractor generally high confidence if it ran
        return 0.95