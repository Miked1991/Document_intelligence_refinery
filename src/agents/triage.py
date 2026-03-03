import pdfplumber
from ..models.document_profile import DocumentProfile, OriginType, LayoutComplexity, DomainHint, ExtractionStrategy
from ..utils.config import load_rules
import hashlib

class TriageAgent:
    def __init__(self, rules_path="rubric/extraction_rules.yaml"):
        self.rules = load_rules(rules_path)

    def profile(self, pdf_path: str) -> DocumentProfile:
        doc_id = hashlib.md5(pdf_path.encode()).hexdigest()[:8]
        with pdfplumber.open(pdf_path) as pdf:
            # Origin type detection
            origin = self._detect_origin(pdf)
            # Layout complexity
            layout = self._detect_layout(pdf)
            # Domain hint (simple keyword-based)
            domain = self._detect_domain(pdf)
            # Estimate extraction cost tier
            estimated_tier = self._estimate_tier(origin, layout, pdf)

        return DocumentProfile(
            doc_id=doc_id,
            origin_type=origin,
            layout_complexity=layout,
            domain_hint=domain,
            estimated_cost_tier=estimated_tier
        )

    def _detect_origin(self, pdf) -> OriginType:
        # Simple heuristic: if first page has no text and images, assume scanned
        first_page = pdf.pages[0]
        text = first_page.extract_text()
        images = first_page.images
        if not text and len(images) > 0:
            return OriginType.scanned_image
        if text and len(images) > 0:
            return OriginType.mixed
        return OriginType.native_digital

    def _detect_layout(self, pdf) -> LayoutComplexity:
        # Very basic: check first page for multiple columns using text x-coordinates
        first_page = pdf.pages[0]
        words = first_page.extract_words()
        if not words:
            return LayoutComplexity.mixed
        x_coords = [w['x0'] for w in words]
        # If x_coords cluster into two distinct ranges, likely multi-column
        # Simplified: check variance
        if max(x_coords) - min(x_coords) > 300:  # rough threshold
            return LayoutComplexity.multi_column
        # Check for tables: look for many words aligned in grid
        # Not implemented; default single_column
        return LayoutComplexity.single_column

    def _detect_domain(self, pdf) -> DomainHint:
        # Simple keyword matching on first few pages
        text = ""
        for i, page in enumerate(pdf.pages[:3]):
            text += page.extract_text() or ""
        text_lower = text.lower()
        if any(kw in text_lower for kw in ['revenue', 'profit', 'financial', 'audit']):
            return DomainHint.financial
        if any(kw in text_lower for kw in ['law', 'court', 'plaintiff', 'defendant']):
            return DomainHint.legal
        return DomainHint.general

    def _estimate_tier(self, origin: OriginType, layout: LayoutComplexity, pdf) -> ExtractionStrategy:
        if origin == OriginType.scanned_image:
            return ExtractionStrategy.vision_augmented
        if layout in [LayoutComplexity.multi_column, LayoutComplexity.table_heavy]:
            return ExtractionStrategy.layout_aware
        # Check character count threshold
        total_chars = sum(len(p.extract_text() or '') for p in pdf.pages[:3])
        if total_chars < 100:
            return ExtractionStrategy.vision_augmented
        return ExtractionStrategy.fast_text