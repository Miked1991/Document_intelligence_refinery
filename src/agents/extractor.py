from ..strategies.fast_text import FastTextExtractor
from ..strategies.layout import LayoutExtractor
from ..strategies.vision import VisionExtractor
from ..models.document_profile import DocumentProfile, ExtractionStrategy
from ..models.extracted_document import ExtractedDocument
from ..utils.confidence import compute_confidence
from ..utils.config import load_rules
import json
import datetime

class ExtractionRouter:
    def __init__(self, rules_path="rubric/extraction_rules.yaml", ledger_path=".refinery/extraction_ledger.jsonl"):
        self.rules = load_rules(rules_path)
        self.ledger_path = ledger_path
        self.extractors = {
            ExtractionStrategy.fast_text: FastTextExtractor(),
            ExtractionStrategy.layout_aware: LayoutExtractor(),
            ExtractionStrategy.vision_augmented: VisionExtractor(),
        }

    def extract(self, pdf_path: str, profile: DocumentProfile) -> ExtractedDocument:
        # Start with estimated tier
        current_strategy = profile.estimated_cost_tier
        extractor = self.extractors[current_strategy]
        doc = extractor.extract(pdf_path)
        conf = extractor.confidence(doc)

        # Escalate if confidence below threshold
        threshold = self.rules['confidence'].get('vision_escalation_threshold', 0.7)
        if conf < threshold and current_strategy != ExtractionStrategy.vision_augmented:
            # Try next level: fast_text -> layout -> vision
            if current_strategy == ExtractionStrategy.fast_text:
                current_strategy = ExtractionStrategy.layout_aware
            elif current_strategy == ExtractionStrategy.layout_aware:
                current_strategy = ExtractionStrategy.vision_augmented
            else:
                # Already at vision, can't escalate further
                pass
            if current_strategy != profile.estimated_cost_tier:
                extractor = self.extractors[current_strategy]
                doc = extractor.extract(pdf_path)
                conf = extractor.confidence(doc)

        # Log to ledger
        self._log_entry(pdf_path, profile, current_strategy, conf, doc)

        return doc

    def _log_entry(self, pdf_path, profile, strategy, conf, doc):
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "pdf_path": pdf_path,
            "doc_id": profile.doc_id,
            "strategy_used": strategy.value,
            "confidence_score": conf,
            "cost_estimate": self._estimate_cost(strategy, doc),
            "processing_time": None,  # could add
        }
        with open(self.ledger_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def _estimate_cost(self, strategy: ExtractionStrategy, doc: ExtractedDocument) -> float:
        # Simplified cost model
        if strategy == ExtractionStrategy.fast_text:
            return 0.01
        elif strategy == ExtractionStrategy.layout_aware:
            return 0.10
        else:
            # Vision cost: per page
            return len(doc.pages) * 0.02