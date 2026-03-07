"""
Extraction Router with confidence-gated escalation.
"""

from typing import Optional, Dict, Any,List
import json
from pathlib import Path
from datetime import datetime
import asyncio

from ..models.document_profile import DocumentProfile, ExtractionStrategy
from ..models.extracted_document import ExtractedDocument
from ..models.provenancechain import AuditEntry
from ..strategies.fast_text import FastTextExtractor
from ..strategies.layout import LayoutExtractor
from ..strategies.vision import VisionExtractor
from ..utils.confidence_scorer import ConfidenceScorer
from ..utils.budget_guard import BudgetGuard


class ExtractionRouter:
    """
    Routes extraction to appropriate strategy with confidence-gated escalation.
    """
    
    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        ledger_path: str = ".refinery/extraction_ledger.jsonl",
        budget_config: Optional[Dict] = None
    ):
        """
        Initialize extraction router.
        
        Args:
            openrouter_api_key: API key for OpenRouter (for vision strategy)
            ledger_path: Path to extraction ledger
            budget_config: Budget configuration
        """
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize strategies
        self.strategies = {
            "fast_text": FastTextExtractor(),
            "layout_aware": LayoutExtractor()
        }
        
        if openrouter_api_key:
            self.strategies["vision_augmented"] = VisionExtractor(openrouter_api_key)
        
        self.confidence_scorer = ConfidenceScorer()
        self.budget_guard = BudgetGuard(**(budget_config or {}))
    
    async def extract(
        self,
        pdf_path: str,
        profile: Optional[DocumentProfile] = None,
        max_escalations: int = 2
    ) -> ExtractedDocument:
        """
        Extract document with automatic escalation on low confidence.
        
        Args:
            pdf_path: Path to PDF file
            profile: Optional document profile (will be generated if not provided)
            max_escalations: Maximum number of escalation steps
            
        Returns:
            Extracted document
        """
        # Get or create profile
        if not profile:
            from .triage import TriageAgent
            triage = TriageAgent()
            profile = await triage.profile_document(pdf_path)
        
        # Start with recommended strategy
        current_strategy = profile.recommended_strategy.value
        
        # Track extraction attempts
        attempts = []
        final_doc = None
        
        for escalation_level in range(max_escalations + 1):
            # Check budget before extraction
            budget_check = self.budget_guard.check_budget(
                current_strategy,
                profile.page_count
            )
            
            if not budget_check["approved"]:
                # Try downgrading if possible
                if current_strategy != "fast_text":
                    current_strategy = "fast_text"
                    continue
                else:
                    raise Exception(f"Budget exceeded: {budget_check['warnings']}")
            
            # Get extractor
            extractor = self.strategies.get(current_strategy)
            if not extractor:
                raise ValueError(f"Unknown strategy: {current_strategy}")
            
            # Perform extraction
            print(f"Attempting extraction with {current_strategy} strategy...")
            doc = await extractor.extract(pdf_path, profile)
            
            # Log attempt
            attempts.append({
                "strategy": current_strategy,
                "confidence": doc.confidence_score,
                "cost": doc.cost_estimate_usd
            })
            
            # Check if we need to escalate
            should_escalate = self.confidence_scorer.should_escalate(
                doc.confidence_score,
                current_strategy,
                profile
            )
            
            if not should_escalate or escalation_level == max_escalations:
                final_doc = doc
                break
            
            # Escalate to next strategy
            next_strategy = self._get_next_strategy(current_strategy)
            if not next_strategy or next_strategy == current_strategy:
                final_doc = doc
                break
            
            print(f"Confidence {doc.confidence_score:.2f} below threshold, escalating to {next_strategy}")
            current_strategy = next_strategy
        
        # Record in ledger
        await self._record_extraction(pdf_path, profile, final_doc, attempts)
        
        return final_doc
    
    def _get_next_strategy(self, current: str) -> Optional[str]:
        """Get next strategy in escalation chain"""
        escalation_path = {
            "fast_text": "layout_aware",
            "layout_aware": "vision_augmented",
            "vision_augmented": None
        }
        return escalation_path.get(current)
    
    async def _record_extraction(
        self,
        pdf_path: str,
        profile: DocumentProfile,
        doc: ExtractedDocument,
        attempts: List[Dict]
    ):
        """Record extraction in ledger"""
        entry = AuditEntry(
            doc_id=profile.doc_id,
            timestamp=datetime.now(),
            strategy_used=doc.extraction_strategy,
            confidence_score=doc.confidence_score,
            cost_estimate_usd=doc.cost_estimate_usd,
            processing_time_seconds=doc.extraction_time_seconds,
            page_count=profile.page_count,
            metadata={
                "filename": profile.filename,
                "attempts": attempts,
                "origin_type": profile.origin_type.value,
                "layout_complexity": profile.layout_complexity.value
            }
        )
        
        # Append to ledger
        with open(self.ledger_path, 'a') as f:
            f.write(entry.model_dump_json() + '\n')
    
    async def batch_extract(
        self,
        pdf_paths: List[str],
        profiles: Optional[Dict[str, DocumentProfile]] = None
    ) -> Dict[str, ExtractedDocument]:
        """
        Extract multiple documents.
        
        Args:
            pdf_paths: List of PDF paths
            profiles: Optional dict mapping path to profile
            
        Returns:
            Dict mapping path to extracted document
        """
        results = {}
        
        for pdf_path in pdf_paths:
            profile = profiles.get(pdf_path) if profiles else None
            try:
                doc = await self.extract(pdf_path, profile)
                results[pdf_path] = doc
            except Exception as e:
                print(f"Error extracting {pdf_path}: {e}")
                results[pdf_path] = None
        
        return results