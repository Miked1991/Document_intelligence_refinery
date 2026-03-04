"""Production-grade extraction router with multi-level confidence gating and escalation."""

from typing import Dict, List, Optional, Tuple, Any, Type
from pathlib import Path
from datetime import datetime
import asyncio
import logging
from pydantic import ValidationError, BaseModel
import yaml
from enum import Enum

from src.models.document_profile import DocumentProfile, ExtractionCostTier
from src.models.extracted_document import ExtractedDocument
from src.models.provenance import ContentHash, ProvenanceItem
from src.strategies.base import ExtractionStrategy
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout import LayoutExtractor
from src.strategies.vision import VisionExtractor
from src.utils.confidence import MultiSignalConfidenceScorer, SignalResult
from src.utils.budget import BudgetGuard, UsageRecord
from src.utils.ledger import ExtractionLedger


class EscalationLevel(str, Enum):
    """Escalation levels with clear semantics."""
    LEVEL_1_FAST = "fast_text"
    LEVEL_2_LAYOUT = "layout_aware"
    LEVEL_3_VISION = "vision_augmented"
    LEVEL_4_HUMAN = "human_review"  # Ultimate fallback


class EscalationResult(BaseModel):
    """Result of escalation process with full context."""
    document_id: str
    final_strategy: EscalationLevel
    confidence: float
    confidence_breakdown: Dict[str, float]
    extraction_result: ExtractedDocument
    escalation_chain: List[Dict[str, Any]]
    warnings: List[str] = []
    errors: List[str] = []
    processing_time_ms: int
    cost: float
    timestamp: datetime = datetime.utcnow()
    
    class Config:
        arbitrary_types_allowed = True


class EscalationRule(BaseModel):
    """Rule for escalation decisions."""
    level: EscalationLevel
    min_confidence: float
    max_retries: int = 1
    timeout_seconds: int = 300
    cost_threshold: Optional[float] = None
    fallback_to: Optional[EscalationLevel] = None
    
    @validator('min_confidence')
    def confidence_in_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError(f'min_confidence must be between 0 and 1, got {v}')
        return v


class EscalationConfig(BaseModel):
    """Complete escalation configuration."""
    rules: Dict[EscalationLevel, EscalationRule]
    global_timeout: int = 3600
    max_escalation_depth: int = 3
    parallel_attempts: bool = False
    cache_results: bool = True
    cache_ttl_seconds: int = 3600
    
    # Confidence thresholds for different actions
    thresholds: Dict[str, float] = {
        "acceptable": 0.7,
        "good": 0.8,
        "excellent": 0.9,
        "escalate": 0.6,
        "critical_failure": 0.4
    }
    
    # Strategy-specific overrides
    strategy_overrides: Dict[EscalationLevel, Dict[str, Any]] = {
        EscalationLevel.LEVEL_1_FAST: {
            "max_pages": 500,
            "min_character_density": 0.0003
        },
        EscalationLevel.LEVEL_2_LAYOUT: {
            "gpu_required": False,
            "batch_size": 4
        },
        EscalationLevel.LEVEL_3_VISION: {
            "models": ["google/gemini-flash-1.5", "openai/gpt-4o-mini"],
            "max_retries_per_page": 2
        }
    }


class ExtractionRouter:
    """
    Production-grade extraction router with multi-level confidence-gated escalation.
    
    Features:
    - Multi-level escalation (fast → layout → vision → human)
    - Confidence scoring with statistical validation
    - Budget enforcement at each level
    - Parallel strategy execution option
    - Caching of results
    - Comprehensive logging and telemetry
    """
    
    def __init__(self, 
                 config_path: Path,
                 ledger: ExtractionLedger,
                 budget_guard: BudgetGuard,
                 confidence_scorer: MultiSignalConfidenceScorer,
                 cache_dir: Optional[Path] = None):
        
        self.logger = logging.getLogger(__name__)
        self.ledger = ledger
        self.budget_guard = budget_guard
        self.confidence_scorer = confidence_scorer
        
        # Load configuration
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            self.config = EscalationConfig(**config_dict['escalation'])
        
        # Initialize strategies
        self.strategies = self._init_strategies(config_dict)
        
        # Setup caching
        self.cache_dir = cache_dir or Path(".refinery/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, Tuple[EscalationResult, datetime]] = {}
        
        # Metrics and telemetry
        self.metrics = {
            "total_processed": 0,
            "escalation_counts": {level: 0 for level in EscalationLevel},
            "average_confidence": 0.0,
            "total_cost": 0.0,
            "average_processing_time": 0.0
        }
    
    def _init_strategies(self, config: Dict) -> Dict[EscalationLevel, ExtractionStrategy]:
        """Initialize extraction strategies with configuration."""
        strategies = {}
        
        # Level 1: Fast Text
        if config['extraction']['fast_text']['enabled']:
            strategies[EscalationLevel.LEVEL_1_FAST] = FastTextExtractor(
                config=config['extraction']['fast_text']['config']
            )
        
        # Level 2: Layout-Aware
        if config['extraction']['layout']['enabled']:
            strategies[EscalationLevel.LEVEL_2_LAYOUT] = LayoutExtractor(
                config=config['extraction']['layout']['mineru_config']
            )
        
        # Level 3: Vision-Augmented
        if config['extraction']['vision']['enabled']:
            strategies[EscalationLevel.LEVEL_3_VISION] = VisionExtractor(
                api_key=config.get('openrouter_api_key'),
                config=config['extraction']['vision']
            )
        
        return strategies
    
    async def extract_with_escalation(self,
                                     pdf_path: Path,
                                     profile: DocumentProfile,
                                     force_strategy: Optional[EscalationLevel] = None) -> EscalationResult:
        """
        Extract document with full multi-level confidence-gated escalation.
        
        Args:
            pdf_path: Path to PDF document
            profile: Document profile from triage agent
            force_strategy: Optional forced strategy (bypasses escalation)
            
        Returns:
            EscalationResult with full provenance and metrics
        """
        start_time = datetime.utcnow()
        document_id = profile.doc_id
        
        self.logger.info(f"Starting extraction for {document_id} with profile {profile.origin_type}")
        
        # Check cache
        if self.config.cache_results and document_id in self.cache:
            cached_result, timestamp = self.cache[document_id]
            age = (datetime.utcnow() - timestamp).seconds
            if age < self.config.cache_ttl_seconds:
                self.logger.info(f"Returning cached result for {document_id}")
                return cached_result
        
        # Check budget before processing
        can_process, budget_message = self.budget_guard.can_process(
            document_id=document_id,
            pages=profile.num_pages,
            strategy=profile.estimated_extraction_cost.value
        )
        
        if not can_process:
            raise BudgetExceededError(f"Budget check failed: {budget_message}")
        
        # Determine initial strategy
        if force_strategy:
            current_level = force_strategy
        else:
            current_level = self._map_profile_to_level(profile)
        
        escalation_chain = []
        final_result = None
        warnings = []
        errors = []
        
        # Multi-level escalation loop
        for attempt in range(self.config.max_escalation_depth):
            self.logger.info(f"Escalation attempt {attempt + 1}: {current_level.value}")
            
            # Get strategy and rule
            strategy = self.strategies.get(current_level)
            if not strategy:
                error_msg = f"No strategy available for level {current_level.value}"
                errors.append(error_msg)
                self.logger.error(error_msg)
                break
            
            rule = self.config.rules[current_level]
            
            try:
                # Apply timeout
                async with asyncio.timeout(rule.timeout_seconds):
                    # Execute extraction
                    extraction_start = datetime.utcnow()
                    extracted_doc = await strategy.extract_async(pdf_path)
                    extraction_time = (datetime.utcnow() - extraction_start).total_seconds() * 1000
                    
                    # Score confidence
                    confidence_result = self.confidence_scorer.score_document(
                        pdf_path, 
                        extracted_doc.dict()
                    )
                    
                    final_confidence = confidence_result['final_confidence']
                    
                    # Create escalation record
                    escalation_record = {
                        "level": current_level.value,
                        "confidence": final_confidence,
                        "processing_time_ms": extraction_time,
                        "strategy_config": rule.dict(),
                        "signal_breakdown": confidence_result['signal_details']
                    }
                    escalation_chain.append(escalation_record)
                    
                    # Check if confidence is acceptable
                    if final_confidence >= rule.min_confidence:
                        self.logger.info(f"Acceptable confidence {final_confidence:.3f} at level {current_level.value}")
                        final_result = extracted_doc
                        break
                    else:
                        warning = f"Low confidence {final_confidence:.3f} at level {current_level.value}, escalating"
                        warnings.append(warning)
                        self.logger.warning(warning)
                        
                        # Check if we should escalate further
                        if final_confidence < self.config.thresholds['escalate']:
                            next_level = self._get_next_level(current_level)
                            if next_level and next_level != current_level:
                                current_level = next_level
                                continue
                            else:
                                # No further escalation possible
                                if final_result is None:
                                    final_result = extracted_doc  # Use best available
                                break
                        else:
                            # Confidence low but not critical, use current result
                            final_result = extracted_doc
                            break
                            
            except asyncio.TimeoutError:
                error = f"Timeout after {rule.timeout_seconds}s at level {current_level.value}"
                errors.append(error)
                self.logger.error(error)
                
                # Try next level on timeout
                next_level = self._get_next_level(current_level)
                if next_level:
                    current_level = next_level
                    continue
                else:
                    break
                    
            except Exception as e:
                error = f"Extraction failed at level {current_level.value}: {str(e)}"
                errors.append(error)
                self.logger.exception(error)
                
                # Try next level on error
                next_level = self._get_next_level(current_level)
                if next_level:
                    current_level = next_level
                    continue
                else:
                    break
        
        # If we never got a result, raise error
        if final_result is None:
            raise ExtractionFailedError(
                f"All extraction strategies failed for {document_id}. Errors: {errors}"
            )
        
        # Calculate final confidence (weighted average across attempts)
        final_confidence = self._calculate_weighted_confidence(escalation_chain)
        
        # Record usage in budget
        usage_record = self.budget_guard.record_usage(
            document_id=document_id,
            strategy=current_level.value,
            pages_processed=profile.num_pages,
            model=self._get_model_for_level(current_level),
            actual_cost=self._estimate_cost(escalation_chain, profile)
        )
        
        # Create result
        processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        result = EscalationResult(
            document_id=document_id,
            final_strategy=current_level,
            confidence=final_confidence,
            confidence_breakdown=self._aggregate_confidence_breakdown(escalation_chain),
            extraction_result=final_result,
            escalation_chain=escalation_chain,
            warnings=warnings,
            errors=errors,
            processing_time_ms=processing_time_ms,
            cost=usage_record.cost
        )
        
        # Validate result invariants
        self._validate_result_invariants(result)
        
        # Cache result
        if self.config.cache_results:
            self.cache[document_id] = (result, datetime.utcnow())
        
        # Log to ledger
        self.ledger.log({
            "doc_id": document_id,
            "final_strategy": current_level.value,
            "confidence": final_confidence,
            "cost": usage_record.cost,
            "processing_time_ms": processing_time_ms,
            "escalation_depth": len(escalation_chain),
            "warnings": warnings,
            "errors": errors,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Update metrics
        self._update_metrics(result)
        
        self.logger.info(f"Extraction completed for {document_id} with confidence {final_confidence:.3f}")
        return result
    
    def _map_profile_to_level(self, profile: DocumentProfile) -> EscalationLevel:
        """Map document profile to initial escalation level."""
        if profile.estimated_extraction_cost == ExtractionCostTier.FAST_TEXT:
            return EscalationLevel.LEVEL_1_FAST
        elif profile.estimated_extraction_cost == ExtractionCostTier.NEEDS_LAYOUT:
            return EscalationLevel.LEVEL_2_LAYOUT
        else:
            return EscalationLevel.LEVEL_3_VISION
    
    def _get_next_level(self, current: EscalationLevel) -> Optional[EscalationLevel]:
        """Get next escalation level."""
        level_order = [
            EscalationLevel.LEVEL_1_FAST,
            EscalationLevel.LEVEL_2_LAYOUT,
            EscalationLevel.LEVEL_3_VISION,
            EscalationLevel.LEVEL_4_HUMAN
        ]
        
        try:
            current_idx = level_order.index(current)
            if current_idx + 1 < len(level_order):
                return level_order[current_idx + 1]
        except ValueError:
            pass
        
        return None
    
    def _calculate_weighted_confidence(self, escalation_chain: List[Dict]) -> float:
        """Calculate weighted confidence across escalation attempts."""
        if not escalation_chain:
            return 0.0
        
        # Weight later attempts more heavily (they use better strategies)
        total_weight = 0.0
        weighted_sum = 0.0
        
        for i, attempt in enumerate(escalation_chain):
            weight = (i + 1) / len(escalation_chain)  # Linear weighting
            weighted_sum += attempt['confidence'] * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _aggregate_confidence_breakdown(self, escalation_chain: List[Dict]) -> Dict[str, float]:
        """Aggregate confidence breakdown across attempts."""
        breakdown = {}
        signal_counts = {}
        
        for attempt in escalation_chain:
            for signal, details in attempt.get('signal_breakdown', {}).items():
                if signal not in breakdown:
                    breakdown[signal] = 0.0
                    signal_counts[signal] = 0
                breakdown[signal] += details['confidence']
                signal_counts[signal] += 1
        
        # Average signals
        for signal in breakdown:
            breakdown[signal] /= signal_counts[signal]
        
        return breakdown
    
    def _get_model_for_level(self, level: EscalationLevel) -> Optional[str]:
        """Get model name for level (for budget tracking)."""
        if level == EscalationLevel.LEVEL_3_VISION:
            return self.config.strategy_overrides[level]['models'][0]
        return None
    
    def _estimate_cost(self, escalation_chain: List[Dict], profile: DocumentProfile) -> float:
        """Estimate cost based on escalation chain."""
        # This would integrate with actual pricing models
        base_costs = {
            EscalationLevel.LEVEL_1_FAST: 0.0,
            EscalationLevel.LEVEL_2_LAYOUT: 0.0261 / 145,  # Per-page cost
            EscalationLevel.LEVEL_3_VISION: 0.00028,  # Per-page cost
        }
        
        total_cost = 0.0
        for attempt in escalation_chain:
            level = EscalationLevel(attempt['level'])
            per_page_cost = base_costs.get(level, 0.0)
            total_cost += per_page_cost * profile.num_pages
        
        return total_cost
    
    def _validate_result_invariants(self, result: EscalationResult):
        """
        Enforce invariants on extraction result.
        
        These invariants ensure data quality and consistency.
        """
        # Invariant 1: Confidence must be between 0 and 1
        if not 0 <= result.confidence <= 1:
            raise ValidationError(f"Confidence {result.confidence} out of range")
        
        # Invariant 2: Must have at least one page
        if not result.extraction_result.pages:
            raise ValidationError("Extraction result has no pages")
        
        # Invariant 3: Page numbers must be sequential
        page_nums = [p.page_num for p in result.extraction_result.pages]
        if page_nums != list(range(1, len(page_nums) + 1)):
            raise ValidationError(f"Page numbers not sequential: {page_nums}")
        
        # Invariant 4: Each table must have headers and rows
        for page in result.extraction_result.pages:
            for table in page.tables:
                if not table.headers:
                    raise ValidationError(f"Table on page {page.page_num} missing headers")
                if not table.rows:
                    raise ValidationError(f"Table on page {page.page_num} has no rows")
        
        # Invariant 5: Content hashes must be present and valid
        for page in result.extraction_result.pages:
            for block in page.text_blocks:
                if not block.content_hash:
                    # Generate if missing
                    block.content_hash = ContentHash.from_content(block.text)
        
        # Invariant 6: Cost must be non-negative
        if result.cost < 0:
            raise ValidationError(f"Negative cost: {result.cost}")
    
    def _update_metrics(self, result: EscalationResult):
        """Update internal metrics."""
        self.metrics["total_processed"] += 1
        self.metrics["escalation_counts"][result.final_strategy] += 1
        self.metrics["total_cost"] += result.cost
        
        # Running averages
        n = self.metrics["total_processed"]
        self.metrics["average_confidence"] = (
            (self.metrics["average_confidence"] * (n - 1) + result.confidence) / n
        )
        self.metrics["average_processing_time"] = (
            (self.metrics["average_processing_time"] * (n - 1) + result.processing_time_ms) / n
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            **self.metrics,
            "cache_size": len(self.cache),
            "budget_summary": self.budget_guard.get_summary()
        }


class BudgetExceededError(Exception):
    """Raised when budget limits are exceeded."""
    pass


class ExtractionFailedError(Exception):
    """Raised when all extraction strategies fail."""
    pass