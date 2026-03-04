"""Main entry point for production pipeline."""

import asyncio
import logging
from pathlib import Path
from typing import Optional
import yaml

from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter, EscalationResult
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryAgent
from src.utils.confidence import MultiSignalConfidenceScorer
from src.utils.budget import BudgetGuard, BudgetLimits, ModelPricing
from src.utils.ledger import ExtractionLedger
from src.models.document_profile import DocumentProfile


class ProductionPipeline:
    """Production-grade document intelligence pipeline."""
    
    def __init__(self, config_path: Path, api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.triage = TriageAgent(config_path)
        self.ledger = ExtractionLedger(Path(".refinery/extraction_ledger.jsonl"))
        
        # Initialize confidence scorer
        self.confidence_scorer = MultiSignalConfidenceScorer(
            self.config['confidence']
        )
        
        # Initialize budget guard
        budget_limits = BudgetLimits(**self.config['budget']['limits'])
        model_pricing = {
            name: ModelPricing(**pricing) 
            for name, pricing in self.config['budget']['model_pricing'].items()
        }
        self.budget_guard = BudgetGuard(
            limits=budget_limits,
            model_pricing=model_pricing,
            storage_path=Path(".refinery/budget")
        )
        
        # Initialize extraction router
        self.extractor = ExtractionRouter(
            config_path=config_path,
            ledger=self.ledger,
            budget_guard=self.budget_guard,
            confidence_scorer=self.confidence_scorer,
            cache_dir=Path(".refinery/cache")
        )
        
        # Initialize other components
        self.chunker = ChunkingEngine(
            rules=self.config['chunking']['rules'],
            validator=None  # Would initialize properly
        )
        
        self.indexer = PageIndexBuilder(
            llm_client=None  # Would initialize properly
        )
        
        self.query_agent = None  # Initialized after indexing
    
    async def process_document(self, pdf_path: Path) -> EscalationResult:
        """Process a single document through the entire pipeline."""
        self.logger.info(f"Processing document: {pdf_path}")
        
        # Step 1: Triage
        profile = self.triage.profile(pdf_path)
        self.logger.info(f"Triage complete: {profile.origin_type}, {profile.layout_complexity}")
        
        # Step 2: Extraction with escalation
        extraction_result = await self.extractor.extract_with_escalation(
            pdf_path, profile
        )
        self.logger.info(f"Extraction complete with confidence {extraction_result.confidence:.3f}")
        
        # Step 3: Validate invariants
        self._validate_extraction(extraction_result)
        
        # Step 4: Chunking
        ldus = self.chunker.chunk(extraction_result.extraction_result)
        self.logger.info(f"Chunking complete: {len(ldus)} LDUs created")
        
        # Step 5: Build PageIndex
        pageindex = self.indexer.build(
            extraction_result.extraction_result, ldus
        )
        
        # Save artifacts
        self._save_artifacts(profile, extraction_result, pageindex)
        
        return extraction_result
    
    def _validate_extraction(self, result: EscalationResult):
        """Additional validation beyond invariants."""
        # Check if confidence meets minimum threshold
        min_acceptable = self.config['escalation']['thresholds']['acceptable']
        if result.confidence < min_acceptable:
            self.logger.warning(
                f"Low confidence {result.confidence:.3f} below threshold {min_acceptable}"
            )
        
        # Check for critical failures
        if result.confidence < self.config['escalation']['thresholds']['critical_failure']:
            raise ValueError(
                f"Critical failure: confidence {result.confidence:.3f} too low"
            )
    
    def _save_artifacts(self, profile: DocumentProfile, result: EscalationResult, pageindex):
        """Save all artifacts to disk."""
        # Save profile
        profile_path = Path(f".refinery/profiles/{profile.doc_id}.json")
        with open(profile_path, 'w') as f:
            f.write(profile.json(indent=2))
        
        # Save extraction result
        result_path = Path(f".refinery/extractions/{profile.doc_id}.json")
        result_path.parent.mkdir(exist_ok=True)
        with open(result_path, 'w') as f:
            f.write(result.json(indent=2))
        
        # Save PageIndex
        index_path = Path(f".refinery/pageindex/{profile.doc_id}.json")
        with open(index_path, 'w') as f:
            f.write(pageindex.json(indent=2))
    
    async def process_batch(self, pdf_paths: List[Path], max_concurrent: int = 4):
        """Process multiple documents concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(pdf_path):
            async with semaphore:
                return await self.process_document(pdf_path)
        
        tasks = [process_with_semaphore(path) for path in pdf_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Summarize results
        successes = [r for r in results if isinstance(r, EscalationResult)]
        failures = [r for r in results if isinstance(r, Exception)]
        
        self.logger.info(f"Batch complete: {len(successes)} succeeded, {len(failures)} failed")
        
        return successes, failures
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline metrics."""
        return {
            "extraction": self.extractor.get_metrics(),
            "budget": self.budget_guard.get_summary(),
            "confidence": self.confidence_scorer.get_statistical_summary()
        }


async def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize pipeline
    pipeline = ProductionPipeline(
        config_path=Path("rubric/production_rules.yaml"),
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    # Process single document
    result = await pipeline.process_document(Path("data/sample.pdf"))
    print(f"Processed with confidence: {result.confidence:.3f}")
    print(f"Cost: ${result.cost:.4f}")
    
    # Process batch
    pdf_files = list(Path("data/corpus").glob("*.pdf"))
    successes, failures = await pipeline.process_batch(pdf_files[:10])
    
    # Print metrics
    print(json.dumps(pipeline.get_pipeline_metrics(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())