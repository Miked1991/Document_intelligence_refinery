import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import yaml
from src.agents.extractor import ExtractionRouter
from src.models.document_profile import DocumentProfile, OriginType, LayoutComplexity, ExtractionCostTier
from src.models.extracted_document import ExtractedDocument

class TestExtractionRouter:
    """Test suite for the Extraction Router."""
    
    @pytest.fixture
    def router(self, tmp_path):
        """Create an ExtractionRouter instance with test rules."""
        rules_path = tmp_path / "extraction_rules.yaml"
        rules = {
            "confidence_thresholds": {
                "fast_text": 0.7,
                "layout": 0.8,
                "escalate": 0.6
            }
        }
        with open(rules_path, 'w') as f:
            yaml.dump(rules, f)
        
        ledger_path = tmp_path / "extraction_ledger.jsonl"
        
        return ExtractionRouter(
            rules_path=rules_path,
            ledger_path=ledger_path,
            api_key="test_api_key"
        )
    
    def test_strategy_mapping(self, router, sample_document_profile):
        """Test mapping from profile to strategy."""
        # Test fast text mapping
        sample_document_profile.estimated_extraction_cost = ExtractionCostTier.FAST_TEXT
        assert router._map_profile_to_strategy(sample_document_profile) == "fast"
        
        # Test layout mapping
        sample_document_profile.estimated_extraction_cost = ExtractionCostTier.NEEDS_LAYOUT
        assert router._map_profile_to_strategy(sample_document_profile) == "layout"
        
        # Test vision mapping
        sample_document_profile.estimated_extraction_cost = ExtractionCostTier.NEEDS_VISION
        assert router._map_profile_to_strategy(sample_document_profile) == "vision"
    
    def test_extraction_with_high_confidence(self, router, sample_pdf_path, sample_document_profile):
        """Test extraction when confidence is high (no escalation)."""
        # Mock strategies
        mock_fast = Mock()
        mock_fast.extract.return_value = ExtractedDocument(pages=[])
        mock_fast.confidence.return_value = 0.9  # High confidence
        
        mock_layout = Mock()
        mock_vision = Mock()
        
        router.strategies = {
            "fast": mock_fast,
            "layout": mock_layout,
            "vision": mock_vision
        }
        
        # Mock profile to use fast text
        sample_document_profile.estimated_extraction_cost = ExtractionCostTier.FAST_TEXT
        
        result = router.extract(sample_pdf_path, sample_document_profile)
        
        # Should only use fast text
        mock_fast.extract.assert_called_once()
        mock_layout.extract.assert_not_called()
        mock_vision.extract.assert_not_called()
    
    def test_extraction_with_escalation(self, router, sample_pdf_path, sample_document_profile):
        """Test extraction with confidence below threshold (escalation)."""
        # Mock strategies
        mock_fast = Mock()
        mock_fast.extract.return_value = ExtractedDocument(pages=[])
        mock_fast.confidence.return_value = 0.5  # Low confidence -> escalate
        
        mock_layout = Mock()
        mock_layout.extract.return_value = ExtractedDocument(pages=[])
        mock_layout.confidence.return_value = 0.85  # High enough
        
        mock_vision = Mock()
        
        router.strategies = {
            "fast": mock_fast,
            "layout": mock_layout,
            "vision": mock_vision
        }
        
        # Mock profile to use fast text initially
        sample_document_profile.estimated_extraction_cost = ExtractionCostTier.FAST_TEXT
        
        with patch.object(router.ledger, 'log') as mock_log:
            result = router.extract(sample_pdf_path, sample_document_profile)
            
            # Should escalate from fast to layout
            mock_fast.extract.assert_called_once()
            mock_layout.extract.assert_called_once()
            mock_vision.extract.assert_not_called()
            assert mock_log.call_count >= 2  # Two ledger entries
    
    def test_extraction_full_escalation_chain(self, router, sample_pdf_path, sample_document_profile):
        """Test full escalation chain A -> B -> C."""
        # Mock strategies
        mock_fast = Mock()
        mock_fast.extract.return_value = ExtractedDocument(pages=[])
        mock_fast.confidence.return_value = 0.5  # Low
        
        mock_layout = Mock()
        mock_layout.extract.return_value = ExtractedDocument(pages=[])
        mock_layout.confidence.return_value = 0.5  # Also low -> escalate further
        
        mock_vision = Mock()
        mock_vision.extract.return_value = ExtractedDocument(pages=[])
        mock_vision.confidence.return_value = 0.95  # High
        
        router.strategies = {
            "fast": mock_fast,
            "layout": mock_layout,
            "vision": mock_vision
        }
        
        # Mock profile to use fast text initially
        sample_document_profile.estimated_extraction_cost = ExtractionCostTier.FAST_TEXT
        
        result = router.extract(sample_pdf_path, sample_document_profile)
        
        # Should escalate all the way to vision
        mock_fast.extract.assert_called_once()
        mock_layout.extract.assert_called_once()
        mock_vision.extract.assert_called_once()
    
    def test_escalation_path(self, router):
        """Test escalation path determination."""
        assert router._escalation_path("fast") == "layout"
        assert router._escalation_path("layout") == "vision"
        assert router._escalation_path("vision") is None