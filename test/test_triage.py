import pytest
from pathlib import Path
import yaml
from unittest.mock import Mock, patch, mock_open
from src.agents.triage import TriageAgent
from src.models.document_profile import OriginType, LayoutComplexity, DomainHint, ExtractionCostTier

class TestTriageAgent:
    """Test suite for the Triage Agent."""
    
    @pytest.fixture
    def triage_agent(self, tmp_path):
        """Create a TriageAgent instance with test rules."""
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
        
        return TriageAgent(rules_path)
    
    def test_initialization(self, triage_agent):
        """Test that the agent initializes correctly."""
        assert triage_agent.rules is not None
        assert "confidence_thresholds" in triage_agent.rules
    
    def test_origin_detection_native_digital(self, triage_agent, sample_pdf_path):
        """Test detection of native digital PDFs."""
        with patch('pdfplumber.open') as mock_pdf:
            # Mock PDF with high text content
            mock_page = Mock()
            mock_page.extract_text.return_value = "This is a lot of text content that exceeds 100 characters. " * 10
            mock_page.images = []
            
            mock_pdf.return_value.__enter__.return_value.pages = [mock_page] * 10
            
            profile = triage_agent.profile(sample_pdf_path)
            assert profile.origin_type == OriginType.NATIVE_DIGITAL
    
    def test_origin_detection_scanned(self, triage_agent, sample_pdf_path):
        """Test detection of scanned PDFs."""
        with patch('pdfplumber.open') as mock_pdf:
            # Mock PDF with no text but many images
            mock_page = Mock()
            mock_page.extract_text.return_value = ""
            mock_page.images = [Mock()] * 5  # Many images
            
            mock_pdf.return_value.__enter__.return_value.pages = [mock_page] * 10
            
            profile = triage_agent.profile(sample_pdf_path)
            assert profile.origin_type == OriginType.SCANNED_IMAGE
    
    def test_layout_complexity_detection(self, triage_agent, sample_pdf_path):
        """Test detection of layout complexity."""
        with patch('pdfplumber.open') as mock_pdf, \
             patch('src.agents.triage.estimate_column_count') as mock_cols, \
             patch('src.agents.triage.image_area_ratio') as mock_ratio:
            
            # Mock a page with multiple columns and tables
            mock_page = Mock()
            mock_page.width = 800
            mock_page.height = 1000
            mock_page.find_tables.return_value = [Mock()] * 10  # Many tables
            
            mock_pdf.return_value.__enter__.return_value.pages = [mock_page]
            mock_cols.return_value = 2
            mock_ratio.return_value = 0.3
            
            profile = triage_agent.profile(sample_pdf_path)
            assert profile.layout_complexity == LayoutComplexity.TABLE_HEAVY
    
    def test_domain_classification_financial(self, triage_agent):
        """Test financial domain classification."""
        text = "The company's revenue increased by 20%. The balance sheet shows strong performance."
        domain = triage_agent._classify_domain(text)
        assert domain == DomainHint.FINANCIAL
    
    def test_domain_classification_legal(self, triage_agent):
        """Test legal domain classification."""
        text = "The plaintiff filed a motion against the defendant in court."
        domain = triage_agent._classify_domain(text)
        assert domain == DomainHint.LEGAL
    
    def test_domain_classification_technical(self, triage_agent):
        """Test technical domain classification."""
        text = "The algorithm processes data using machine learning models."
        domain = triage_agent._classify_domain(text)
        assert domain == DomainHint.TECHNICAL
    
    def test_domain_classification_medical(self, triage_agent):
        """Test medical domain classification."""
        text = "The patient was diagnosed with hypertension and prescribed treatment."
        domain = triage_agent._classify_domain(text)
        assert domain == DomainHint.MEDICAL
    
    def test_domain_classification_general(self, triage_agent):
        """Test general domain classification."""
        text = "This is a general document about various topics."
        domain = triage_agent._classify_domain(text)
        assert domain == DomainHint.GENERAL
    
    def test_extraction_cost_tier_selection(self, triage_agent):
        """Test that correct cost tier is selected based on profile."""
        # Test native digital simple -> FAST_TEXT
        profile = triage_agent.profile.__wrapped__(  # Bypass mocks
            triage_agent, 
            Path("dummy.pdf")
        )
        # Mock the internal calls
        with patch('pdfplumber.open') as mock_pdf:
            mock_page = Mock()
            mock_page.extract_text.return_value = "text"
            mock_pdf.return_value.__enter__.return_value.pages = [mock_page]
            
            profile.origin_type = OriginType.NATIVE_DIGITAL
            profile.layout_complexity = LayoutComplexity.SINGLE_COLUMN
            assert profile.estimated_extraction_cost == ExtractionCostTier.FAST_TEXT
            
            # Test complex layout -> NEEDS_LAYOUT
            profile.layout_complexity = LayoutComplexity.TABLE_HEAVY
            assert profile.estimated_extraction_cost == ExtractionCostTier.NEEDS_LAYOUT
            
            # Test scanned -> NEEDS_VISION
            profile.origin_type = OriginType.SCANNED_IMAGE
            assert profile.estimated_extraction_cost == ExtractionCostTier.NEEDS_VISION