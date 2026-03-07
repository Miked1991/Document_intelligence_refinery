"""
Base classes for extraction strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional
from ..models.extracted_document import ExtractedDocument
from ..models.document_profile import DocumentProfile


class ExtractionStrategy(ABC):
    """
    Abstract base class for all extraction strategies.
    """
    
    def __init__(self, name: str):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
        """
        self.name = name
    
    @abstractmethod
    async def extract(
        self,
        pdf_path: str,
        profile: Optional[DocumentProfile] = None
    ) -> ExtractedDocument:
        """
        Extract content from PDF.
        
        Args:
            pdf_path: Path to PDF file
            profile: Optional document profile for guidance
            
        Returns:
            Extracted document
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, pdf_path: str) -> float:
        """
        Estimate extraction cost.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Estimated cost in USD
        """
        pass
    
    def can_handle(self, profile: DocumentProfile) -> bool:
        """
        Check if strategy can handle the document profile.
        
        Args:
            profile: Document profile
            
        Returns:
            True if strategy can handle
        """
        return True