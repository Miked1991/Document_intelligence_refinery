"""
Triage Agent for document classification and profiling.
FIXED: image_area_ratio calculation now properly capped at 1.0.
"""

import pdfplumber
from typing import Optional, Dict, List, Tuple
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import hashlib

from ..models.document_profile import (
    DocumentProfile, OriginType, LayoutComplexity,
    DomainHint, ExtractionStrategy
)
from ..utils.confidence_scorer import ConfidenceScorer


class TriageAgent:
    """
    Agent for classifying documents and generating profiles.
    """
    
    def __init__(self, rules_path: Optional[str] = None):
        """
        Initialize triage agent.
        
        Args:
            rules_path: Path to extraction rules YAML
        """
        self.confidence_scorer = ConfidenceScorer()
        self.rules = self._load_rules(rules_path)
        
        # Domain keyword mapping
        self.domain_keywords = {
            DomainHint.FINANCIAL: [
                'revenue', 'profit', 'loss', 'income', 'expense',
                'balance sheet', 'income statement', 'cash flow',
                'audit', 'financial', 'fiscal', 'quarter', 'annual',
                'dividend', 'earnings', 'eps', 'ebitda', 'tax'
            ],
            DomainHint.LEGAL: [
                'agreement', 'contract', 'clause', 'party', 'hereby',
                'witness', 'legal', 'court', 'law', 'regulation',
                'compliance', 'statute', 'provision', 'liability'
            ],
            DomainHint.TECHNICAL: [
                'specification', 'technical', 'system', 'interface',
                'api', 'implementation', 'algorithm', 'function',
                'parameter', 'configuration', 'architecture'
            ],
            DomainHint.MEDICAL: [
                'patient', 'clinical', 'diagnosis', 'treatment',
                'medical', 'health', 'prescription', 'dose',
                'symptom', 'disease', 'therapy', 'hospital'
            ]
        }
    
    def _load_rules(self, rules_path: Optional[str]) -> Dict:
        """Load extraction rules from YAML"""
        default_rules = {
            "character_density_threshold": 100,
            "image_ratio_threshold": 0.3,
            "table_density_threshold": 0.1,
            "multi_column_threshold": 1.5,
            "confidence_thresholds": {
                "fast_text": 0.6,
                "layout_aware": 0.7,
                "vision_augmented": 0.8
            }
        }
        
        if rules_path and Path(rules_path).exists():
            import yaml
            with open(rules_path) as f:
                custom_rules = yaml.safe_load(f)
                default_rules.update(custom_rules)
        
        return default_rules
    
    async def profile_document(self, pdf_path: str) -> DocumentProfile:
        """
        Generate comprehensive document profile.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            DocumentProfile with classification
        """
        doc_id = hashlib.md5(pdf_path.encode()).hexdigest()[:12]
        filename = Path(pdf_path).name
        
        # Analyze document
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            file_size = Path(pdf_path).stat().st_size
            
            # Collect page statistics
            char_densities = []
            image_ratios = []
            table_counts = []
            font_present = False
            text_pages = 0
            
            for page in pdf.pages:
                # Character density
                text = page.extract_text() or ""
                char_count = len(text)
                page_area = page.width * page.height
                char_density = char_count / page_area if page_area > 0 else 0
                char_densities.append(char_density)
                
                if char_count > 100:
                    text_pages += 1
                
                # Image ratio - FIXED: Ensure ratio doesn't exceed 1.0
                if page.images:
                    image_area = sum(
                        img.get('width', 0) * img.get('height', 0)
                        for img in page.images
                    )
                    # Calculate ratio and cap at 1.0
                    raw_ratio = image_area / page_area if page_area > 0 else 0
                    image_ratio = min(1.0, raw_ratio)  # FIXED: Cap at 1.0
                    image_ratios.append(image_ratio)
                else:
                    image_ratios.append(0.0)
                
                # Table detection (heuristic)
                tables = page.extract_tables()
                table_counts.append(len(tables))
                
                # Font presence
                if page.chars and any('fontname' in c for c in page.chars):
                    font_present = True
            
            # Calculate statistics
            char_density_stats = {
                "mean": float(np.mean(char_densities)) if char_densities else 0,
                "min": float(np.min(char_densities)) if char_densities else 0,
                "max": float(np.max(char_densities)) if char_densities else 0,
                "std": float(np.std(char_densities)) if char_densities else 0
            }
            
            # FIXED: Ensure average image ratio is capped at 1.0
            avg_image_ratio = float(np.mean(image_ratios)) if image_ratios else 0
            avg_image_ratio = min(1.0, avg_image_ratio)  # Extra safety cap
            
            avg_tables_per_page = float(np.mean(table_counts)) if table_counts else 0
            
            # Classifications
            origin_type = self._classify_origin(
                text_pages, page_count, avg_image_ratio, font_present
            )
            
            layout_complexity = self._classify_layout(
                char_densities, avg_tables_per_page, pdf
            )
            
            # Extract sample text for domain classification
            sample_text = ""
            for i, page in enumerate(pdf.pages[:5]):  # First 5 pages
                sample_text += page.extract_text() or ""
            
            domain_hint = self._classify_domain(sample_text)
            
            # Recommended strategy
            recommended = self._recommend_strategy(
                origin_type, layout_complexity,
                avg_image_ratio, avg_tables_per_page
            )
        
        return DocumentProfile(
            doc_id=doc_id,
            filename=filename,
            file_size_bytes=file_size,
            page_count=page_count,
            origin_type=origin_type,
            layout_complexity=layout_complexity,
            language={"en": 0.9},  # Simplified
            domain_hint=domain_hint,
            recommended_strategy=recommended,
            character_density_stats=char_density_stats,
            image_area_ratio=avg_image_ratio,  # Now guaranteed to be ≤ 1.0
            has_embedded_fonts=font_present,
            table_count_estimate=int(avg_tables_per_page * page_count),
            figure_count_estimate=0,  # Would need proper figure detection
            created_at=datetime.now()
        )
    
    def _classify_origin(
        self,
        text_pages: int,
        total_pages: int,
        image_ratio: float,
        has_fonts: bool
    ) -> OriginType:
        """Classify document origin type"""
        text_ratio = text_pages / total_pages if total_pages > 0 else 0
        
        if text_ratio > 0.8 and has_fonts:
            return OriginType.NATIVE_DIGITAL
        elif image_ratio > 0.5:
            if text_ratio < 0.2:
                return OriginType.SCANNED_IMAGE
            else:
                return OriginType.MIXED
        elif 0.3 < text_ratio < 0.8:
            return OriginType.MIXED
        else:
            return OriginType.SCANNED_IMAGE
    
    def _classify_layout(
        self,
        char_densities: List[float],
        tables_per_page: float,
        pdf
    ) -> LayoutComplexity:
        """Classify layout complexity"""
        # Check for tables
        if tables_per_page > 0.3:
            return LayoutComplexity.TABLE_HEAVY
        
        # Check for multi-column (simplified heuristic)
        if char_densities:
            # Multi-column documents often have varying densities
            density_std = np.std(char_densities)
            if density_std > 0.5:  # Threshold needs tuning
                return LayoutComplexity.MULTI_COLUMN
        
        # Check first page for multi-column
        try:
            first_page = pdf.pages[0]
            words = first_page.extract_words()
            
            if words:
                # Group by y-coordinate to find lines
                lines = {}
                for word in words:
                    y_key = round(word['top'], 0)
                    if y_key not in lines:
                        lines[y_key] = []
                    lines[y_key].append(word)
                
                # Check if lines have multiple x-clusters (columns)
                multi_column_count = 0
                for line_words in lines.values():
                    x_positions = [w['x0'] for w in line_words]
                    if len(x_positions) > 3:
                        # Check for gaps
                        x_sorted = sorted(x_positions)
                        gaps = np.diff(x_sorted)
                        if any(gap > 100 for gap in gaps):  # Large gap indicates column break
                            multi_column_count += 1
                
                if multi_column_count > len(lines) * 0.3:
                    return LayoutComplexity.MULTI_COLUMN
        except:
            pass
        
        return LayoutComplexity.SINGLE_COLUMN
    
    def _classify_domain(self, text: str) -> DomainHint:
        """Classify document domain based on keywords"""
        if not text:
            return DomainHint.GENERAL
        
        text_lower = text.lower()
        scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[domain] = score / len(keywords) if keywords else 0
        
        if not scores:
            return DomainHint.GENERAL
        
        # Get domain with highest score
        best_domain = max(scores, key=scores.get)
        
        # If score is too low, return GENERAL
        if scores[best_domain] < 0.05:
            return DomainHint.GENERAL
        
        return best_domain
    
    def _recommend_strategy(
        self,
        origin: OriginType,
        layout: LayoutComplexity,
        image_ratio: float,
        tables_per_page: float
    ) -> ExtractionStrategy:
        """Recommend extraction strategy"""
        if origin == OriginType.SCANNED_IMAGE:
            return ExtractionStrategy.VISION_AUGMENTED
        elif origin == OriginType.MIXED and image_ratio > 0.3:
            return ExtractionStrategy.VISION_AUGMENTED
        
        if layout in [LayoutComplexity.TABLE_HEAVY, LayoutComplexity.MULTI_COLUMN]:
            return ExtractionStrategy.LAYOUT_AWARE
        
        if tables_per_page > 0.2:
            return ExtractionStrategy.LAYOUT_AWARE
        
        return ExtractionStrategy.FAST_TEXT
    
    async def save_profile(self, profile: DocumentProfile, output_dir: str = ".refinery/profiles"):
        """Save profile to JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / f"{profile.doc_id}.json"
        with open(filepath, 'w') as f:
            f.write(profile.model_dump_json(indent=2))
        
        return filepath