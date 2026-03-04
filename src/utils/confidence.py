"""Multi-signal confidence scoring with statistical validation."""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np
from dataclasses import dataclass
from scipy import stats
import pdfplumber
from pathlib import Path


class ConfidenceSignal(str, Enum):
    """Types of confidence signals."""
    CHARACTER_DENSITY = "character_density"
    FONT_CONSISTENCY = "font_consistency"
    TABLE_COMPLETENESS = "table_completeness"
    READING_ORDER_COHERENCE = "reading_order_coherence"
    LANGUAGE_MODEL_PERPLEXITY = "language_model_perplexity"
    OCR_CONFIDENCE = "ocr_confidence"
    LAYOUT_STABILITY = "layout_stability"
    CROSS_PAGE_CONSISTENCY = "cross_page_consistency"
    ENTITY_RECOGNITION_CONFIDENCE = "entity_recognition_confidence"
    NUMERICAL_PRECISION = "numerical_precision"


class SignalWeights(BaseModel):
    """Weight configuration for confidence signals."""
    weights: Dict[ConfidenceSignal, float] = Field(default_factory=dict)
    
    @validator('weights')
    def weights_must_sum_to_one(cls, v):
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:  # Allow small floating point error
            raise ValueError(f'Weights must sum to 1.0, got {total}')
        return v
    
    def get_weight(self, signal: ConfidenceSignal) -> float:
        """Get weight for a signal, defaulting to equal weighting."""
        return self.weights.get(signal, 1.0 / len(ConfidenceSignal))


@dataclass
class SignalResult:
    """Result of a single confidence signal."""
    signal: ConfidenceSignal
    value: float
    confidence: float
    metadata: Dict[str, Any]
    sample_size: int
    variance: Optional[float] = None


class CharacterDensityAnalyzer:
    """Analyze character density patterns."""
    
    def __init__(self, expected_density: float = 0.0005, 
                 min_chars_per_page: int = 100):
        self.expected_density = expected_density
        self.min_chars_per_page = min_chars_per_page
    
    def analyze(self, page) -> SignalResult:
        """Analyze character density for a page."""
        text = page.extract_text() or ""
        chars = len(text)
        area = page.width * page.height
        
        if area == 0:
            density = 0
        else:
            density = chars / area
        
        # Calculate confidence based on deviation from expected
        if chars < self.min_chars_per_page:
            confidence = max(0, chars / self.min_chars_per_page)
        else:
            ratio = density / self.expected_density
            if ratio > 2.0:
                confidence = 2.0 / ratio  # Penalize too-high density (possible OCR errors)
            else:
                confidence = ratio
        
        return SignalResult(
            signal=ConfidenceSignal.CHARACTER_DENSITY,
            value=density,
            confidence=min(1.0, confidence),
            metadata={"char_count": chars, "area": area},
            sample_size=1
        )


class FontConsistencyAnalyzer:
    """Analyze font consistency across document."""
    
    def analyze(self, pdf_path: Path) -> SignalResult:
        """Analyze font consistency."""
        font_counts = {}
        total_chars = 0
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                if hasattr(page, 'chars'):
                    for char in page.chars:
                        font = char.get('fontname', 'unknown')
                        font_counts[font] = font_counts.get(font, 0) + 1
                        total_chars += 1
        
        if total_chars == 0:
            return SignalResult(
                signal=ConfidenceSignal.FONT_CONSISTENCY,
                value=0.0,
                confidence=0.0,
                metadata={"error": "No characters found"},
                sample_size=0
            )
        
        # Calculate entropy of font distribution
        proportions = [count / total_chars for count in font_counts.values()]
        entropy = stats.entropy(proportions)
        max_entropy = np.log(len(font_counts)) if font_counts else 0
        
        # Normalize entropy to confidence (lower entropy = more consistent = higher confidence)
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
            confidence = 1.0 - normalized_entropy
        else:
            confidence = 1.0
        
        return SignalResult(
            signal=ConfidenceSignal.FONT_CONSISTENCY,
            value=entropy,
            confidence=confidence,
            metadata={"unique_fonts": len(font_counts), "total_chars": total_chars},
            sample_size=total_chars
        )


class TableCompletenessAnalyzer:
    """Analyze table extraction completeness."""
    
    def __init__(self, min_rows: int = 2, min_cols: int = 2):
        self.min_rows = min_rows
        self.min_cols = min_cols
    
    def analyze(self, table_data: List[List[Optional[str]]]) -> SignalResult:
        """Analyze table completeness."""
        if not table_data or len(table_data) < self.min_rows:
            return SignalResult(
                signal=ConfidenceSignal.TABLE_COMPLETENESS,
                value=0.0,
                confidence=0.0,
                metadata={"error": "Table too small"},
                sample_size=0
            )
        
        rows = len(table_data)
        cols = max(len(row) for row in table_data) if table_data else 0
        
        if cols < self.min_cols:
            return SignalResult(
                signal=ConfidenceSignal.TABLE_COMPLETENESS,
                value=0.0,
                confidence=0.0,
                metadata={"error": "Too few columns"},
                sample_size=0
            )
        
        # Check for empty cells
        total_cells = rows * cols
        filled_cells = 0
        
        for row in table_data:
            # Pad row to consistent length
            padded_row = row + [None] * (cols - len(row))
            filled_cells += sum(1 for cell in padded_row if cell and cell.strip())
        
        completeness = filled_cells / total_cells if total_cells > 0 else 0
        
        # Check for structural consistency (all rows same length)
        row_lengths = [len(row) for row in table_data]
        length_consistency = 1.0 - (np.std(row_lengths) / (np.mean(row_lengths) + 1e-10))
        
        # Combine signals
        confidence = completeness * 0.7 + length_consistency * 0.3
        
        return SignalResult(
            signal=ConfidenceSignal.TABLE_COMPLETENESS,
            value=completeness,
            confidence=min(1.0, confidence),
            metadata={
                "rows": rows,
                "cols": cols,
                "filled_cells": filled_cells,
                "total_cells": total_cells
            },
            sample_size=total_cells
        )


class ReadingOrderAnalyzer:
    """Analyze reading order coherence."""
    
    def analyze(self, text_blocks: List[Dict]) -> SignalResult:
        """Analyze if reading order makes sense."""
        if len(text_blocks) < 2:
            return SignalResult(
                signal=ConfidenceSignal.READING_ORDER_COHERENCE,
                value=1.0,
                confidence=1.0,
                metadata={"blocks": len(text_blocks)},
                sample_size=len(text_blocks)
            )
        
        # Check for logical progression (y-coordinate should generally decrease)
        y_coords = [block.get('bbox', [0, 0, 0, 0])[1] for block in text_blocks]
        y_diffs = np.diff(y_coords)
        
        # In proper reading order, y should stay same or increase slowly
        # Large decreases might indicate column breaks
        reasonable_diffs = sum(1 for diff in y_diffs if -100 < diff < 50)
        coherence = reasonable_diffs / len(y_diffs)
        
        # Check for text continuity (last words of one block should relate to first of next)
        continuity_score = 0.0
        for i in range(len(text_blocks) - 1):
            last_words = text_blocks[i].get('text', '').split()[-3:]
            first_words = text_blocks[i+1].get('text', '').split()[:3]
            
            if last_words and first_words:
                # Simple heuristic: check for lowercase start (continuation)
                if first_words[0] and first_words[0][0].islower():
                    continuity_score += 1
        
        continuity = continuity_score / (len(text_blocks) - 1) if len(text_blocks) > 1 else 1.0
        
        # Combine signals
        confidence = coherence * 0.6 + continuity * 0.4
        
        return SignalResult(
            signal=ConfidenceSignal.READING_ORDER_COHERENCE,
            value=coherence,
            confidence=confidence,
            metadata={
                "coherence": coherence,
                "continuity": continuity,
                "blocks": len(text_blocks)
            },
            sample_size=len(text_blocks)
        )


class NumericalPrecisionAnalyzer:
    """Analyze numerical precision in extracted data."""
    
    def __init__(self, expected_precision: int = 2):
        self.expected_precision = expected_precision
    
    def analyze(self, numbers: List[str]) -> SignalResult:
        """Analyze numerical precision."""
        if not numbers:
            return SignalResult(
                signal=ConfidenceSignal.NUMERICAL_PRECISION,
                value=1.0,
                confidence=1.0,
                metadata={"numbers_found": 0},
                sample_size=0
            )
        
        precision_scores = []
        valid_numbers = 0
        
        for num_str in numbers:
            try:
                # Try to parse as number
                if '.' in num_str:
                    decimal_places = len(num_str.split('.')[1])
                else:
                    decimal_places = 0
                
                # Check if precision matches expected
                if decimal_places <= self.expected_precision:
                    precision_scores.append(1.0)
                else:
                    # Penalize extra precision (possible OCR artifacts)
                    precision_scores.append(max(0, 1.0 - (decimal_places - self.expected_precision) * 0.2))
                
                valid_numbers += 1
            except:
                continue
        
        if valid_numbers == 0:
            return SignalResult(
                signal=ConfidenceSignal.NUMERICAL_PRECISION,
                value=0.0,
                confidence=0.0,
                metadata={"valid_numbers": 0},
                sample_size=len(numbers)
            )
        
        avg_precision = np.mean(precision_scores)
        
        return SignalResult(
            signal=ConfidenceSignal.NUMERICAL_PRECISION,
            value=avg_precision,
            confidence=avg_precision,
            metadata={
                "total_numbers": len(numbers),
                "valid_numbers": valid_numbers,
                "avg_precision": avg_precision
            },
            sample_size=valid_numbers
        )


class MultiSignalConfidenceScorer:
    """Orchestrates multiple confidence signals with statistical validation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weights = SignalWeights(weights=config.get('signal_weights', {}))
        self.analyzers = self._init_analyzers()
        self.history: List[Dict] = []
    
    def _init_analyzers(self) -> Dict[ConfidenceSignal, Any]:
        """Initialize analyzers based on config."""
        analyzers = {}
        
        if self.config.get('enable_character_density', True):
            analyzers[ConfidenceSignal.CHARACTER_DENSITY] = CharacterDensityAnalyzer(
                expected_density=self.config.get('expected_density', 0.0005),
                min_chars_per_page=self.config.get('min_chars_per_page', 100)
            )
        
        if self.config.get('enable_font_consistency', True):
            analyzers[ConfidenceSignal.FONT_CONSISTENCY] = FontConsistencyAnalyzer()
        
        if self.config.get('enable_table_completeness', True):
            analyzers[ConfidenceSignal.TABLE_COMPLETENESS] = TableCompletenessAnalyzer(
                min_rows=self.config.get('min_table_rows', 2),
                min_cols=self.config.get('min_table_cols', 2)
            )
        
        if self.config.get('enable_reading_order', True):
            analyzers[ConfidenceSignal.READING_ORDER_COHERENCE] = ReadingOrderAnalyzer()
        
        if self.config.get('enable_numerical_precision', True):
            analyzers[ConfidenceSignal.NUMERICAL_PRECISION] = NumericalPrecisionAnalyzer(
                expected_precision=self.config.get('expected_numerical_precision', 2)
            )
        
        return analyzers
    
    def score_document(self, pdf_path: Path, extraction_result: Dict) -> Dict[str, Any]:
        """Compute comprehensive confidence score for a document."""
        signals = []
        
        with pdfplumber.open(pdf_path) as pdf:
            # Page-level signals
            page_signals = []
            for page_num, page in enumerate(pdf.pages):
                page_results = []
                
                if ConfidenceSignal.CHARACTER_DENSITY in self.analyzers:
                    result = self.analyzers[ConfidenceSignal.CHARACTER_DENSITY].analyze(page)
                    page_results.append(result)
                
                if page_results:
                    page_confidence = np.mean([r.confidence for r in page_results])
                    page_signals.append(page_confidence)
            
            if page_signals:
                signals.append(SignalResult(
                    signal=ConfidenceSignal.CROSS_PAGE_CONSISTENCY,
                    value=np.mean(page_signals),
                    confidence=1.0 - np.std(page_signals) if len(page_signals) > 1 else 1.0,
                    metadata={
                        "mean_page_confidence": np.mean(page_signals),
                        "std_page_confidence": np.std(page_signals) if len(page_signals) > 1 else 0
                    },
                    sample_size=len(page_signals)
                ))
        
        # Document-level signals
        if ConfidenceSignal.FONT_CONSISTENCY in self.analyzers:
            signals.append(self.analyzers[ConfidenceSignal.FONT_CONSISTENCY].analyze(pdf_path))
        
        # Extraction-level signals
        if extraction_result.get('tables'):
            table_results = []
            for table in extraction_result['tables']:
                if ConfidenceSignal.TABLE_COMPLETENESS in self.analyzers:
                    result = self.analyzers[ConfidenceSignal.TABLE_COMPLETENESS].analyze(table.get('data', []))
                    table_results.append(result)
            
            if table_results:
                signals.append(SignalResult(
                    signal=ConfidenceSignal.TABLE_COMPLETENESS,
                    value=np.mean([r.value for r in table_results]),
                    confidence=np.mean([r.confidence for r in table_results]),
                    metadata={"table_count": len(table_results)},
                    sample_size=len(table_results)
                ))
        
        if extraction_result.get('text_blocks'):
            if ConfidenceSignal.READING_ORDER_COHERENCE in self.analyzers:
                signals.append(self.analyzers[ConfidenceSignal.READING_ORDER_COHERENCE].analyze(
                    extraction_result.get('text_blocks', [])
                ))
        
        if extraction_result.get('numbers'):
            if ConfidenceSignal.NUMERICAL_PRECISION in self.analyzers:
                signals.append(self.analyzers[ConfidenceSignal.NUMERICAL_PRECISION].analyze(
                    extraction_result.get('numbers', [])
                ))
        
        # Calculate weighted confidence
        total_confidence = 0.0
        total_weight = 0.0
        
        signal_details = {}
        for signal in signals:
            weight = self.weights.get_weight(signal.signal)
            total_confidence += signal.confidence * weight
            total_weight += weight
            
            signal_details[signal.signal.value] = {
                "value": signal.value,
                "confidence": signal.confidence,
                "metadata": signal.metadata,
                "sample_size": signal.sample_size,
                "weight": weight
            }
        
        final_confidence = total_confidence / total_weight if total_weight > 0 else 0.0
        
        # Store in history
        result = {
            "final_confidence": final_confidence,
            "signal_details": signal_details,
            "signal_count": len(signals),
            "timestamp": datetime.utcnow().isoformat()
        }
        self.history.append(result)
        
        return result
    
    def get_statistical_summary(self) -> Dict[str, Any]:
        """Get statistical summary of all scores."""
        if not self.history:
            return {}
        
        confidences = [h['final_confidence'] for h in self.history]
        
        return {
            "mean": np.mean(confidences),
            "median": np.median(confidences),
            "std": np.std(confidences),
            "min": np.min(confidences),
            "max": np.max(confidences),
            "percentiles": {
                "25": np.percentile(confidences, 25),
                "50": np.percentile(confidences, 50),
                "75": np.percentile(confidences, 75),
                "90": np.percentile(confidences, 90),
                "95": np.percentile(confidences, 95)
            },
            "sample_size": len(confidences)
        }