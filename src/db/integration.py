"""
Integration layer between extraction pipeline and databases.
FIXED: Added validation for numeric value conversion to handle empty strings.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import re

from ..models.ldu import LDU, ChunkCollection
from ..models.extracted_document import ExtractedDocument, Table
from .vector_store import VectorStore
from .fact_table import FactTable


class DatabaseIntegrator:
    """
    Integrates vector store and fact table with extraction pipeline.
    """
    
    def __init__(
        self,
        vector_store_path: str = ".refinery/vectors",
        fact_table_path: str = ".refinery/facts/facts.db"
    ):
        """
        Initialize database integrator.
        
        Args:
            vector_store_path: Path for vector store
            fact_table_path: Path for fact table
        """
        self.vector_store = VectorStore(persist_directory=vector_store_path)
        self.fact_table = FactTable(db_path=fact_table_path)
    
    def process_document(
        self,
        doc: ExtractedDocument,
        chunks: ChunkCollection
    ) -> Dict[str, Any]:
        """
        Process a document through both databases.
        
        Args:
            doc: Extracted document
            chunks: Chunk collection
            
        Returns:
            Processing statistics
        """
        stats = {
            'doc_id': doc.doc_id,
            'chunks_added': 0,
            'facts_extracted': 0,
            'entities_extracted': 0
        }
        
        # 1. Add chunks to vector store
        stats['chunks_added'] = self.vector_store.add_chunks(chunks)
        
        # 2. Extract and insert facts
        facts = self._extract_facts_from_document(doc, chunks)
        if facts['financial']:
            self.fact_table.insert_financial_facts_batch(facts['financial'])
            stats['facts_extracted'] = len(facts['financial'])
        
        # 3. Extract entities
        entities = self._extract_entities_from_document(doc, chunks)
        for entity in entities:
            self.fact_table.insert_entity(entity)
        stats['entities_extracted'] = len(entities)
        
        # 4. Extract key-value pairs
        kv_pairs = self._extract_key_values_from_document(doc, chunks)
        for kv in kv_pairs:
            self.fact_table.insert_key_value(kv)
        stats['kv_extracted'] = len(kv_pairs)
        
        # 5. Update document stats
        self.fact_table.update_document_stats(
            doc_id=doc.doc_id,
            filename=doc.filename,
            page_count=doc.page_count
        )
        
        return stats
    
    def _extract_facts_from_document(
        self,
        doc: ExtractedDocument,
        chunks: ChunkCollection
    ) -> Dict[str, List]:
        """Extract facts from document"""
        facts = {
            'financial': []
        }
        
        # Extract from tables
        for table in doc.tables:
            table_facts = self._extract_facts_from_table(table, doc.doc_id)
            facts['financial'].extend(table_facts)
        
        # Extract from text chunks
        for chunk in chunks.chunks:
            if chunk.chunk_type.value == 'paragraph' or chunk.chunk_type.value == 'section':
                # Get page and bbox from first reference
                page = chunk.page_refs[0] if chunk.page_refs else 1
                bbox = chunk.bounding_boxes[0] if chunk.bounding_boxes else None
                
                chunk_facts = self.fact_table.extract_financial_facts_from_text(
                    text=chunk.content,
                    doc_id=doc.doc_id,
                    page=page,
                    bbox=bbox
                )
                facts['financial'].extend(chunk_facts)
        
        return facts
    
    def _extract_facts_from_table(self, table: Table, doc_id: str) -> List[Dict]:
        """
        Extract facts from a table.
        FIXED: Added validation for numeric value conversion.
        """
        facts = []
        
        # Common financial headers
        revenue_keywords = ['revenue', 'sales', 'income', 'turnover']
        profit_keywords = ['profit', 'net income', 'earnings', 'net profit']
        expense_keywords = ['expense', 'cost', 'operating expense']
        
        # Check headers
        headers = [h.lower() for h in table.headers]
        
        for row_idx, row in enumerate(table.rows):
            row_text = ' '.join(row)
            
            # Try to extract year from row
            year = None
            year_match = re.search(r'(20\d{2}|19\d{2})', row_text)
            if year_match:
                year = int(year_match.group())
            
            # Check each cell
            for col_idx, cell in enumerate(row):
                if col_idx < len(headers):
                    header = headers[col_idx]
                    
                    # Check if this looks like a financial fact
                    if any(kw in header for kw in revenue_keywords + profit_keywords + expense_keywords):
                        # Check if cell contains a number
                        number_match = re.search(r'[\$€£]?\s*([\d,]+(?:\.\d+)?)\s*(?:billion|million|thousand|B|M|K)?', cell)
                        if number_match:
                            # FIXED: Validate and clean the number string
                            number_str = number_match.group(1).replace(',', '').strip()
                            
                            # Skip empty strings
                            if not number_str:
                                continue
                            
                            try:
                                numeric_value = float(number_str)
                                
                                # Check for billions/millions suffix in the full cell, not just the matched group
                                suffix_match = re.search(r'(billion|million|thousand|B|M|K)', cell, re.IGNORECASE)
                                if suffix_match:
                                    suffix = suffix_match.group().lower()
                                    if suffix.startswith('b'):
                                        numeric_value *= 1_000_000_000
                                    elif suffix.startswith('m'):
                                        numeric_value *= 1_000_000
                                    elif suffix.startswith('k'):
                                        numeric_value *= 1_000
                                
                                fact_type = 'revenue' if any(kw in header for kw in revenue_keywords) else \
                                           'profit' if any(kw in header for kw in profit_keywords) else \
                                           'expenses'
                                
                                facts.append({
                                    'fact_type': fact_type,
                                    'value': cell,
                                    'numeric_value': numeric_value,
                                    'currency': 'USD' if '$' in cell else 'EUR' if '€' in cell else 'GBP' if '£' in cell else None,
                                    'year': year,
                                    'doc_id': doc_id,
                                    'source_page': table.bbox.page_number if table.bbox else 1,
                                    'source_bbox': json.dumps(table.bbox.to_dict()) if table.bbox else None,
                                    'confidence': 0.95,
                                    'entity': header
                                })
                            except ValueError:
                                # Skip if conversion fails
                                continue
        
        return facts
    
    def _extract_entities_from_document(
        self,
        doc: ExtractedDocument,
        chunks: ChunkCollection
    ) -> List[Dict]:
        """Extract named entities from document"""
        entities = []
        
        # Simple entity extraction patterns
        patterns = {
            'organization': r'\b[A-Z][a-z]+ (?:Bank|Corporation|Inc|Ltd|LLC|Company|Group|Holdings)\b',
            'person': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'date': r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
            'percentage': r'\b\d+(?:\.\d+)?%\b',
            'currency_amount': r'\b[\$€£]\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand)?\b'
        }
        
        for chunk in chunks.chunks:
            if chunk.chunk_type.value in ['paragraph', 'section', 'table']:
                page = chunk.page_refs[0] if chunk.page_refs else 1
                bbox = chunk.bounding_boxes[0] if chunk.bounding_boxes else None
                
                for entity_type, pattern in patterns.items():
                    matches = re.finditer(pattern, chunk.content)
                    for match in matches:
                        entities.append({
                            'entity_type': entity_type,
                            'entity_name': match.group(),
                            'context': chunk.content[max(0, match.start()-50):min(len(chunk.content), match.end()+50)],
                            'doc_id': doc.doc_id,
                            'source_page': page,
                            'source_bbox': bbox,
                            'chunk_id': chunk.chunk_id,
                            'confidence': 0.85
                        })
        
        return entities
    
    def _extract_key_values_from_document(
        self,
        doc: ExtractedDocument,
        chunks: ChunkCollection
    ) -> List[Dict]:
        """Extract key-value pairs from document"""
        kv_pairs = []
        
        # Patterns for key-value pairs
        patterns = [
            (r'([A-Za-z\s]+):\s*([^,\n]+)', 'colon'),
            (r'([A-Za-z\s]+)\s*=\s*([^,\n]+)', 'equals'),
            (r'([A-Za-z\s]+)\s+is\s+([^,\n]+)', 'is')
        ]
        
        for chunk in chunks.chunks:
            if chunk.chunk_type.value in ['paragraph', 'section']:
                page = chunk.page_refs[0] if chunk.page_refs else 1
                bbox = chunk.bounding_boxes[0] if chunk.bounding_boxes else None
                
                for pattern, pattern_type in patterns:
                    matches = re.finditer(pattern, chunk.content)
                    for match in matches:
                        key = match.group(1).strip()
                        value = match.group(2).strip()
                        
                        # Determine value type
                        value_type = 'string'
                        if value.replace('.', '').replace(',', '').isdigit():
                            value_type = 'number'
                        elif '$' in value or '€' in value or '£' in value:
                            value_type = 'currency'
                        elif '%' in value:
                            value_type = 'percentage'
                        
                        kv_pairs.append({
                            'key': key,
                            'value': value,
                            'value_type': value_type,
                            'doc_id': doc.doc_id,
                            'source_page': page,
                            'source_bbox': bbox,
                            'chunk_id': chunk.chunk_id,
                            'confidence': 0.9
                        })
        
        return kv_pairs
    
    def search_with_facts(
        self,
        query: str,
        doc_id: Optional[str] = None,
        use_facts: bool = True
    ) -> Dict[str, Any]:
        """
        Search using both vector store and fact table.
        
        Args:
            query: Search query
            doc_id: Optional document filter
            use_facts: Whether to include fact table results
            
        Returns:
            Combined search results
        """
        results = {
            'vector_results': [],
            'fact_results': [],
            'answer': None
        }
        
        # 1. Vector search
        vector_results = self.vector_store.search(query, n_results=5, doc_id=doc_id)
        results['vector_results'] = vector_results
        
        # 2. Fact table search if query looks like a fact question
        if use_facts:
            fact_results = self._search_facts(query, doc_id)
            results['fact_results'] = fact_results
        
        # 3. Generate combined answer
        results['answer'] = self._generate_answer(query, results)
        
        return results
    
    def _search_facts(self, query: str, doc_id: Optional[str]) -> List[Dict]:
        """Search fact table based on query"""
        fact_results = []
        query_lower = query.lower()
        
        # Check for financial fact queries
        if 'revenue' in query_lower:
            facts = self.fact_table.query_financial_facts(
                fact_type='revenue',
                doc_id=doc_id,
                limit=3
            )
            fact_results.extend(facts)
        
        if 'profit' in query_lower or 'earnings' in query_lower:
            facts = self.fact_table.query_financial_facts(
                fact_type='profit',
                doc_id=doc_id,
                limit=3
            )
            fact_results.extend(facts)
        
        # Check for year-specific queries
        year_match = re.search(r'20\d{2}', query)
        if year_match:
            year = int(year_match.group())
            facts = self.fact_table.query_financial_facts(
                doc_id=doc_id,
                year=year,
                limit=5
            )
            fact_results.extend(facts)
        
        return fact_results
    
    def _generate_answer(self, query: str, results: Dict) -> str:
        """Generate answer from search results"""
        if results['fact_results']:
            # Use fact results for factual answers
            fact = results['fact_results'][0]
            return f"Based on the document, {fact['fact_type']} was {fact['value']}."
        elif results['vector_results']:
            # Use vector results for contextual answers
            return f"Found relevant information: {results['vector_results'][0]['content'][:200]}..."
        else:
            return "No relevant information found."
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics for all databases"""
        return {
            'vector_store': self.vector_store.get_collection_stats(),
            'fact_table': self.fact_table.get_stats()
        }