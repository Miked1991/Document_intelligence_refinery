"""
Query Interface Agent with provenance tracking.
FIXED: Handle both PageIndex objects and dict representations.
"""

from typing import List, Dict, Optional, Any
import sqlite3
from pathlib import Path

from ..models.page_index import PageIndex
from ..models.provenancechain import ProvenanceChain, SourceCitation
from ..models.ldu import LDU


class QueryAgent:
    """
    LangGraph-style agent for querying documents with provenance.
    FIXED: Handle both PageIndex objects and dict representations.
    """
    
    def __init__(
        self,
        vector_store=None,
        fact_table_path: Optional[str] = None,
        page_indices: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize query agent.
        
        Args:
            vector_store: Vector store for semantic search
            fact_table_path: Path to SQLite fact table
            page_indices: Dict mapping doc_id to PageIndex (object or dict)
        """
        self.vector_store = vector_store
        self.page_indices = page_indices or {}
        self.fact_conn = None
        
        if fact_table_path:
            try:
                self.fact_conn = sqlite3.connect(fact_table_path)
                self.fact_conn.row_factory = sqlite3.Row
            except Exception as e:
                print(f"⚠️ Could not connect to fact table: {e}")
    
    def _normalize_page_index(self, index_data: Any) -> Optional[Dict]:
        """
        Normalize page index data to a consistent format.
        
        Args:
            index_data: Either a PageIndex object or a dict
            
        Returns:
            Normalized dictionary representation
        """
        if index_data is None:
            return None
        
        # If it's already a PageIndex object, convert to dict
        if hasattr(index_data, 'sections') and hasattr(index_data, 'root_sections'):
            return {
                'sections': index_data.sections,
                'root_sections': index_data.root_sections,
                'doc_id': index_data.doc_id,
                'total_pages': index_data.total_pages
            }
        
        # If it's a dict, assume it's already in the right format
        if isinstance(index_data, dict):
            # Ensure it has the required structure
            if 'sections' not in index_data:
                # Try to convert from saved JSON format
                if 'sections' in index_data and isinstance(index_data['sections'], dict):
                    # Already has sections
                    pass
                else:
                    # Create empty structure
                    index_data = {
                        'sections': {},
                        'root_sections': [],
                        'doc_id': index_data.get('doc_id', 'unknown'),
                        'total_pages': index_data.get('total_pages', 0)
                    }
            return index_data
        
        return None
    
    async def query(self, question: str, doc_id: Optional[str] = None) -> ProvenanceChain:
        """
        Answer question with provenance.
        
        Args:
            question: Natural language question
            doc_id: Optional document ID to restrict search
            
        Returns:
            Provenance chain with answer and citations
        """
        # Step 1: Navigate with PageIndex (if doc specified)
        relevant_sections = []
        if doc_id and doc_id in self.page_indices:
            index_data = self._normalize_page_index(self.page_indices[doc_id])
            if index_data:
                relevant_sections = self._pageindex_navigate(question, index_data)
        
        # Step 2: Try structured query first (for facts)
        fact_answer = None
        if self.fact_conn and ("revenue" in question.lower() or "profit" in question.lower() or "expense" in question.lower()):
            fact_answer = self._structured_query(question, doc_id)
            if fact_answer and fact_answer.get("confidence", 0) > 0.8:
                return self._fact_to_provenance(fact_answer)
        
        # Step 3: Semantic search
        search_results = await self._semantic_search(
            question, doc_id, relevant_sections
        )
        
        # Step 4: Generate answer
        answer = self._generate_answer(question, search_results)
        
        # Step 5: Build provenance chain
        citations = self._build_citations(search_results)
        
        return ProvenanceChain(
            citations=citations,
            answer_text=answer,
            confidence_score=sum(c.confidence for c in citations) / len(citations) if citations else 0.5,
            verification_status="unverified"
        )
    
    def _pageindex_navigate(self, question: str, index_data: Dict) -> List[str]:
        """
        Navigate PageIndex to find relevant sections.
        
        Args:
            question: User question
            index_data: Normalized page index dictionary
            
        Returns:
            List of relevant section IDs
        """
        if not index_data or 'sections' not in index_data:
            return []
        
        sections = index_data['sections']
        relevant_sections = []
        
        # Simple keyword matching
        keywords = set(question.lower().split())
        
        for section_id, section in sections.items():
            # Handle both dict sections and object sections
            if isinstance(section, dict):
                title = section.get('title', '').lower()
                summary = section.get('summary', '').lower()
                entities = section.get('key_entities', [])
            else:
                # Assume it's a Section object
                title = getattr(section, 'title', '').lower()
                summary = getattr(section, 'summary', '').lower() if hasattr(section, 'summary') else ''
                entities = getattr(section, 'key_entities', [])
            
            # Check title
            if any(k in title for k in keywords):
                relevant_sections.append(section_id)
                continue
            
            # Check summary
            if summary and any(k in summary for k in keywords):
                relevant_sections.append(section_id)
                continue
            
            # Check entities
            if entities:
                entity_text = ' '.join(entities).lower()
                if any(k in entity_text for k in keywords):
                    relevant_sections.append(section_id)
                    continue
        
        return relevant_sections[:3]  # Top 3
    
    async def _semantic_search(
        self,
        question: str,
        doc_id: Optional[str],
        sections: List[str]
    ) -> List[Dict]:
        """Perform semantic search"""
        if not self.vector_store:
            return []
        
        try:
            # Use the vector store's search method
            results = self.vector_store.search(
                query=question,
                n_results=5,
                doc_id=doc_id
            )
            return results
        except Exception as e:
            print(f"⚠️ Vector search failed: {e}")
            return []
    
    def _structured_query(self, question: str, doc_id: Optional[str] = None) -> Optional[Dict]:
        """Query fact table for structured data"""
        if not self.fact_conn:
            return None
        
        cursor = self.fact_conn.cursor()
        
        try:
            # Simple pattern matching
            if "revenue" in question.lower():
                query = """
                    SELECT value, year, source_page, confidence
                    FROM financial_facts
                    WHERE fact_type = 'revenue'
                """
                params = []
                
                if doc_id:
                    query += " AND doc_id = ?"
                    params.append(doc_id)
                
                query += " ORDER BY year DESC LIMIT 1"
                
                cursor.execute(query, params)
                row = cursor.fetchone()
                
                if row:
                    return {
                        "answer": f"Revenue was {row['value']} in {row['year']}",
                        "confidence": row['confidence'],
                        "citations": [{
                            "page": row['source_page'],
                            "document": f"{doc_id}.pdf" if doc_id else "financial_report.pdf"
                        }]
                    }
            
            if "profit" in question.lower() or "earnings" in question.lower():
                query = """
                    SELECT value, year, source_page, confidence
                    FROM financial_facts
                    WHERE fact_type IN ('profit', 'net income', 'earnings')
                """
                params = []
                
                if doc_id:
                    query += " AND doc_id = ?"
                    params.append(doc_id)
                
                query += " ORDER BY year DESC LIMIT 1"
                
                cursor.execute(query, params)
                row = cursor.fetchone()
                
                if row:
                    return {
                        "answer": f"Profit was {row['value']} in {row['year']}",
                        "confidence": row['confidence'],
                        "citations": [{
                            "page": row['source_page'],
                            "document": f"{doc_id}.pdf" if doc_id else "financial_report.pdf"
                        }]
                    }
        except Exception as e:
            print(f"⚠️ Fact table query failed: {e}")
        
        return None
    
    def _fact_to_provenance(self, fact: Dict) -> ProvenanceChain:
        """Convert fact result to provenance chain"""
        citations = []
        for cit in fact.get("citations", []):
            citations.append(SourceCitation(
                document_name=cit.get("document", "unknown.pdf"),
                page_number=cit.get("page", 1),
                content_hash="fact_hash",
                confidence=fact.get("confidence", 0.9)
            ))
        
        return ProvenanceChain(
            citations=citations,
            answer_text=fact["answer"],
            confidence_score=fact["confidence"],
            verification_status="verified"
        )
    
    def _build_citations(self, search_results: List[Dict]) -> List[SourceCitation]:
        """Build citations from search results"""
        citations = []
        
        for result in search_results:
            try:
                citation = SourceCitation(
                    document_name=f"{result.get('doc_id', 'unknown')}.pdf",
                    page_number=result.get("page", 1),
                    bbox=result.get("bbox"),
                    content_hash=result.get("chunk_id", "unknown"),
                    extracted_text=result.get("content", "")[:200],
                    confidence=result.get("score", 0.8)
                )
                citations.append(citation)
            except Exception as e:
                print(f"⚠️ Error building citation: {e}")
                continue
        
        return citations
    
    def _generate_answer(self, question: str, results: List[Dict]) -> str:
        """Generate answer from search results"""
        if not results:
            return "No relevant information found."
        
        # Simple concatenation for demo
        # In production, would use LLM
        answer_parts = []
        for result in results[:2]:  # Limit to first 2 results
            content = result.get("content", "")
            if content:
                answer_parts.append(content)
        
        if not answer_parts:
            return "No relevant information found."
        
        return " ".join(answer_parts)