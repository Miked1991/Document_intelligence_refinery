"""
Fact table implementation for structured data extraction and querying.
Uses SQLite for storing and querying extracted facts from documents.
"""

import sqlite3
import json
import hashlib
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import re


class FactTable:
    """
    SQLite-based fact table for structured data extraction and querying.
    Stores financial, numerical, and entity facts with provenance.
    """
    
    def __init__(self, db_path: str = ".refinery/facts/facts.db"):
        """
        Initialize fact table database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Financial facts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS financial_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    numeric_value REAL,
                    currency TEXT,
                    year INTEGER,
                    quarter INTEGER,
                    period TEXT,
                    entity TEXT,
                    doc_id TEXT NOT NULL,
                    source_page INTEGER,
                    source_bbox TEXT,
                    chunk_id TEXT,
                    content_hash TEXT,
                    confidence REAL DEFAULT 1.0,
                    extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(doc_id, fact_type, value, source_page)
                )
            ''')
            
            # Create indexes for financial facts
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_financial_type ON financial_facts(fact_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_financial_year ON financial_facts(year)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_financial_doc ON financial_facts(doc_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_financial_value ON financial_facts(numeric_value)')
            
            # Named entities table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS named_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_type TEXT NOT NULL,
                    entity_name TEXT NOT NULL,
                    context TEXT,
                    doc_id TEXT NOT NULL,
                    source_page INTEGER,
                    source_bbox TEXT,
                    chunk_id TEXT,
                    confidence REAL DEFAULT 1.0,
                    extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(doc_id, entity_type, entity_name, source_page)
                )
            ''')
            
            # Create indexes for entities
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_type ON named_entities(entity_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_name ON named_entities(entity_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_doc ON named_entities(doc_id)')
            
            # Key-value pairs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS key_value_pairs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    value_type TEXT,
                    doc_id TEXT NOT NULL,
                    source_page INTEGER,
                    source_bbox TEXT,
                    chunk_id TEXT,
                    confidence REAL DEFAULT 1.0,
                    extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(doc_id, key, value, source_page)
                )
            ''')
            
            # Create indexes for key-value pairs
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_kv_key ON key_value_pairs(key)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_kv_value ON key_value_pairs(value)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_kv_doc ON key_value_pairs(doc_id)')
            
            # Document metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS document_metadata (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    page_count INTEGER,
                    extraction_date TIMESTAMP,
                    fact_count INTEGER DEFAULT 0,
                    entity_count INTEGER DEFAULT 0,
                    kv_count INTEGER DEFAULT 0,
                    metadata TEXT
                )
            ''')
            
            # Extraction ledger sync
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS extraction_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    extraction_time TIMESTAMP,
                    facts_extracted INTEGER DEFAULT 0,
                    entities_extracted INTEGER DEFAULT 0,
                    status TEXT,
                    error_message TEXT
                )
            ''')
            
            conn.commit()
    
    # ========== FINANCIAL FACTS ==========
    
    def extract_financial_facts_from_text(self, text: str, doc_id: str, page: int = 1, bbox: Optional[Dict] = None) -> List[Dict]:
        """
        Extract financial facts from text using regex patterns.
        
        Args:
            text: Text to extract from
            doc_id: Document ID
            page: Page number
            bbox: Bounding box coordinates
            
        Returns:
            List of extracted financial facts
        """
        facts = []
        
        # Patterns for financial data
        patterns = {
            'revenue': [
                r'(?:revenue|turnover|sales|income)\s*(?:was|is|:)?\s*[\$€£]?\s*([\d,]+(?:\.\d+)?)\s*(?:billion|million|thousand|B|M|K)?',
                r'[\$€£]\s*([\d,]+(?:\.\d+)?)\s*(?:billion|million|thousand|B|M|K)?\s*(?:in|for)?\s*(?:revenue|turnover|sales)',
            ],
            'profit': [
                r'(?:profit|net income|earnings)\s*(?:was|is|:)?\s*[\$€£]?\s*([\d,]+(?:\.\d+)?)\s*(?:billion|million|thousand|B|M|K)?',
                r'[\$€£]\s*([\d,]+(?:\.\d+)?)\s*(?:billion|million|thousand|B|M|K)?\s*(?:in|for)?\s*(?:profit|net income)',
            ],
            'expenses': [
                r'(?:expenses|costs|operating expenses)\s*(?:was|is|:)?\s*[\$€£]?\s*([\d,]+(?:\.\d+)?)\s*(?:billion|million|thousand|B|M|K)?',
            ],
            'eps': [
                r'(?:EPS|earnings per share)\s*(?:was|is|:)?\s*[\$€£]?\s*([\d,]+(?:\.\d+)?)',
            ],
            'total_assets': [
                r'(?:total assets|assets)\s*(?:was|is|:)?\s*[\$€£]?\s*([\d,]+(?:\.\d+)?)\s*(?:billion|million|thousand|B|M|K)?',
            ],
            'total_liabilities': [
                r'(?:total liabilities|liabilities)\s*(?:was|is|:)?\s*[\$€£]?\s*([\d,]+(?:\.\d+)?)\s*(?:billion|million|thousand|B|M|K)?',
            ]
        }
        
        # Year patterns
        year_pattern = r'(?:FY|fiscal year|year ended|as of)?\s*(?:20\d{2}|19\d{2})'
        
        # Extract year
        year_match = re.search(r'(?:20\d{2}|19\d{2})', text)
        year = int(year_match.group()) if year_match else None
        
        # Quarter pattern
        quarter_pattern = r'Q[1-4]'
        quarter_match = re.search(quarter_pattern, text)
        quarter = int(quarter_match.group()[1]) if quarter_match else None
        
        # Currency detection
        currency = 'USD'  # Default
        if '€' in text:
            currency = 'EUR'
        elif '£' in text:
            currency = 'GBP'
        
        for fact_type, patterns_list in patterns.items():
            for pattern in patterns_list:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Extract numeric value
                    value_str = match.group(1).replace(',', '')
                    try:
                        numeric_value = float(value_str)
                        
                        # Check for billions/millions suffix
                        suffix_match = re.search(r'(billion|million|thousand|B|M|K)', text[match.end():match.end()+10], re.IGNORECASE)
                        if suffix_match:
                            suffix = suffix_match.group().lower()
                            if suffix.startswith('b'):
                                numeric_value *= 1_000_000_000
                            elif suffix.startswith('m'):
                                numeric_value *= 1_000_000
                            elif suffix.startswith('k'):
                                numeric_value *= 1_000
                        
                        facts.append({
                            'fact_type': fact_type,
                            'value': match.group(0),
                            'numeric_value': numeric_value,
                            'currency': currency,
                            'year': year,
                            'quarter': quarter,
                            'doc_id': doc_id,
                            'source_page': page,
                            'source_bbox': json.dumps(bbox) if bbox else None,
                            'confidence': 0.9
                        })
                    except ValueError:
                        continue
        
        return facts
    
    def insert_financial_fact(self, fact: Dict) -> int:
        """Insert a financial fact into the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR IGNORE INTO financial_facts
                (fact_type, value, numeric_value, currency, year, quarter, period,
                 entity, doc_id, source_page, source_bbox, chunk_id, content_hash, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fact.get('fact_type'),
                fact.get('value'),
                fact.get('numeric_value'),
                fact.get('currency'),
                fact.get('year'),
                fact.get('quarter'),
                fact.get('period'),
                fact.get('entity'),
                fact.get('doc_id'),
                fact.get('source_page'),
                fact.get('source_bbox'),
                fact.get('chunk_id'),
                fact.get('content_hash'),
                fact.get('confidence', 1.0)
            ))
            
            return cursor.lastrowid
    
    def insert_financial_facts_batch(self, facts: List[Dict]) -> int:
        """Insert multiple financial facts"""
        count = 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for fact in facts:
                try:
                    cursor.execute('''
                        INSERT OR IGNORE INTO financial_facts
                        (fact_type, value, numeric_value, currency, year, quarter, period,
                         entity, doc_id, source_page, source_bbox, chunk_id, content_hash, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        fact.get('fact_type'),
                        fact.get('value'),
                        fact.get('numeric_value'),
                        fact.get('currency'),
                        fact.get('year'),
                        fact.get('quarter'),
                        fact.get('period'),
                        fact.get('entity'),
                        fact.get('doc_id'),
                        fact.get('source_page'),
                        fact.get('source_bbox'),
                        fact.get('chunk_id'),
                        fact.get('content_hash'),
                        fact.get('confidence', 1.0)
                    ))
                    count += 1
                except Exception as e:
                    print(f"Error inserting fact: {e}")
            
            conn.commit()
        
        return count
    
    def query_financial_facts(
        self,
        fact_type: Optional[str] = None,
        doc_id: Optional[str] = None,
        year: Optional[int] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Query financial facts with filters"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM financial_facts WHERE 1=1"
            params = []
            
            if fact_type:
                query += " AND fact_type = ?"
                params.append(fact_type)
            
            if doc_id:
                query += " AND doc_id = ?"
                params.append(doc_id)
            
            if year:
                query += " AND year = ?"
                params.append(year)
            
            if min_value is not None:
                query += " AND numeric_value >= ?"
                params.append(min_value)
            
            if max_value is not None:
                query += " AND numeric_value <= ?"
                params.append(max_value)
            
            query += f" ORDER BY year DESC, numeric_value DESC LIMIT {limit}"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    # ========== NAMED ENTITIES ==========
    
    def insert_entity(self, entity: Dict) -> int:
        """Insert a named entity"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR IGNORE INTO named_entities
                (entity_type, entity_name, context, doc_id, source_page, source_bbox, chunk_id, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entity.get('entity_type'),
                entity.get('entity_name'),
                entity.get('context'),
                entity.get('doc_id'),
                entity.get('source_page'),
                json.dumps(entity.get('source_bbox')) if entity.get('source_bbox') else None,
                entity.get('chunk_id'),
                entity.get('confidence', 1.0)
            ))
            
            return cursor.lastrowid
    
    def query_entities(
        self,
        entity_type: Optional[str] = None,
        doc_id: Optional[str] = None,
        search_name: Optional[str] = None
    ) -> List[Dict]:
        """Query named entities"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM named_entities WHERE 1=1"
            params = []
            
            if entity_type:
                query += " AND entity_type = ?"
                params.append(entity_type)
            
            if doc_id:
                query += " AND doc_id = ?"
                params.append(doc_id)
            
            if search_name:
                query += " AND entity_name LIKE ?"
                params.append(f"%{search_name}%")
            
            query += " ORDER BY confidence DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    # ========== KEY-VALUE PAIRS ==========
    
    def insert_key_value(self, kv: Dict) -> int:
        """Insert a key-value pair"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR IGNORE INTO key_value_pairs
                (key, value, value_type, doc_id, source_page, source_bbox, chunk_id, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                kv.get('key'),
                kv.get('value'),
                kv.get('value_type'),
                kv.get('doc_id'),
                kv.get('source_page'),
                json.dumps(kv.get('source_bbox')) if kv.get('source_bbox') else None,
                kv.get('chunk_id'),
                kv.get('confidence', 1.0)
            ))
            
            return cursor.lastrowid
    
    def query_key_values(
        self,
        key: Optional[str] = None,
        doc_id: Optional[str] = None,
        value_type: Optional[str] = None
    ) -> List[Dict]:
        """Query key-value pairs"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM key_value_pairs WHERE 1=1"
            params = []
            
            if key:
                query += " AND key = ?"
                params.append(key)
            
            if doc_id:
                query += " AND doc_id = ?"
                params.append(doc_id)
            
            if value_type:
                query += " AND value_type = ?"
                params.append(value_type)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    # ========== DOCUMENT METADATA ==========
    
    def update_document_stats(self, doc_id: str, filename: str, page_count: int = 0):
        """Update document statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count facts for this document
            cursor.execute('SELECT COUNT(*) FROM financial_facts WHERE doc_id = ?', (doc_id,))
            fact_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM named_entities WHERE doc_id = ?', (doc_id,))
            entity_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM key_value_pairs WHERE doc_id = ?', (doc_id,))
            kv_count = cursor.fetchone()[0]
            
            cursor.execute('''
                INSERT OR REPLACE INTO document_metadata
                (doc_id, filename, page_count, extraction_date, fact_count, entity_count, kv_count)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?)
            ''', (doc_id, filename, page_count, fact_count, entity_count, kv_count))
            
            conn.commit()
    
    def get_document_stats(self, doc_id: str) -> Optional[Dict]:
        """Get document statistics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM document_metadata WHERE doc_id = ?', (doc_id,))
            row = cursor.fetchone()
            
            return dict(row) if row else None
    
    # ========== AGGREGATION AND ANALYSIS ==========
    
    def get_financial_summary(self, doc_id: str) -> Dict[str, Any]:
        """Get financial summary for a document"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            summary = {}
            
            # Get revenue
            cursor.execute('''
                SELECT value, numeric_value, year, quarter 
                FROM financial_facts 
                WHERE doc_id = ? AND fact_type = 'revenue'
                ORDER BY year DESC, quarter DESC
                LIMIT 1
            ''', (doc_id,))
            revenue = cursor.fetchone()
            if revenue:
                summary['revenue'] = {
                    'value': revenue[0],
                    'numeric': revenue[1],
                    'year': revenue[2],
                    'quarter': revenue[3]
                }
            
            # Get profit
            cursor.execute('''
                SELECT value, numeric_value, year, quarter 
                FROM financial_facts 
                WHERE doc_id = ? AND fact_type = 'profit'
                ORDER BY year DESC, quarter DESC
                LIMIT 1
            ''', (doc_id,))
            profit = cursor.fetchone()
            if profit:
                summary['profit'] = {
                    'value': profit[0],
                    'numeric': profit[1],
                    'year': profit[2],
                    'quarter': profit[3]
                }
            
            # Calculate profit margin if both exist
            if 'revenue' in summary and 'profit' in summary:
                if summary['revenue']['numeric'] and summary['revenue']['numeric'] > 0:
                    margin = (summary['profit']['numeric'] / summary['revenue']['numeric']) * 100
                    summary['profit_margin'] = f"{margin:.1f}%"
            
            return summary
    
    def compare_documents(self, doc_ids: List[str], fact_type: str = 'revenue') -> List[Dict]:
        """Compare facts across documents"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            placeholders = ','.join(['?'] * len(doc_ids))
            cursor.execute(f'''
                SELECT doc_id, year, value, numeric_value 
                FROM financial_facts 
                WHERE doc_id IN ({placeholders}) AND fact_type = ?
                ORDER BY doc_id, year DESC
            ''', doc_ids + [fact_type])
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    # ========== EXPORT ==========
    
    def export_to_json(self, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Export facts to JSON format"""
        export = {
            'financial_facts': [],
            'named_entities': [],
            'key_value_pairs': [],
            'documents': []
        }
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if doc_id:
                cursor.execute('SELECT * FROM financial_facts WHERE doc_id = ?', (doc_id,))
                export['financial_facts'] = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute('SELECT * FROM named_entities WHERE doc_id = ?', (doc_id,))
                export['named_entities'] = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute('SELECT * FROM key_value_pairs WHERE doc_id = ?', (doc_id,))
                export['key_value_pairs'] = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute('SELECT * FROM document_metadata WHERE doc_id = ?', (doc_id,))
                doc_row = cursor.fetchone()
                if doc_row:
                    export['documents'] = [dict(doc_row)]
            else:
                cursor.execute('SELECT * FROM financial_facts')
                export['financial_facts'] = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute('SELECT * FROM named_entities')
                export['named_entities'] = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute('SELECT * FROM key_value_pairs')
                export['key_value_pairs'] = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute('SELECT * FROM document_metadata')
                export['documents'] = [dict(row) for row in cursor.fetchall()]
        
        return export
    
    def export_to_csv(self, table: str, output_path: str):
        """Export a table to CSV"""
        import csv
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get column names
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Get data
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            
            # Write CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                writer.writerows(rows)
            
            print(f"✅ Exported {len(rows)} rows to {output_path}")
    
    # ========== MAINTENANCE ==========
    
    def vacuum(self):
        """Vacuum database to reclaim space"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")
            print("✅ Database vacuumed")
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            cursor.execute('SELECT COUNT(*) FROM financial_facts')
            stats['financial_facts'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM named_entities')
            stats['named_entities'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM key_value_pairs')
            stats['key_value_pairs'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM document_metadata')
            stats['documents'] = cursor.fetchone()[0]
            
            return stats