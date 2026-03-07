#!/usr/bin/env python
"""
Run the complete pipeline on a single document.
"""

import asyncio
import json
import sys
from pathlib import Path

from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryAgent
from src.db.vector_store import VectorStore
from src.db.fact_table import FactTable
import sqlite3

async def process_document(pdf_path: str, api_key: str):
    """Process a single document through the entire pipeline"""
    
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path}")
    print(f"{'='*60}\n")
    
    # 1. TRIAGE AGENT
    print("📋 Stage 1: Triage Agent")
    triage = TriageAgent("config/extraction_rules.yaml")
    profile = await triage.profile_document(pdf_path)
    
    print(f"  - Document ID: {profile.doc_id}")
    print(f"  - Origin Type: {profile.origin_type.value}")
    print(f"  - Layout: {profile.layout_complexity.value}")
    print(f"  - Domain: {profile.domain_hint.value}")
    print(f"  - Recommended Strategy: {profile.recommended_strategy.value}")
    
    # Save profile
    await triage.save_profile(profile)
    
    # 2. EXTRACTION ROUTER
    print("\n🔧 Stage 2: Extraction Router")
    router = ExtractionRouter(
        openrouter_api_key=api_key,
        ledger_path=".refinery/extraction_ledger.jsonl"
    )
    
    doc = await router.extract(pdf_path, profile)
    
    print(f"  - Strategy Used: {doc.extraction_strategy}")
    print(f"  - Confidence: {doc.confidence_score:.2f}")
    print(f"  - Cost: ${doc.cost_estimate_usd:.4f}")
    print(f"  - Tables Found: {len(doc.tables)}")
    print(f"  - Figures Found: {len(doc.figures)}")
    
    # 3. SEMANTIC CHUNKING
    print("\n📦 Stage 3: Semantic Chunking Engine")
    chunker = ChunkingEngine(max_tokens=512)
    chunks = chunker.chunk(doc)
    
    print(f"  - Total Chunks: {chunks.chunk_count}")
    print(f"  - Total Tokens: {chunks.total_tokens}")
    print(f"  - Avg Tokens/Chunk: {chunks.total_tokens/chunks.chunk_count:.0f}")
    
    # Show sample chunks
    print("\n  Sample Chunks:")
    for i, chunk in enumerate(chunks.chunks[:3]):
        print(f"    {i+1}. Type: {chunk.chunk_type.value}")
        print(f"       Content: {chunk.content[:100]}...")
    
    # 4. PAGEINDEX BUILDER
    print("\n🌲 Stage 4: PageIndex Builder")
    indexer = PageIndexBuilder()
    page_index = indexer.build_index(chunks)
    
    print(f"  - Root Sections: {len(page_index.root_sections)}")
    print(f"  - Total Sections: {len(page_index.sections)}")
    
    # Show section hierarchy
    print("\n  Section Hierarchy:")
    def print_section(section_id, indent=0):
        section = page_index.sections[section_id]
        print(f"    {'  ' * indent}📄 {section.title} (p.{section.page_start})")
        for child_id in section.child_sections:
            print_section(child_id, indent + 1)
    
    for root_id in page_index.root_sections[:3]:  # Show first 3 roots
        print_section(root_id)
    
    # Save index
    await indexer.save_index(page_index)
    
    # 5. SETUP VECTOR STORE AND FACT TABLE
    print("\n💾 Stage 5: Setting up Data Layer")
    
    # Vector store
    vector_store = VectorStore("chromadb", f".refinery/vectors/{doc.doc_id}")
    # In production, you would add chunks to vector store here
    
    # Fact table
    fact_table = FactTable(f".refinery/facts/{doc.doc_id}.db")
    fact_table.create_tables()
    # Extract and insert facts
    
    # 6. QUERY AGENT
    print("\n🤖 Stage 6: Query Agent Ready")
    
    # Load page indices
    page_indices = {doc.doc_id: page_index}
    
    query_agent = QueryAgent(
        vector_store=vector_store,
        fact_table_path=f".refinery/facts/{doc.doc_id}.db",
        page_indices=page_indices
    )
    
    print("\n✅ Pipeline complete!")
    print(f"\nOutput files:")
    print(f"  - Profile: .refinery/profiles/{profile.doc_id}.json")
    print(f"  - Ledger: .refinery/extraction_ledger.jsonl")
    print(f"  - PageIndex: .refinery/pageindex/{doc.doc_id}_pageindex.json")
    
    return {
        "profile": profile,
        "document": doc,
        "chunks": chunks,
        "page_index": page_index,
        "query_agent": query_agent
    }

async def run_queries(query_agent, doc_id):
    """Run sample queries to test the system"""
    
    print("\n" + "="*60)
    print("TESTING QUERY AGENT")
    print("="*60)
    
    test_questions = [
        "What was the revenue for 2023?",
        "Show me the balance sheet",
        "What are the key financial highlights?",
        "Tell me about expenses"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        
        try:
            result = await query_agent.query(question, doc_id)
            
            print(f"📝 Answer: {result.answer_text[:200]}...")
            print(f"📊 Confidence: {result.confidence_score:.2f}")
            print(f"🔍 Status: {result.verification_status}")
            print(f"📚 Citations: {len(result.citations)}")
            
            # Show first citation
            if result.citations:
                cit = result.citations[0]
                print(f"   Source: {cit.document_name}, page {cit.page_number}")
                if cit.extracted_text:
                    print(f"   Excerpt: {cit.extracted_text[:100]}...")
        except Exception as e:
            print(f"   Error: {e}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Document Intelligence Refinery")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY env)")
    parser.add_argument("--query", action="store_true", help="Run test queries after processing")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OpenRouter API key required")
        print("Set OPENROUTER_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Run pipeline
    pdf_path = args.pdf
    if not Path(pdf_path).exists():
        print(f"❌ Error: File not found: {pdf_path}")
        sys.exit(1)
    
    results = asyncio.run(process_document(pdf_path, api_key))
    
    if args.query:
        asyncio.run(run_queries(results["query_agent"], results["profile"].doc_id))

if __name__ == "__main__":
    import os
    main()