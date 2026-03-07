#!/usr/bin/env python
"""
Run the complete pipeline on a single document.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryAgent
from src.db.vector_store import VectorStore
from src.db.fact_table import FactTable
from src.db.integration import DatabaseIntegrator
from src.utils.budget_guard import BudgetGuard
from dotenv import load_dotenv
load_dotenv()



async def process_document(pdf_path: str, api_key: str):
    """Process a single document through the entire pipeline"""
    #pdf_path="data/data/2013-E.C-Audit-finding-information.pdf"
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    
    print(f"\n{'='*60}")
    print(f"📄 Document Intelligence Refinery")
    print(f"{'='*60}")
    print(f"Processing: {pdf_path}")
    print(f"{'='*60}\n")
    
    # 1. TRIAGE AGENT
    print("📋 Stage 1: Triage Agent")
    triage = TriageAgent("config/extraction_rules.yaml")
    profile = await triage.profile_document(pdf_path)
    
    print(f"  📌 Document ID: {profile.doc_id}")
    print(f"  📌 Origin Type: {profile.origin_type.value}")
    print(f"  📌 Layout: {profile.layout_complexity.value}")
    print(f"  📌 Domain: {profile.domain_hint.value}")
    print(f"  📌 Pages: {profile.page_count}")
    print(f"  📌 Recommended Strategy: {profile.recommended_strategy.value}")
    
    # Save profile
    await triage.save_profile(profile)
    print(f"  ✅ Profile saved to .refinery/profiles/{profile.doc_id}.json")
    
    # 2. EXTRACTION ROUTER
    print("\n🔧 Stage 2: Extraction Router")
    router = ExtractionRouter(
        openrouter_api_key=api_key,
        ledger_path=".refinery/extraction_ledger.jsonl",
        budget_config={
            "daily_budget_usd": 10.0,
            "per_document_budget_usd": 0.50
        }
    )
    
    doc = await router.extract(pdf_path, profile)
    
    print(f"  ✅ Strategy Used: {doc.extraction_strategy}")
    print(f"  ✅ Confidence: {doc.confidence_score:.2%}")
    print(f"  ✅ Cost: ${doc.cost_estimate_usd:.4f}")
    print(f"  ✅ Time: {doc.extraction_time_seconds:.1f}s")
    print(f"  ✅ Tables Found: {len(doc.tables)}")
    print(f"  ✅ Figures Found: {len(doc.figures)}")
    print(f"  ✅ Text Blocks: {len([b for b in doc.blocks if b.block_type.value == 'text'])}")
    
    # Show sample table if available
    if doc.tables:
        print(f"\n  Sample Table:")
        print(f"    Headers: {doc.tables[0].headers}")
        print(f"    Rows: {len(doc.tables[0].rows)}")
        print(f"    Caption: {doc.tables[0].caption}")
    
    # 3. SEMANTIC CHUNKING
    print("\n📦 Stage 3: Semantic Chunking Engine")
    chunker = ChunkingEngine(max_tokens=512)
    chunks = chunker.chunk(doc)
    
    print(f"  ✅ Total Chunks: {chunks.chunk_count}")
    print(f"  ✅ Total Tokens: {chunks.total_tokens:,}")
    #print(f"  ✅ Avg Tokens/Chunk: {chunks.total_tokens/chunks.chunk_count:.0f}")
    
    # Show chunk distribution
    chunk_types = {}
    for chunk in chunks.chunks:
        chunk_types[chunk.chunk_type.value] = chunk_types.get(chunk.chunk_type.value, 0) + 1
    
    print(f"  📊 Chunk Types:")
    for ctype, count in chunk_types.items():
        print(f"     - {ctype}: {count}")
    
    # Show sample chunks
    print(f"\n  Sample Chunks:")
    for i, chunk in enumerate(chunks.chunks[:3]):
        print(f"    {i+1}. [{chunk.chunk_type.value}] {chunk.content[:100]}...")
    
    # 4. PAGEINDEX BUILDER
    print("\n🌲 Stage 4: PageIndex Builder")
    indexer = PageIndexBuilder()
    page_index = indexer.build_index(chunks)
    
    print(f"  ✅ Root Sections: {len(page_index.root_sections)}")
    print(f"  ✅ Total Sections: {len(page_index.sections)}")
    
    # Show section hierarchy
    def print_section(section_id, indent=0):
        section = page_index.sections[section_id]
        prefix = "  " * indent
        print(f"    {prefix}📄 {section.title} (p.{section.page_start})")
        if section.summary:
            print(f"    {prefix}   📝 {section.summary[:80]}...")
        for child_id in section.child_sections[:3]:  # Limit children shown
            print_section(child_id, indent + 1)
    
    print(f"\n  📚 Section Hierarchy:")
    for root_id in page_index.root_sections[:3]:  # Show first 3 roots
        print_section(root_id)
    
    # Save index
    await indexer.save_index(page_index)
    print(f"\n  ✅ PageIndex saved to .refinery/pageindex/{doc.doc_id}_pageindex.json")
    
    # 5. DATABASE INTEGRATION
    print("\n💾 Stage 5: Database Integration")
    integrator = DatabaseIntegrator(
        vector_store_path=".refinery/vectors",
        fact_table_path=".refinery/facts/facts.db"
    )
    
    stats = integrator.process_document(doc, chunks)
    
    print(f"  ✅ Vector Store: {stats['chunks_added']} chunks added")
    print(f"  ✅ Fact Table: {stats['facts_extracted']} financial facts")
    print(f"  ✅ Entities: {stats['entities_extracted']} named entities")
    print(f"  ✅ Key-Value Pairs: {stats.get('kv_extracted', 0)} extracted")
    
    # 6. QUERY AGENT
    print("\n🤖 Stage 6: Query Agent")
    
    # Load page indices
    page_indices = {doc.doc_id: page_index}
    
    query_agent = QueryAgent(
        vector_store=integrator.vector_store,
        fact_table_path=".refinery/facts/facts.db",
        page_indices=page_indices
    )
    
    print(f"  ✅ Query Agent ready")
    
    # 7. SUMMARY
    print(f"\n{'='*60}")
    print(f"✅ PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\n📁 Output files:")
    print(f"  - Profile: .refinery/profiles/{profile.doc_id}.json")
    print(f"  - Ledger: .refinery/extraction_ledger.jsonl")
    print(f"  - PageIndex: .refinery/pageindex/{doc.doc_id}_pageindex.json")
    print(f"  - Vector Store: .refinery/vectors/")
    print(f"  - Fact Table: .refinery/facts/facts.db")
    
    return {
        "profile": profile,
        "document": doc,
        "chunks": chunks,
        "page_index": page_index,
        "query_agent": query_agent,
        "integrator": integrator
    }


async def run_queries(query_agent, doc_id):
    """Run sample queries to test the system"""
    
    print(f"\n{'='*60}")
    print(f"🔍 TESTING QUERY AGENT")
    print(f"{'='*60}")
    
    test_questions = [
        "What was the revenue?",
        "Show me the profit",
        "What are the key financial highlights?",
        "Tell me about expenses",
        "What year is this report from?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n❓ Q{i}: {question}")
        
        try:
            result = await query_agent.query(question, doc_id)
            
            print(f"📝 A: {result.answer_text[:200]}")
            print(f"📊 Confidence: {result.confidence_score:.2%}")
            print(f"🔍 Status: {result.verification_status}")
            print(f"📚 Citations: {len(result.citations)}")
            
            # Show first citation
            if result.citations:
                cit = result.citations[0]
                print(f"   📄 Source: {cit.document_name}, page {cit.page_number}")
                if cit.extracted_text:
                    print(f"   💬 Excerpt: {cit.extracted_text[:100]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Document Intelligence Refinery on a single document")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY env)")
    parser.add_argument("--query", action="store_true", help="Run test queries after processing")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to disk")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OpenRouter API key required")
        print("   Set OPENROUTER_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Check if file exists
    pdf_path = args.pdf
    if not Path(pdf_path).exists():
        print(f"❌ Error: File not found: {pdf_path}")
        sys.exit(1)
    
    # Run pipeline
    try:
        results = asyncio.run(process_document(pdf_path, api_key))
        
        if args.query:
            asyncio.run(run_queries(results["query_agent"], results["profile"].doc_id))
        
        print(f"\n✨ Done!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()