#!/usr/bin/env python
"""
Interactive demo with multi-PDF input support.
Process multiple PDFs at once and query across all documents interactively.
FIXED: Proper PageIndex loading and error handling.
"""

import asyncio
import json
import sys
import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryAgent
from src.db.integration import DatabaseIntegrator
from src.db.fact_table import FactTable
from src.db.vector_store import VectorStore

# Try to import PageIndex model
try:
    from src.models.page_index import PageIndex
    HAS_PAGEINDEX_MODEL = True
except ImportError:
    HAS_PAGEINDEX_MODEL = False

# Try to import rich for beautiful output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich import print as rprint
    from rich.markdown import Markdown
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None


class DocumentInfo:
    """Information about a processed document"""
    def __init__(self, doc_id: str, filename: str, date: datetime, page_count: int, fact_count: int = 0):
        self.doc_id = doc_id
        self.filename = filename
        self.date = date
        self.page_count = page_count
        self.fact_count = fact_count
        self.status = "processed"
        self.page_index = None  # Will be loaded on demand


class BatchProcessor:
    """Process multiple PDFs in batch"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.triage = TriageAgent("config/extraction_rules.yaml")
        self.router = ExtractionRouter(openrouter_api_key=api_key)
        self.chunker = ChunkingEngine(max_tokens=512)
        self.indexer = PageIndexBuilder()
        self.integrator = DatabaseIntegrator()
        
        self.results = {}
        self.processed_docs = []
        self.failed_docs = []
    
    async def process_single(self, pdf_path: Path, progress=None, task_id=None) -> Optional[DocumentInfo]:
        """Process a single PDF file"""
        try:
            if HAS_RICH and progress and task_id is not None:
                progress.update(task_id, description=f"[cyan]Processing: {pdf_path.name}")
            
            # Step 1: Triage
            profile = await self.triage.profile_document(str(pdf_path))
            await self.triage.save_profile(profile)
            
            if HAS_RICH and progress and task_id is not None:
                progress.update(task_id, advance=20, description=f"[cyan]Extracting: {pdf_path.name}")
            
            # Step 2: Extraction
            doc = await self.router.extract(str(pdf_path), profile)
            
            if HAS_RICH and progress and task_id is not None:
                progress.update(task_id, advance=20, description=f"[cyan]Chunking: {pdf_path.name}")
            
            # Step 3: Chunking
            chunks = self.chunker.chunk(doc)
            
            if HAS_RICH and progress and task_id is not None:
                progress.update(task_id, advance=20, description=f"[cyan]Building index: {pdf_path.name}")
            
            # Step 4: PageIndex
            page_index = self.indexer.build_index(chunks)
            await self.indexer.save_index(page_index)
            
            if HAS_RICH and progress and task_id is not None:
                progress.update(task_id, advance=20, description=f"[cyan]Integrating: {pdf_path.name}")
            
            # Step 5: Database integration
            self.integrator.process_document(doc, chunks)
            
            # Store result
            doc_info = DocumentInfo(
                doc_id=profile.doc_id,
                filename=pdf_path.name,
                date=datetime.now(),
                page_count=profile.page_count,
                fact_count=len(doc.tables)
            )
            
            if HAS_RICH and progress and task_id is not None:
                progress.update(task_id, advance=20, description=f"[green]Completed: {pdf_path.name}")
            
            return doc_info
            
        except Exception as e:
            if HAS_RICH and progress and task_id is not None:
                progress.update(task_id, description=f"[red]Failed: {pdf_path.name} - {str(e)[:50]}")
            print(f"❌ Error processing {pdf_path.name}: {e}")
            return None
    
    async def process_multiple(self, pdf_paths: List[Path]) -> List[DocumentInfo]:
        """Process multiple PDF files"""
        self.processed_docs = []
        self.failed_docs = []
        
        if HAS_RICH:
            # Create progress display
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                tasks = []
                for pdf_path in pdf_paths:
                    task_id = progress.add_task(
                        f"[cyan]Pending: {pdf_path.name}",
                        total=100
                    )
                    tasks.append((pdf_path, task_id))
                
                for pdf_path, task_id in tasks:
                    doc_info = await self.process_single(pdf_path, progress, task_id)
                    if doc_info:
                        self.processed_docs.append(doc_info)
                    else:
                        self.failed_docs.append(pdf_path.name)
        else:
            # Simple console output
            total = len(pdf_paths)
            for i, pdf_path in enumerate(pdf_paths, 1):
                print(f"\n[{i}/{total}] Processing: {pdf_path.name}")
                doc_info = await self.process_single(pdf_path)
                if doc_info:
                    self.processed_docs.append(doc_info)
                    print(f"  ✅ Completed: {pdf_path.name}")
                else:
                    self.failed_docs.append(pdf_path.name)
                    print(f"  ❌ Failed: {pdf_path.name}")
        
        return self.processed_docs
    
    def get_summary(self) -> Dict:
        """Get processing summary"""
        return {
            'total': len(self.processed_docs) + len(self.failed_docs),
            'successful': len(self.processed_docs),
            'failed': len(self.failed_docs),
            'failed_files': self.failed_docs,
            'documents': self.processed_docs
        }


class InteractiveQuerySession:
    """Manages interactive query session with multi-document support"""
    
    def __init__(self, integrator: DatabaseIntegrator, page_indices_dir: str = ".refinery/pageindex"):
        self.integrator = integrator
        self.page_indices_dir = Path(page_indices_dir)
        self.documents: Dict[str, DocumentInfo] = {}
        self.current_doc_id: Optional[str] = None
        self.query_agent = None
        self.search_all = False
        
        # Load all available documents
        self._load_documents()
        
        # Initialize query agent with all documents
        self._init_query_agent()
    
    def _load_documents(self):
        """Load all processed documents from fact table and page indices"""
        # Load from page indices directory
        if self.page_indices_dir.exists():
            for index_file in self.page_indices_dir.glob("*_pageindex.json"):
                try:
                    with open(index_file, 'r') as f:
                        data = json.load(f)
                    
                    doc_id = data.get('doc_id')
                    filename = data.get('filename', f"{doc_id}.pdf")
                    
                    # Try to get creation time from file
                    mtime = datetime.fromtimestamp(index_file.stat().st_mtime)
                    
                    self.documents[doc_id] = DocumentInfo(
                        doc_id=doc_id,
                        filename=filename,
                        date=mtime,
                        page_count=data.get('total_pages', 0)
                    )
                except Exception as e:
                    if HAS_RICH:
                        console.print(f"[dim]Error loading {index_file.name}: {e}[/dim]")
        
        # Also try to load from fact table metadata
        try:
            # This would need a method to get document list from fact table
            pass
        except:
            pass
    
    def _load_page_index(self, doc_id: str) -> Optional[Any]:
        """Load page index for a document, returning either PageIndex object or dict"""
        index_path = self.page_indices_dir / f"{doc_id}_pageindex.json"
        if not index_path.exists():
            return None
        
        try:
            with open(index_path, 'r') as f:
                data = json.load(f)
            
            # Try to create PageIndex object if model is available
            if HAS_PAGEINDEX_MODEL:
                try:
                    from src.models.page_index import PageIndex
                    return PageIndex(**data)
                except Exception as e:
                    if HAS_RICH:
                        console.print(f"[dim]Could not create PageIndex object for {doc_id}, using dict: {e}[/dim]")
                    return data
            else:
                return data
        except Exception as e:
            if HAS_RICH:
                console.print(f"[dim]Error loading page index for {doc_id}: {e}[/dim]")
            return None
    
    def _init_query_agent(self):
        """Initialize query agent with all document page indices"""
        page_indices = {}
        
        # Load page indices for all documents
        for doc_id in self.documents:
            index_data = self._load_page_index(doc_id)
            if index_data:
                page_indices[doc_id] = index_data
        
        self.query_agent = QueryAgent(
            vector_store=self.integrator.vector_store,
            fact_table_path=str(Path(".refinery/facts/facts.db")),
            page_indices=page_indices
        )
        
        # Set default document if available
        if self.documents and not self.current_doc_id:
            self.current_doc_id = list(self.documents.keys())[0]
    
    def add_documents(self, docs: List[DocumentInfo]):
        """Add newly processed documents to the session"""
        for doc in docs:
            self.documents[doc.doc_id] = doc
        
        # Re-initialize query agent with updated documents
        self._init_query_agent()
    
    def list_documents(self):
        """List all available documents"""
        if not self.documents:
            if HAS_RICH:
                console.print("[yellow]No documents available. Process some PDFs first.[/yellow]")
            else:
                print("No documents available. Process some PDFs first.")
            return
        
        if HAS_RICH:
            table = Table(title="📚 Available Documents", show_header=True, box=None)
            table.add_column("#", style="cyan", width=4)
            table.add_column("Document ID", style="green", width=20)
            table.add_column("Filename", style="white", width=40)
            table.add_column("Pages", style="yellow", justify="right", width=6)
            table.add_column("Processed", style="blue", width=16)
            table.add_column("Facts", style="magenta", justify="right", width=6)
            table.add_column("Status", style="red", width=10)
            
            for idx, (doc_id, info) in enumerate(self.documents.items(), 1):
                status = ""
                if self.search_all:
                    status = "🌍 ALL MODE"
                elif doc_id == self.current_doc_id:
                    status = "✅ CURRENT"
                
                # Truncate long strings
                doc_id_display = doc_id[:18] + "..." if len(doc_id) > 20 else doc_id
                filename_display = info.filename[:38] + "..." if len(info.filename) > 40 else info.filename
                
                table.add_row(
                    str(idx),
                    doc_id_display,
                    filename_display,
                    str(info.page_count),
                    info.date.strftime("%Y-%m-%d %H:%M"),
                    str(info.fact_count),
                    status
                )
            
            console.print(table)
        else:
            print("\n📚 Available Documents:")
            for idx, (doc_id, info) in enumerate(self.documents.items(), 1):
                current = " (current)" if doc_id == self.current_doc_id else ""
                all_mode = " (all mode)" if self.search_all else ""
                print(f"  {idx}. {doc_id} - {info.filename} [{info.page_count} pages]{current}{all_mode}")
    
    def show_current_document(self):
        """Show currently selected document"""
        if self.search_all:
            if HAS_RICH:
                panel = Panel(
                    f"[bold green]Searching across ALL documents[/bold green]\n"
                    f"Total documents: {len(self.documents)}",
                    title="🌍 Global Search Mode",
                    border_style="green"
                )
                console.print(panel)
            else:
                print(f"\n🌍 Global Search Mode - Searching across all {len(self.documents)} documents")
            return
        
        if self.current_doc_id and self.current_doc_id in self.documents:
            info = self.documents[self.current_doc_id]
            if HAS_RICH:
                panel = Panel(
                    f"[bold green]{info.filename}[/bold green]\n"
                    f"ID: {info.doc_id}\n"
                    f"Pages: {info.page_count}\n"
                    f"Processed: {info.date.strftime('%Y-%m-%d %H:%M')}\n"
                    f"Facts: {info.fact_count}",
                    title="📄 Current Document",
                    border_style="green"
                )
                console.print(panel)
            else:
                print(f"\n📄 Current Document: {info.filename}")
                print(f"  ID: {info.doc_id}")
                print(f"  Pages: {info.page_count}")
                print(f"  Processed: {info.date}")
        else:
            if HAS_RICH:
                console.print("[yellow]No document currently selected[/yellow]")
            else:
                print("No document currently selected")
    
    def switch_document(self, selector: str) -> bool:
        """Switch to a different document by index, ID, or filename"""
        # Turn off global search mode when switching to specific document
        self.search_all = False
        
        # Try by index
        if selector.isdigit():
            idx = int(selector) - 1
            doc_list = list(self.documents.keys())
            if 0 <= idx < len(doc_list):
                self.current_doc_id = doc_list[idx]
                return True
        
        # Try by document ID
        if selector in self.documents:
            self.current_doc_id = selector
            return True
        
        # Try by filename (partial match)
        selector_lower = selector.lower()
        for doc_id, info in self.documents.items():
            if selector_lower in info.filename.lower():
                self.current_doc_id = doc_id
                return True
        
        return False
    
    def set_search_all(self, enabled: bool = True):
        """Set global search mode"""
        self.search_all = enabled
    
    def parse_document_from_query(self, question: str) -> Tuple[Optional[str], str]:
        """Extract document reference from natural language query"""
        # Pattern: "in [document name]" at the end
        in_pattern = r'\s+in\s+([a-zA-Z0-9_\-\.]+(?:\s+[a-zA-Z0-9_\-\.]+)*)\s*$'
        in_match = re.search(in_pattern, question, re.IGNORECASE)
        
        if in_match:
            doc_ref = in_match.group(1).strip()
            clean_question = question[:in_match.start()].strip()
            
            # Try to match the reference to a document
            for doc_id, info in self.documents.items():
                if doc_ref.lower() in info.filename.lower() or doc_ref.lower() in doc_id.lower():
                    return doc_id, clean_question
            
            # If no match, try fuzzy matching
            for doc_id, info in self.documents.items():
                if doc_ref.lower() == doc_id.lower()[:len(doc_ref)]:
                    return doc_id, clean_question
        
        # Pattern: "from [document name]"
        from_pattern = r'\s+from\s+([a-zA-Z0-9_\-\.]+(?:\s+[a-zA-Z0-9_\-\.]+)*)\s*$'
        from_match = re.search(from_pattern, question, re.IGNORECASE)
        
        if from_match:
            doc_ref = from_match.group(1).strip()
            clean_question = question[:from_match.start()].strip()
            
            for doc_id, info in self.documents.items():
                if doc_ref.lower() in info.filename.lower() or doc_ref.lower() in doc_id.lower():
                    return doc_id, clean_question
        
        return None, question
    
    def parse_comparison_query(self, question: str) -> Optional[Dict]:
        """Parse comparison queries between documents"""
        # Pattern: "compare [fact] between [doc1] and [doc2]"
        compare_pattern = r'compare\s+([a-zA-Z\s]+?)\s+between\s+([a-zA-Z0-9_\-\.]+(?:\s+[a-zA-Z0-9_\-\.]+)*)\s+and\s+([a-zA-Z0-9_\-\.]+(?:\s+[a-zA-Z0-9_\-\.]+)*)$'
        compare_match = re.search(compare_pattern, question, re.IGNORECASE)
        
        if compare_match:
            fact = compare_match.group(1).strip()
            doc1_ref = compare_match.group(2).strip()
            doc2_ref = compare_match.group(3).strip()
            
            doc1_id = None
            doc2_id = None
            
            # Match documents
            for doc_id, info in self.documents.items():
                if doc1_ref.lower() in info.filename.lower() or doc1_ref.lower() in doc_id.lower():
                    doc1_id = doc_id
                if doc2_ref.lower() in info.filename.lower() or doc2_ref.lower() in doc_id.lower():
                    doc2_id = doc_id
            
            if doc1_id and doc2_id:
                return {
                    'fact': fact,
                    'doc1': doc1_id,
                    'doc2': doc2_id,
                    'question': question
                }
        
        return None
    
    async def process_comparison(self, comparison: Dict) -> str:
        """Process a comparison query"""
        fact = comparison['fact']
        doc1 = comparison['doc1']
        doc2 = comparison['doc2']
        
        # Get facts from both documents
        facts1 = self.integrator.fact_table.query_financial_facts(
            doc_id=doc1,
            limit=5
        )
        
        facts2 = self.integrator.fact_table.query_financial_facts(
            doc_id=doc2,
            limit=5
        )
        
        # Format comparison result
        if HAS_RICH:
            result = f"## Comparison: {fact}\n\n"
        else:
            result = f"\nComparison: {fact}\n\n"
        
        doc1_name = self.documents[doc1].filename if doc1 in self.documents else doc1
        doc2_name = self.documents[doc2].filename if doc2 in self.documents else doc2
        
        result += f"**{doc1_name}**\n"
        found1 = False
        for f in facts1[:3]:
            if fact.lower() in f.get('fact_type', '').lower():
                result += f"  • {f.get('fact_type')}: {f.get('value')} ({f.get('year')})\n"
                found1 = True
        if not found1:
            result += "  No matching facts found\n"
        
        result += f"\n**{doc2_name}**\n"
        found2 = False
        for f in facts2[:3]:
            if fact.lower() in f.get('fact_type', '').lower():
                result += f"  • {f.get('fact_type')}: {f.get('value')} ({f.get('year')})\n"
                found2 = True
        if not found2:
            result += "  No matching facts found\n"
        
        return result
    
    async def process_query(self, question: str):
        """Process a query with full multi-document support"""
        
        # Check for comparison query
        comparison = self.parse_comparison_query(question)
        if comparison:
            result_text = await self.process_comparison(comparison)
            if HAS_RICH:
                console.print(Panel(Markdown(result_text), title="Comparison", border_style="blue"))
            else:
                print(result_text)
            return
        
        # Extract document from query
        doc_id, clean_question = self.parse_document_from_query(question)
        
        # If document specified in query, use that
        if doc_id:
            if HAS_RICH and doc_id in self.documents:
                console.print(f"[dim]Searching in: {self.documents[doc_id].filename}[/dim]")
            result = await self.query_agent.query(clean_question, doc_id)
            self._display_result(result)
        
        # Otherwise use current mode
        elif self.search_all:
            if HAS_RICH:
                console.print(f"[dim]Searching across ALL {len(self.documents)} documents[/dim]")
            result = await self.query_agent.query(question, None)  # Search all
            self._display_result(result)
        
        # Use current document if set
        elif self.current_doc_id:
            if HAS_RICH and self.current_doc_id in self.documents:
                console.print(f"[dim]Searching in: {self.documents[self.current_doc_id].filename}[/dim]")
            result = await self.query_agent.query(question, self.current_doc_id)
            self._display_result(result)
        
        # If no context, search all
        else:
            if HAS_RICH:
                console.print(f"[dim]Searching across ALL {len(self.documents)} documents[/dim]")
            result = await self.query_agent.query(question, None)
            self._display_result(result)
    
    def _display_result(self, result):
        """Display query result"""
        if HAS_RICH:
            # Answer panel
            console.print(Panel(result.answer_text, title="Answer", border_style="green"))
            
            # Confidence
            confidence_color = "green" if result.confidence_score > 0.7 else "yellow" if result.confidence_score > 0.4 else "red"
            console.print(f"Confidence: [{confidence_color}]{result.confidence_score:.2%}[/{confidence_color}] | Verification: {result.verification_status}")
            
            # Citations
            if result.citations:
                citations_table = Table(title="Sources", show_header=True, box=None)
                citations_table.add_column("#", style="cyan", width=4)
                citations_table.add_column("Document", style="white")
                citations_table.add_column("Page", style="green", justify="right", width=6)
                citations_table.add_column("Excerpt", style="dim")
                
                for i, cit in enumerate(result.citations[:5], 1):
                    excerpt = cit.extracted_text[:80] + "..." if cit.extracted_text and len(cit.extracted_text) > 80 else cit.extracted_text or ""
                    citations_table.add_row(
                        str(i),
                        cit.document_name,
                        str(cit.page_number),
                        excerpt
                    )
                
                console.print(citations_table)
                
                if len(result.citations) > 5:
                    console.print(f"[dim]... and {len(result.citations) - 5} more sources[/dim]")
        else:
            print(f"\nAnswer: {result.answer_text}")
            print(f"Confidence: {result.confidence_score:.2%} | Verification: {result.verification_status}")
            
            if result.citations:
                print("\nSources:")
                for i, cit in enumerate(result.citations[:3], 1):
                    print(f"  {i}. {cit.document_name}, page {cit.page_number}")
                    if cit.extracted_text:
                        print(f"     \"{cit.extracted_text[:100]}...\"")
    
    def show_help(self):
        """Show interactive help"""
        if HAS_RICH:
            help_table = Table(title="📖 Interactive Query Help", show_header=True, box=None)
            help_table.add_column("Command/Pattern", style="cyan", width=30)
            help_table.add_column("Description", style="white", width=50)
            
            help_table.add_row("/list, /ls", "List all available documents")
            help_table.add_row("/use [index|id|name]", "Switch to a specific document")
            help_table.add_row("/current, /now", "Show currently selected document")
            help_table.add_row("/all, /global", "Search across ALL documents")
            help_table.add_row("/stats", "Show database statistics")
            help_table.add_row("/help, /?", "Show this help")
            help_table.add_row("/exit, /quit", "Exit interactive mode")
            help_table.add_row("", "")
            help_table.add_row("[question] in [document]", "Query specific document (e.g., 'revenue in cbe_2023')")
            help_table.add_row("[question] from [document]", "Query specific document (e.g., 'profit from tax_2022')")
            help_table.add_row("compare [fact] between [doc1] and [doc2]", "Compare across documents")
            
            console.print(help_table)
        else:
            print("\n📖 Interactive Query Help")
            print("=" * 50)
            print("Document Selection Commands:")
            print("  /list                 - List all documents")
            print("  /use [index|id|name]  - Switch document")
            print("  /current              - Show current document")
            print("  /all                  - Search all documents")
            print("\nQuery Patterns:")
            print("  [question] in [doc]    - Query specific document")
            print("  [question] from [doc]  - Query specific document")
            print("  compare [fact] between [doc1] and [doc2] - Cross-document comparison")
            print("\nOther Commands:")
            print("  /stats                 - Show database stats")
            print("  /help                  - This help")
            print("  /exit                  - Exit")
    
    def show_stats(self):
        """Show database statistics"""
        stats = self.integrator.get_database_stats()
        
        if HAS_RICH:
            stats_table = Table(title="📊 Database Statistics", box=None)
            stats_table.add_column("Component", style="cyan")
            stats_table.add_column("Value", style="white", justify="right")
            
            stats_table.add_row("Total Documents", str(len(self.documents)))
            stats_table.add_row("Vector Store Chunks", str(stats['vector_store'].get('total_chunks', 0)))
            stats_table.add_row("Documents in Vector Store", str(stats['vector_store'].get('documents', 0)))
            stats_table.add_row("Financial Facts", str(stats['fact_table'].get('financial_facts', 0)))
            stats_table.add_row("Named Entities", str(stats['fact_table'].get('named_entities', 0)))
            stats_table.add_row("Key-Value Pairs", str(stats['fact_table'].get('key_value_pairs', 0)))
            
            console.print(stats_table)
        else:
            print("\n📊 Database Statistics:")
            print(f"  Total Documents: {len(self.documents)}")
            print(f"  Vector Store Chunks: {stats['vector_store'].get('total_chunks', 0)}")
            print(f"  Documents in Vector Store: {stats['vector_store'].get('documents', 0)}")
            print(f"  Financial Facts: {stats['fact_table'].get('financial_facts', 0)}")
            print(f"  Named Entities: {stats['fact_table'].get('named_entities', 0)}")
            print(f"  Key-Value Pairs: {stats['fact_table'].get('key_value_pairs', 0)}")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Intelligence Refinery - Multi-PDF Demo")
    parser.add_argument("--pdf", nargs="+", help="PDF file(s) to process (can specify multiple)")
    parser.add_argument("--pdf-dir", help="Directory containing PDF files to process")
    parser.add_argument("--pattern", default="*.pdf", help="File pattern when using --pdf-dir (default: *.pdf)")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY env)")
    parser.add_argument("--no-rich", action="store_true", help="Disable rich formatting")
    parser.add_argument("--max", type=int, default=0, help="Maximum number of PDFs to process (0 = all)")
    
    args = parser.parse_args()
    
    # Handle rich formatting
    global HAS_RICH, console
    if args.no_rich:
        HAS_RICH = False
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OpenRouter API key required")
        print("   Set OPENROUTER_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Collect PDF files
    pdf_files = []
    
    if args.pdf:
        for pdf_arg in args.pdf:
            pdf_path = Path(pdf_arg)
            if pdf_path.exists():
                pdf_files.append(pdf_path)
            else:
                print(f"⚠️ Warning: File not found: {pdf_arg}")
    
    if args.pdf_dir:
        dir_path = Path(args.pdf_dir)
        if dir_path.exists() and dir_path.is_dir():
            found_files = list(dir_path.glob(args.pattern))
            pdf_files.extend(found_files)
            print(f"📂 Found {len(found_files)} PDFs in {args.pdf_dir}")
        else:
            print(f"⚠️ Warning: Directory not found: {args.pdf_dir}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_pdfs = []
    for pdf in pdf_files:
        if pdf.name not in seen:
            seen.add(pdf.name)
            unique_pdfs.append(pdf)
    
    pdf_files = unique_pdfs
    
    # Apply max limit
    if args.max > 0 and len(pdf_files) > args.max:
        pdf_files = pdf_files[:args.max]
        print(f"📊 Limiting to first {args.max} PDFs")
    
    if not pdf_files:
        print("❌ No PDF files to process")
        print("\nUsage examples:")
        print("  python demo.py --pdf doc1.pdf doc2.pdf doc3.pdf")
        print("  python demo.py --pdf-dir ./data/input --pattern '*.pdf'")
        print("  python demo.py --pdf report1.pdf --pdf-dir ./reports --max 5")
        sys.exit(1)
    
    # Display processing plan
    if HAS_RICH:
        console.print(Panel.fit(
            f"[bold cyan]Document Intelligence Refinery - Multi-PDF Demo[/bold cyan]\n"
            f"Processing {len(pdf_files)} PDF files",
            border_style="cyan"
        ))
        
        file_table = Table(show_header=True, box=None)
        file_table.add_column("#", style="cyan", width=4)
        file_table.add_column("Filename", style="white")
        file_table.add_column("Size", style="yellow", justify="right")
        
        for i, pdf in enumerate(pdf_files[:10], 1):
            size = pdf.stat().st_size / 1024
            file_table.add_row(str(i), pdf.name, f"{size:.1f} KB")
        
        if len(pdf_files) > 10:
            file_table.add_row("...", f"and {len(pdf_files) - 10} more", "")
        
        console.print(file_table)
    else:
        print(f"\n📚 Processing {len(pdf_files)} PDF files:")
        for i, pdf in enumerate(pdf_files, 1):
            print(f"  {i}. {pdf.name}")
    
    # Confirm with user
    if HAS_RICH:
        proceed = Confirm.ask("\nProceed with processing?")
    else:
        response = input("\nProceed with processing? (y/N): ").strip().lower()
        proceed = response in ['y', 'yes']
    
    if not proceed:
        print("❌ Cancelled by user")
        sys.exit(0)
    
    # Process PDFs
    processor = BatchProcessor(api_key)
    processed = await processor.process_multiple(pdf_files)
    
    # Show summary
    summary = processor.get_summary()
    
    if HAS_RICH:
        summary_panel = Panel(
            f"[green]✅ Successful: {summary['successful']}[/green]\n"
            f"[red]❌ Failed: {summary['failed']}[/red]",
            title="Processing Complete",
            border_style="green" if summary['failed'] == 0 else "yellow"
        )
        console.print(summary_panel)
        
        if summary['failed'] > 0:
            console.print("[red]Failed files:[/red]")
            for f in summary['failed_files']:
                console.print(f"  • {f}")
    else:
        print(f"\n{'='*50}")
        print(f"Processing Complete")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        if summary['failed'] > 0:
            print("Failed files:")
            for f in summary['failed_files']:
                print(f"  • {f}")
    
    if summary['successful'] == 0:
        print("❌ No documents were successfully processed")
        sys.exit(1)
    
    # Initialize query session
    query_session = InteractiveQuerySession(processor.integrator)
    query_session.add_documents(processed)
    
    # Interactive query mode
    if HAS_RICH:
        console.print("\n" + "="*60)
        console.print("[bold cyan]🔍 INTERACTIVE QUERY MODE[/bold cyan]")
        console.print("[dim]Type /help for commands, /exit to quit[/dim]\n")
    else:
        print("\n" + "="*60)
        print("🔍 INTERACTIVE QUERY MODE")
        print("Type /help for commands, /exit to quit\n")
    
    # Show available documents
    query_session.list_documents()
    query_session.show_current_document()
    
    while True:
        # Get user input
        if HAS_RICH:
            question = Prompt.ask("\n[bold yellow]Your question[/bold yellow]")
        else:
            question = input("\nYour question > ").strip()
        
        if not question:
            continue
        
        # Handle commands
        cmd = question.lower()
        
        if cmd in ['/exit', '/quit', 'exit', 'quit']:
            if HAS_RICH:
                console.print("[green]Exiting query mode. Goodbye![/green]")
            else:
                print("Exiting query mode. Goodbye!")
            break
        
        if cmd in ['/help', '/?', 'help', '?']:
            query_session.show_help()
            continue
        
        if cmd in ['/list', '/ls']:
            query_session.list_documents()
            continue
        
        if cmd in ['/current', '/now']:
            query_session.show_current_document()
            continue
        
        if cmd in ['/all', '/global']:
            query_session.set_search_all(True)
            if HAS_RICH:
                console.print("[green]Now searching across ALL documents[/green]")
            else:
                print("Now searching across ALL documents")
            continue
        
        if cmd in ['/stats']:
            query_session.show_stats()
            continue
        
        if cmd.startswith('/use'):
            parts = cmd.split(maxsplit=1)
            if len(parts) == 2:
                selector = parts[1].strip()
                if query_session.switch_document(selector):
                    if HAS_RICH:
                        console.print(f"[green]Now using: {query_session.current_doc_id}[/green]")
                    else:
                        print(f"Now using: {query_session.current_doc_id}")
                    query_session.show_current_document()
                else:
                    if HAS_RICH:
                        console.print(f"[red]Document not found: {selector}[/red]")
                    else:
                        print(f"Document not found: {selector}")
            else:
                if HAS_RICH:
                    console.print("[yellow]Usage: /use [index|id|name][/yellow]")
                else:
                    print("Usage: /use [index|id|name]")
            continue
        
        # Process natural language query
        await query_session.process_query(question)


if __name__ == "__main__":
    asyncio.run(main())