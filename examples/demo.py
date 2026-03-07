#!/usr/bin/env python
"""
Interactive demo following the challenge protocol with interactive query interface.
FIXED: Removed api_key parameter from PageIndexBuilder initialization.
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryAgent
from src.db.integration import DatabaseIntegrator

# Try to import rich for beautiful output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.prompt import Prompt
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None


class DemoRunner:
    """Run the complete demo sequence with interactive query interface"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.triage = TriageAgent("config/extraction_rules.yaml")
        self.router = ExtractionRouter(openrouter_api_key=api_key)
        self.chunker = ChunkingEngine(max_tokens=512)
        
        # FIXED: Removed api_key parameter - PageIndexBuilder doesn't accept it
        self.indexer = PageIndexBuilder()  # No api_key parameter
        
        self.integrator = DatabaseIntegrator()
        
        self.results = {}
        self.query_agent = None
    
    def print_header(self, text):
        """Print formatted header"""
        if HAS_RICH:
            console.print(Panel.fit(f"[bold cyan]{text}[/bold cyan]", border_style="cyan"))
        else:
            print(f"\n{'='*60}")
            print(f"  {text}")
            print(f"{'='*60}")
    
    def print_step(self, step_num, title):
        """Print step header"""
        if HAS_RICH:
            console.print(f"\n[bold yellow]Step {step_num}: {title}[/bold yellow]")
        else:
            print(f"\n--- Step {step_num}: {title} ---")
    
    def print_success(self, message):
        """Print success message"""
        if HAS_RICH:
            console.print(f"  [green]✓[/green] {message}")
        else:
            print(f"  ✅ {message}")
    
    def print_info(self, message):
        """Print info message"""
        if HAS_RICH:
            console.print(f"  [blue]ℹ[/blue] {message}")
        else:
            print(f"  📌 {message}")
    
    def print_warning(self, message):
        """Print warning message"""
        if HAS_RICH:
            console.print(f"  [yellow]⚠[/yellow] {message}")
        else:
            print(f"  ⚠️ {message}")
    
    def print_error(self, message):
        """Print error message"""
        if HAS_RICH:
            console.print(f"  [red]✗[/red] {message}")
        else:
            print(f"  ❌ {message}")
    
    async def step1_triage(self, pdf_path: str):
        """Step 1: Triage Agent"""
        self.print_step(1, "Triage Agent")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console
        ) if HAS_RICH else nullcontext():
            profile = await self.triage.profile_document(pdf_path)
            await self.triage.save_profile(profile)
        
        if HAS_RICH:
            table = Table(show_header=False, box=None)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Document ID", profile.doc_id)
            table.add_row("Filename", profile.filename)
            table.add_row("Pages", str(profile.page_count))
            table.add_row("Origin Type", f"[green]{profile.origin_type.value}[/green]")
            table.add_row("Layout Complexity", f"[yellow]{profile.layout_complexity.value}[/yellow]")
            table.add_row("Domain Hint", f"[blue]{profile.domain_hint.value}[/blue]")
            table.add_row("Has Embedded Fonts", "✅ Yes" if profile.has_embedded_fonts else "❌ No")
            table.add_row("Image Area Ratio", f"{profile.image_area_ratio:.2%}")
            table.add_row("Est. Tables", str(profile.table_count_estimate))
            table.add_row("Recommended Strategy", f"[bold green]{profile.recommended_strategy.value}[/bold green]")
            
            console.print(table)
        else:
            print(f"  Document ID: {profile.doc_id}")
            print(f"  Filename: {profile.filename}")
            print(f"  Pages: {profile.page_count}")
            print(f"  Origin Type: {profile.origin_type.value}")
            print(f"  Layout Complexity: {profile.layout_complexity.value}")
            print(f"  Domain Hint: {profile.domain_hint.value}")
            print(f"  Recommended Strategy: {profile.recommended_strategy.value}")
        
        self.results['profile'] = profile
        return profile
    
    async def step2_extraction(self, pdf_path: str, profile):
        """Step 2: Extraction"""
        self.print_step(2, "Extraction Router")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console
        ) if HAS_RICH else nullcontext():
            doc = await self.router.extract(pdf_path, profile)
        
        self.print_success(f"Extracted with [bold]{doc.extraction_strategy}[/bold] strategy")
        self.print_info(f"Confidence: {doc.confidence_score:.2%}")
        self.print_info(f"Cost: ${doc.cost_estimate_usd:.4f}")
        self.print_info(f"Processing time: {doc.extraction_time_seconds:.1f}s")
        self.print_info(f"Found {len(doc.tables)} tables, {len(doc.figures)} figures")
        
        # Show sample table if available
        if doc.tables and HAS_RICH:
            console.print("\n[bold]Sample Table:[/bold]")
            table_data = doc.tables[0]
            table_json = json.dumps(table_data.model_dump(), indent=2)
            syntax = Syntax(table_json, "json", theme="monokai")
            console.print(syntax)
        
        self.results['document'] = doc
        return doc
    
    async def step3_chunking(self, doc):
        """Step 3: Semantic Chunking"""
        self.print_step(3, "Semantic Chunking Engine")
        
        chunks = self.chunker.chunk(doc)
        
        self.print_success(f"Created {chunks.chunk_count} Logical Document Units")
        self.print_info(f"Total tokens: {chunks.total_tokens:,}")
        
        # Show chunk distribution
        chunk_types = {}
        for chunk in chunks.chunks:
            chunk_types[chunk.chunk_type.value] = chunk_types.get(chunk.chunk_type.value, 0) + 1
        
        if HAS_RICH:
            table = Table(title="Chunk Types")
            table.add_column("Type", style="cyan")
            table.add_column("Count", style="white")
            table.add_column("Percentage", style="green")
            
            for ctype, count in chunk_types.items():
                percentage = count / chunks.chunk_count * 100
                table.add_row(ctype, str(count), f"{percentage:.1f}%")
            
            console.print(table)
        else:
            print(f"\n  Chunk Types:")
            for ctype, count in chunk_types.items():
                print(f"    - {ctype}: {count}")
        
        self.results['chunks'] = chunks
        return chunks
    
    async def step4_pageindex(self, chunks):
        """Step 4: PageIndex Builder"""
        self.print_step(4, "PageIndex Builder")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console
        ) if HAS_RICH else nullcontext():
            page_index = self.indexer.build_index(chunks)  # Note: not await - build_index is sync
            await self.indexer.save_index(page_index)
        
        self.print_success(f"Built navigation tree with {len(page_index.sections)} sections")
        self.print_info(f"Root sections: {len(page_index.root_sections)}")
        
        # Display section tree
        def print_section(section_id, indent=0):
            section = page_index.sections[section_id]
            prefix = "  " * indent
            line = f"{prefix}📄 {section.title} (p.{section.page_start})"
            
            if HAS_RICH:
                console.print(f"    {line}")
            else:
                print(f"    {line}")
            
            if section.summary and indent == 0:
                summary_line = f"{prefix}   📝 {section.summary[:100]}..."
                if HAS_RICH:
                    console.print(f"    [dim]{summary_line}[/dim]")
                else:
                    print(f"    {summary_line}")
            
            for child_id in section.child_sections[:3]:
                print_section(child_id, indent + 1)
        
        print(f"\n  Section Hierarchy:")
        for root_id in page_index.root_sections[:3]:
            print_section(root_id)
        
        self.results['page_index'] = page_index
        return page_index
    
    async def step5_setup_query_agent(self, doc_id, page_index):
        """Step 5: Setup Query Agent"""
        self.print_step(5, "Setting up Query Agent")
        
        # Process document in database integrator
        self.integrator.process_document(self.results['document'], self.results['chunks'])
        
        # Setup query agent
        self.query_agent = QueryAgent(
            vector_store=self.integrator.vector_store,
            fact_table_path=".refinery/facts/facts.db",
            page_indices={doc_id: page_index}
        )
        
        self.print_success("Query Agent ready with:")
        
        # Get stats
        vector_stats = self.integrator.vector_store.get_collection_stats()
        fact_stats = self.integrator.fact_table.get_stats()
        
        self.print_info(f"  • Vector store: {vector_stats.get('total_chunks', 0)} chunks")
        self.print_info(f"  • Fact table: {fact_stats.get('financial_facts', 0)} facts")
        self.print_info(f"  • PageIndex: {len(page_index.sections)} sections")
    
    async def interactive_query_mode(self, doc_id):
        """Interactive query interface"""
        if not self.query_agent:
            self.print_error("Query agent not initialized")
            return
        
        self.print_header("🔍 INTERACTIVE QUERY MODE")
        
        if HAS_RICH:
            console.print("\n[bold cyan]Ask questions about the document[/bold cyan]")
            console.print("[dim]Type 'exit' to quit, 'help' for commands[/dim]\n")
        else:
            print("\nAsk questions about the document")
            print("Type 'exit' to quit, 'help' for commands\n")
        
        while True:
            # Get user input
            if HAS_RICH:
                question = Prompt.ask("\n[bold yellow]Your question[/bold yellow]")
            else:
                question = input("\nYour question > ").strip()
            
            # Handle commands
            if question.lower() in ['exit', 'quit', 'q']:
                self.print_info("Exiting query mode")
                break
            
            if question.lower() in ['help', '?']:
                self._show_help()
                continue
            
            if question.lower() in ['stats', 'status']:
                self._show_stats()
                continue
            
            if question.lower() in ['sections', 'index']:
                self._show_sections()
                continue
            
            if not question:
                continue
            
            # Process query
            await self._process_query(question, doc_id)
    
    def _show_help(self):
        """Show help information"""
        if HAS_RICH:
            help_table = Table(title="Available Commands", show_header=True)
            help_table.add_column("Command", style="cyan")
            help_table.add_column("Description", style="white")
            
            help_table.add_row("exit, quit, q", "Exit query mode")
            help_table.add_row("help, ?", "Show this help")
            help_table.add_row("stats, status", "Show database statistics")
            help_table.add_row("sections, index", "Show document sections")
            help_table.add_row("[any question]", "Ask about the document")
            
            console.print(help_table)
        else:
            print("\nCommands:")
            print("  exit, quit, q - Exit query mode")
            print("  help, ? - Show this help")
            print("  stats, status - Show database statistics")
            print("  sections, index - Show document sections")
            print("  [any question] - Ask about the document")
    
    def _show_stats(self):
        """Show database statistics"""
        stats = self.integrator.get_database_stats()
        
        if HAS_RICH:
            stats_table = Table(title="Database Statistics")
            stats_table.add_column("Component", style="cyan")
            stats_table.add_column("Value", style="white")
            
            stats_table.add_row("Vector Store Chunks", str(stats['vector_store'].get('total_chunks', 0)))
            stats_table.add_row("Documents", str(stats['vector_store'].get('documents', 0)))
            stats_table.add_row("Financial Facts", str(stats['fact_table'].get('financial_facts', 0)))
            stats_table.add_row("Named Entities", str(stats['fact_table'].get('named_entities', 0)))
            stats_table.add_row("Key-Value Pairs", str(stats['fact_table'].get('key_value_pairs', 0)))
            
            console.print(stats_table)
        else:
            print("\nDatabase Statistics:")
            print(f"  Vector Store Chunks: {stats['vector_store'].get('total_chunks', 0)}")
            print(f"  Documents: {stats['vector_store'].get('documents', 0)}")
            print(f"  Financial Facts: {stats['fact_table'].get('financial_facts', 0)}")
            print(f"  Named Entities: {stats['fact_table'].get('named_entities', 0)}")
            print(f"  Key-Value Pairs: {stats['fact_table'].get('key_value_pairs', 0)}")
    
    def _show_sections(self):
        """Show document sections"""
        page_index = self.results.get('page_index')
        if not page_index:
            self.print_error("No page index available")
            return
        
        if HAS_RICH:
            sections_table = Table(title="Document Sections")
            sections_table.add_column("Section", style="cyan")
            sections_table.add_column("Page", style="white")
            sections_table.add_column("Type", style="green")
            sections_table.add_column("Entities", style="yellow")
            
            for section_id in page_index.root_sections[:10]:  # Show first 10
                section = page_index.sections[section_id]
                data_types = ", ".join([dt.value for dt in section.data_types_present[:3]])
                entities = ", ".join(section.key_entities[:3]) if section.key_entities else ""
                
                sections_table.add_row(
                    section.title[:40],
                    str(section.page_start),
                    data_types,
                    entities[:30]
                )
            
            console.print(sections_table)
        else:
            print("\nDocument Sections:")
            for section_id in page_index.root_sections[:10]:
                section = page_index.sections[section_id]
                print(f"  • {section.title} (p.{section.page_start})")
    
    async def _process_query(self, question: str, doc_id: str):
        """Process a single query and display results"""
        if HAS_RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
                console=console
            ) as progress:
                progress.add_task(description="Searching...", total=None)
                result = await self.query_agent.query(question, doc_id)
        else:
            print("  Searching...")
            result = await self.query_agent.query(question, doc_id)
        
        # Display answer
        if HAS_RICH:
            console.print("\n[bold green]Answer:[/bold green]")
            console.print(Panel(result.answer_text, border_style="green"))
            
            # Show confidence
            confidence_color = "green" if result.confidence_score > 0.7 else "yellow" if result.confidence_score > 0.4 else "red"
            console.print(f"Confidence: [{confidence_color}]{result.confidence_score:.2%}[/{confidence_color}]")
            console.print(f"Verification: [bold]{result.verification_status}[/bold]")
            
            # Show citations
            if result.citations:
                citations_table = Table(title="Sources", show_header=True)
                citations_table.add_column("#", style="cyan")
                citations_table.add_column("Document", style="white")
                citations_table.add_column("Page", style="green")
                citations_table.add_column("Excerpt", style="dim")
                
                for i, cit in enumerate(result.citations[:3], 1):
                    excerpt = cit.extracted_text[:80] + "..." if cit.extracted_text and len(cit.extracted_text) > 80 else cit.extracted_text or ""
                    citations_table.add_row(
                        str(i),
                        cit.document_name,
                        str(cit.page_number),
                        excerpt
                    )
                
                console.print(citations_table)
                
                if len(result.citations) > 3:
                    console.print(f"[dim]... and {len(result.citations) - 3} more sources[/dim]")
        else:
            print(f"\nAnswer: {result.answer_text}")
            print(f"Confidence: {result.confidence_score:.2%}")
            print(f"Verification: {result.verification_status}")
            
            if result.citations:
                print("\nSources:")
                for i, cit in enumerate(result.citations[:3], 1):
                    print(f"  {i}. {cit.document_name}, page {cit.page_number}")
                    if cit.extracted_text:
                        print(f"     \"{cit.extracted_text[:100]}...\"")
    
    async def run(self, pdf_path: str):
        """Run the complete demo"""
        self.print_header("Document Intelligence Refinery Demo")
        
        if HAS_RICH:
            console.print(f"\n[bold]Document:[/bold] {Path(pdf_path).name}")
        else:
            print(f"\nDocument: {Path(pdf_path).name}")
        
        try:
            # Step 1
            profile = await self.step1_triage(pdf_path)
            
            # Step 2
            doc = await self.step2_extraction(pdf_path, profile)
            
            # Step 3
            chunks = await self.step3_chunking(doc)
            
            # Step 4
            page_index = await self.step4_pageindex(chunks)
            
            # Step 5 - Setup query agent
            await self.step5_setup_query_agent(profile.doc_id, page_index)
            
            # Interactive query mode
            await self.interactive_query_mode(profile.doc_id)
            
            self.print_header("Demo Complete!")
            
        except KeyboardInterrupt:
            self.print_warning("\nDemo interrupted by user")
            return 1
        except Exception as e:
            self.print_error(f"Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        return 0


class nullcontext:
    """Null context manager for when rich is not available"""
    def __enter__(self): return self
    def __exit__(self, *args): pass


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the Document Intelligence Refinery demo with interactive query interface")
    parser.add_argument("--pdf", required=True, help="PDF file to process")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY env)")
    parser.add_argument("--no-rich", action="store_true", help="Disable rich formatting")
    
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
    
    # Check if file exists
    pdf_path = args.pdf
    if not Path(pdf_path).exists():
        print(f"❌ Error: File not found: {pdf_path}")
        sys.exit(1)
    
    # Run demo
    demo = DemoRunner(api_key)
    
    try:
        exit_code = asyncio.run(demo.run(pdf_path))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()