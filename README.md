# Document_intelligence_refinery
The repository contains a production-grade, multi-stage agentic pipeline for extracting structured, queryable, spatially-indexed knowledge from heterogeneous documents (PDFs, scanned images, reports...). The system is designed following the Master Thinker philosophy graceful degradation, confidence-gated escalation, and provenance preservation.

### Key Features

- **Multi-Strategy Extraction**: Fast text (pdfplumber), layout-aware (MinerU/Docking), and vision-based (OpenRouter VLM)
- **Confidence-Gated Escalation**: Automatically falls back to more sophisticated strategies when confidence is low
- **Spatial Provenance**: Every extracted fact includes bounding box coordinates and page references
- **Semantic Chunking**: Respects logical document units (tables, figures, lists) instead of token counts
- **PageIndex Navigation**: Hierarchical document structure for efficient retrieval without vector search
- **Audit Mode**: Verify claims against source documents with full provenance
- **Cost Budget Guard**: Prevents unexpected API costs with configurable limits

## 🏗 Architecture

 flowchart TD
    subgraph Input[Input Documents]
        A1[PDFs - Native Digital]
        A2[PDFs - Scanned]
        A3[Excel/CSV Reports]
        A4[Word Docs/Slides]
    end

    subgraph Stage1[Triage Agent]
        B[Document Profiler]
        C[Origin Detection<br/>Layout Analysis<br/>Domain Classification]
    end

    subgraph Stage2[Extraction Layer]
        direction LR
        D[FastTextExtractor<br/>pdfplumber]
        E[LayoutExtractor<br/>MinerU/Docking]
        F[VisionExtractor<br/>OpenRouter VLM]
        G{Router +<br/>Escalation Guard}
    end

    subgraph Stage3[Semantic Chunking]
        H[Chunking Engine]
        I[Logical Document Units]
        J[ChunkValidator<br/>5 Rules Enforcement]
    end

    subgraph Stage4[Indexing]
        K[PageIndex Builder]
        L[(Vector Store)]
        M[(SQLite Fact Table)]
    end

    subgraph Stage5[Query Interface]
        N[LangGraph Agent]
        O[Tools:<br/>Navigate, Search, SQL]
        P[ProvenanceChain<br/>+ Audit]
    end

    A1 & A2 & A3 & A4 --> B --> C
    C --> G
    G --> D & E & F
    D & E & F --> H --> I --> J
    J --> K & L & M
    K & L & M --> N --> O --> P

# Clone the repository

git clone <https://github.com/yourusername/doc-intel-refinery.git>
cd doc-intel-refinery

# Set up environment

python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies

uv pip install -r requirements.txt

# Install optional advanced extractors

uv pip install git+<https://github.com/opendatalab/MinerU.git>
uv pip install git+<https://github.com/DS4SD/docking.git>

# Configure API key

export OPENROUTER_API_KEY="your-api-key-here"

# Run on a sample document

python -m src.main --input ./data/sample.pdf --output .refinery

# View results

ls .refinery/profiles/
cat .refinery/extraction_ledger.jsonl
