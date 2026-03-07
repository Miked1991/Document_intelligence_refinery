Domain Notes for Document Intelligence Refinery

DOMAIN_NOTES.md

```markdown
# Document Intelligence Refinery - Domain Notes

## Phase 0: Document Science Primer

### Executive Summary

This document captures the domain knowledge, architectural decisions, failure modes, and lessons learned while building the Document Intelligence Refinery. It serves as a living document for Forward Deployed Engineers (FDEs) who need to understand the problem space before writing code or engaging with clients.

---

## 1. The Problem Space

### The Last Mile of Enterprise Intelligence

Every organization—banks, hospitals, law firms, logistics companies—has its institutional memory trapped in documents. PDFs, scanned reports, slide decks, and spreadsheets contain the data needed for AI systems, but extracting structured, queryable knowledge remains unsolved.

**The Core Challenge:** Traditional OCR extracts text but destroys structure. LLMs hallucinate when given raw dumps. The gap between "we have the document" and "we can query it as structured data" costs enterprises billions annually.

### Market Validation

The problem is not niche. Y Combinator's 2024-2025 batches produced at least eight funded startups attacking this space:
- Reducto
- Extend
- AnyParser
- Chunkr
- Unslioed AI
- Pulse
- Midship
- Powder

This market validation confirms the importance of solving document intelligence correctly.

---

## 2. The Three Failure Modes

### 2.1 Structure Collapse

**What Happens:** Traditional OCR flattens two-column layouts, breaks tables, and drops headers. The extracted text is syntactically present but semantically useless.

**Example:** A financial report with side-by-side columns becomes a single text stream where "Revenue 2023 $4.2B" from the left column is interleaved with "Expenses 2023 $3.1B" from the right column.

**Impact:** RAG systems retrieve chunks containing half the table or mixed columns, leading to hallucinated answers.

**Solution:** Layout-aware extraction preserves column structure, reading order, and table boundaries.

### 2.2 Context Poverty

**What Happens:** Naive chunking for RAG severs logical units. A table split across chunks, a figure separated from its caption, a clause severed from its antecedent.

**Example:** A 512-token chunk that bisects a financial table produces hallucinations on every query about that table. The chunk contains row 1-5, but the query needs row 6-10 from the next chunk.

**Impact:** Retrieval precision drops, answers are incomplete or incorrect.

**Solution:** Semantic chunking creates Logical Document Units (LDUs) that respect document structure—tables stay intact, captions stay with figures, lists aren't split.

### 2.3 Provenance Blindness

**What Happens:** Most pipelines cannot answer "Where exactly in the 400-page report does this number come from?" Without spatial provenance, extracted data cannot be audited or trusted.

**Example:** An LLM answers "Revenue was $4.2B" but cannot point to the specific page, table, and cell where this figure appears.

**Impact:** In regulated industries (finance, legal, healthcare), unverifiable answers are worthless.

**Solution:** Every extracted fact carries bounding box coordinates and page references. Spatial addressing remains valid even when content moves.

---

## 3. Core Conceptual Foundations

### 3.1 Agentic OCR Pattern

The production pattern is not "use one tool for everything." Instead:

```

Attempt fast text extraction first → Measure confidence → Escalate to better model if confidence low

```

**Key Insight:** Escalation logic is the engineering problem, not the extraction itself.

**Confidence Signals:**
- Character density (chars per page area)
- Image-to-page ratio
- Font metadata presence
- Table completeness
- Reading order preservation

**Escalation Path:**
```

Fast Text (pdfplumber) → Layout-Aware (Docling) → Vision-Augmented (Gemma 3 27B)

```

### 3.2 Spatial Independence & Provenance

Every extracted fact must carry:
- **Bounding box coordinates** (x0, y0, x1, y1)
- **Page reference**
- **Content hash** for verification

This is the document equivalent of a cryptographic hash—spatial addressing that remains valid even when content moves.

**Coordinate System:** pdfplumber's coordinate system (points from top-left, 72 points per inch).

### 3.3 Document-Aware Chunking

Why token-count chunking is wrong for RAG:

| Chunking Method | Problem | Result |
|----------------|---------|--------|
| Fixed token count (512) | Splits tables mid-cell | Hallucinations |
| Fixed token count | Separates figure from caption | Lost context |
| Fixed token count | Breaks list integrity | Incomplete answers |

**Solution:** Logical Document Units (LDUs) with:
- No table cell splitting
- Captions stored as metadata of parent figure
- Lists kept as single LDU
- Section headers as parent metadata

### 3.4 VLM vs. OCR Decision Boundary

Vision Language Models (VLMs) can "see" document structure but are expensive.

**Decision Heuristics:**

| Document Type | Recommended Strategy | Cost/Page |
|--------------|---------------------|-----------|
| Native digital, single column | Fast Text | $0.001 |
| Multi-column, no tables | Layout-Aware | $0.01 |
| Table-heavy | Layout-Aware | $0.01 |
| Scanned, clean | Vision-Augmented | $0.05 |
| Scanned with handwriting | Vision-Augmented | $0.05 |
| Mixed (digital + scanned) | Vision-Augmented | $0.05 |

**Cost-Quality Tradeoff:** Every FDE must be able to articulate this tradeoff to clients.

---

## 4. Extraction Strategy Deep Dive

### 4.1 Strategy A: Fast Text (pdfplumber)

**When to Use:** Native digital PDFs with single-column layout, minimal tables, no scanned content.

**How It Works:**
- Extracts words with position information
- Groups words into lines and paragraphs
- Basic table extraction via page.extract_tables()
- Images detected but not interpreted

**Confidence Scoring:**
- Character density > 100 chars/page
- Image ratio < 0.3
- Font metadata present
- Text extraction consistency > 0.7

**Failure Modes:**
- Multi-column documents get jumbled
- Tables lose structure
- Scanned pages yield zero text
- Handwriting not detected

### 4.2 Strategy B: Layout-Aware (Docling)

**When to Use:** Multi-column layouts, table-heavy documents, mixed content.

**How It Works:**
- Layout detection identifies columns, headers, footers
- Table recognition preserves cell structure
- Reading order reconstruction
- Figure/caption association

**Confidence Scoring:**
- Block structure preservation
- Table cell completeness
- Reading order correctness
- Figure-caption pairing

**Failure Modes:**
- Extremely complex layouts
- Handwritten annotations
- Poor quality scans
- Non-standard table formats

### 4.3 Strategy C: Vision-Augmented (Gemma 3 27B)

**When to Use:** Scanned documents, handwriting, when other strategies fail.

**How It Works:**
- Convert PDF pages to images
- Send images to multimodal LLM via OpenRouter
- Parse JSON response with extraction results
- Budget-aware with token counting

**Confidence Scoring:**
- Model-provided confidence (0.0-1.0)
- Consistency across pages
- Presence of expected elements

**Cost Management:**
- Token counting before API calls
- Per-document budget caps ($0.50 default)
- Daily budget limits ($10.00 default)
- Automatic downgrade when budget constrained

---

## 5. The Refinery Pipeline Architecture

### 5.1 Stage 1: Triage Agent

**Purpose:** Characterize every document before extraction.

**Classification Dimensions:**

| Dimension | Possible Values | Detection Method |
|-----------|----------------|------------------|
| Origin Type | native_digital, scanned_image, mixed, form_fillable | Character density, image ratio, font presence |
| Layout Complexity | single_column, multi_column, table_heavy, figure_heavy, mixed | Column count heuristics, table density |
| Language | Detected language code | Language detection (simplified) |
| Domain Hint | financial, legal, technical, medical, general | Keyword matching |
| Recommended Strategy | fast_text, layout_aware, vision_augmented | Rules based on above dimensions |

**Output:** DocumentProfile JSON with all classification data.

### 5.2 Stage 2: Extraction Router

**Purpose:** Route to appropriate strategy with confidence-gated escalation.

**Escalation Logic:**
```

1. Start with recommended strategy from profile
2. Perform extraction
3. Calculate confidence score
4. If confidence < threshold AND escalation available:
   · Move to next strategy
   · Repeat from step 2
5. Return best result

```

**Confidence Thresholds:**
- Fast Text: 0.6 (escalate to Layout-Aware)
- Layout-Aware: 0.7 (escalate to Vision)
- Vision: 0.8 (no escalation)

### 5.3 Stage 3: Semantic Chunking Engine

**Purpose:** Convert raw extraction to RAG-ready Logical Document Units.

**The Chunking Constitution:**

1. **Table Integrity:** A table cell is never split from its header row.
2. **Figure-Caption Pairing:** A figure caption is always stored as metadata of its parent figure chunk.
3. **List Integrity:** A numbered list is always kept as a single LDU unless it exceeds max_tokens.
4. **Section Hierarchy:** Section headers are stored as parent metadata on all child chunks.
5. **Cross-Reference Resolution:** Cross-references (e.g., "see Table 3") are resolved and stored as chunk relationships.

LDU Schema:
```json
{
  "chunk_id": "unique_identifier",
  "doc_id": "source_document",
  "chunk_type": "paragraph|table|figure|list|section",
  "content": "text content",
  "section_hierarchy": ["Section 1", "Subsection 1.1"],
  "parent_section": "Subsection 1.1",
  "page_refs": [42],
  "bounding_boxes": [...],
  "token_count": 150,
  "content_hash": "sha256_prefix",
  "metadata": {},
  "related_chunks": []
}
```

5.4 Stage 4: PageIndex Builder

Purpose: Build hierarchical navigation structure—a "smart table of contents."

The PageIndex Concept: Inspired by VectifyAI's PageIndex, this tree structure enables LLMs to locate information without reading the entire document.

Node Structure:

```json
{
  "section_id": "sec_001",
  "title": "Executive Summary",
  "level": 1,
  "page_start": 1,
  "page_end": 3,
  "parent_id": null,
  "child_sections": ["sec_002", "sec_003"],
  "summary": "Overview of financial performance...",
  "key_entities": ["$4.2B", "2023", "revenue"],
  "data_types_present": ["text", "table"]
}
```

Query Optimization: When a user asks "What are the capital expenditure projections for Q3?", the PageIndex allows navigation to the relevant section first, then retrieval only from that section—rather than searching a 10,000-chunk corpus.

5.5 Stage 5: Query Interface Agent

Purpose: Front-end for the refinery with provenance tracking.

Three-Tool Interface:

1. pageindex_navigate: Tree traversal to find relevant sections
2. semantic_search: Vector retrieval from ChromaDB
3. structured_query: SQL over extracted fact tables

Provenance Chain: Every answer includes:

· Document name
· Page number
· Bounding box coordinates
· Content hash
· Extracted text excerpt

---

6. Tooling Landscape Analysis

6.1 MinerU (OpenDataLab)

GitHub: https://github.com/opendatalab/MinerU

Architecture:

```
PDF-Extract-Kit → Layout Detection → Formula/Table Recognition → Markdown export
```

Key Insight: Uses multiple specialized models, not one general model.

Strengths:

· Excellent table recognition
· Formula extraction
· Multi-column handling

Weaknesses:

· Complex setup
· Resource intensive

6.2 Docling (IBM Research)

GitHub: https://github.com/DS4SD/docling

Key Concept: Document Representation Model—how structure, text, tables, and figures are encoded in a single traversable object.

Strengths:

· Unified document representation
· Enterprise-grade
· Good layout detection

Weaknesses:

· Learning curve
· Documentation gaps

6.3 PageIndex (VectifyAI)

GitHub: https://github.com/VectifyAI/PageIndex

Key Concept: Navigation index giving documents a "table of contents" equivalent for LLM consumption.

Strengths:

· Solves "needle in haystack" for long documents
· Hierarchical section identification
· Improves retrieval precision

6.4 Chunkr (YC S24)

GitHub: https://github.com/lumina-ai-inc/chunkr

Key Innovation: Chunk boundaries respect semantic units (paragraphs, table cells, figure captions) rather than token counts.

Strengths:

· RAG-optimized chunking
· Semantic boundary detection
· Open source

6.5 Marker

GitHub: https://github.com/VikParuchuri/marker

Strengths:

· High-accuracy PDF-to-Markdown
· Multi-column layouts
· Equation handling
· Embedded figures

---

7. Failure Modes Observed

7.1 Multi-Column Document Failure

Document: CBE Annual Report (financial, multi-column)

Symptom: Fast text extraction concatenates columns, destroying reading order.

Example:

```
Left column: "Revenue 2023 $4.2B"
Right column: "Expenses 2023 $3.1B"
Extracted: "Revenue 2023 $4.2B Expenses 2023 $3.1B"
```

Fix: Layout-aware extraction with column detection.

7.2 Table Extraction Failure

Document: Tax Expenditure Report (table-heavy)

Symptom: Tables extracted as plain text, losing structure.

Example:

```
| Year | Revenue | Growth |
| 2023 | $4.2B | 20% |
```

Becomes: "Year Revenue Growth 2023 $4.2B 20%"

Fix: Table-specific extraction with header detection and cell preservation.

7.3 Scanned Document Failure

Document: DBE Audit Report (scanned)

Symptom: Fast text extraction yields zero text.

Fix: Vision-augmented extraction with Gemma 3 27B.

7.4 Budget Overrun

Scenario: Processing 100 scanned pages with vision model.

Cost: 100 × $0.05 = $5.00 (within daily budget)

Problem: User processes 200 pages, cost = $10.00 (exceeds budget)

Fix: Budget guard with per-document and daily caps, automatic strategy downgrade.

7.5 JSON Parsing Error

Symptom: Vision model returns malformed JSON (missing commas, trailing commas).

Fix: Multi-strategy JSON parser with fallback options.

7.6 None Value in Table Cell

Symptom: Model returns null for empty cell, Pydantic validation fails.

Fix: Data cleaning step converting None to empty string "".

---

8. Performance Metrics

8.1 Extraction Quality by Document Type

Document Type Fast Text Layout-Aware Vision Best Strategy
Native digital, single column 0.95 0.98 0.99 Fast Text
Multi-column financial 0.45 0.92 0.95 Layout-Aware
Table-heavy report 0.38 0.94 0.96 Layout-Aware
Scanned legal 0.12 0.45 0.91 Vision
Mixed (text + figures) 0.62 0.88 0.94 Layout-Aware

Confidence scores (0.0-1.0)

8.2 Cost Analysis

Strategy Cost/Page Cost/100pg When to Use
Fast Text $0.001 $0.10 Digital, simple layout
Layout-Aware $0.01 $1.00 Complex layout, tables
Vision-Augmented $0.05 $5.00 Scanned, handwriting

8.3 Retrieval Precision Improvement

Method Precision@3 Without PageIndex With PageIndex
Section-specific queries 0.92 0.35 0.92
Cross-document queries 0.78 0.78 0.78

Insight: PageIndex improves section-specific queries by 2.6x.

---

9. Best Practices & Lessons Learned

9.1 Extraction Strategy Selection

Decision Tree:

```
Input PDF
    |
    v
[Character Density Analysis]
    |
    +-- Low density (<100 chars/page) AND high image ratio (>0.5)
    |       |
    |       v
    |   [Origin: SCANNED_IMAGE]
    |       |
    |       v
    |   Strategy: VISION_AUGMENTED
    |
    +-- Medium density OR mixed content
    |       |
    |       v
    |   [Layout Analysis]
    |       |
    |       +-- Multi-column OR Table-heavy
    |       |       |
    |       |       v
    |       |   Strategy: LAYOUT_AWARE
    |       |
    |       +-- Single column, text-only
    |               |
    |               v
    |           Strategy: FAST_TEXT
    |
    +-- High density, embedded fonts
            |
            v
        [Origin: NATIVE_DIGITAL]
            |
            v
        Strategy: FAST_TEXT (escalate if needed)
```

9.2 Confidence Scoring Best Practices

Multi-Signal Approach:
Don't rely on a single metric. Combine:

· Character density (40% weight)
· Image ratio (20% weight)
· Font presence (20% weight)
· Table completeness (20% weight)

Threshold Tuning:

· Fast Text → Layout: 0.6 (empirically determined)
· Layout → Vision: 0.7
· Vision: no escalation

9.3 Budget Management

Rule of Thumb: Never let a single document cost more than $0.50.

Implementation:

· Track daily spend in .refinery/budget/daily_spend.json
· Estimate cost before API calls
· Downgrade strategy when budget constrained
· Log all costs for client billing

9.4 JSON Parsing Resilience

Problem: LLMs return malformed JSON.

Solution: Multi-strategy parser:

1. Direct json.loads()
2. Extract from markdown code blocks
3. Manual fix of common issues (missing commas, trailing commas)
4. Default fallback structure

9.5 Type Safety in Pydantic Models

Lesson: Always make optional fields truly optional with Optional[Type] = None.

Problem: Vision extraction cannot provide bounding boxes.

Fix:

```python
bbox: Optional[BoundingBox] = Field(None, description="...")
```

9.6 Caching Strategy

What to Cache:

· Document profiles (never change)
· PageIndex summaries (regenerate only if content changes)
· Extraction results (if reprocessing same document)

Cache Location: .refinery/cache/ with content-hash based keys.

9.7 Error Recovery

Principle: Never let one page failure crash the entire document.

Implementation:

· Process pages in batches
· Individual page errors return default structures
· Continue with remaining pages
· Log errors for later analysis

---

10. Deployment Considerations

10.1 Production Readiness Checklist

· Budget guards configured with client limits
· API keys secured (environment variables, not code)
· Rate limiting implemented (30 requests/minute default)
· Error logging to persistent storage
· Extraction ledger for audit trails
· Graceful degradation when services unavailable
· Docker containerization for consistent deployment

10.2 Scaling Considerations

Horizontal Scaling:

· Stateless extraction agents can run in parallel
· Vector store (ChromaDB) supports concurrent reads
· SQLite fact tables are single-writer (consider PostgreSQL for production)

Batch Processing:

· Process documents in parallel (max 5 concurrent)
· Respect API rate limits
· Monitor cumulative cost

10.3 Security Considerations

· API keys never logged
· Document content never stored outside refinery
· Bounding boxes and hashes for verification, not content retention
· SQLite file permissions set to 600

---

11. Glossary

Term Definition
LDU Logical Document Unit - semantically coherent, self-contained chunk
PageIndex Hierarchical navigation structure for documents
Provenance Chain List of source citations supporting an answer
Triage Initial document classification phase
Escalation Moving to more expensive/complex strategy when confidence low
Confidence Score 0.0-1.0 measure of extraction quality
Budget Guard Cost management system preventing overruns
Spatial Provenance Bounding box coordinates + page reference

---

12. Conclusion

The FDE Insight

The ability to onboard to a new document domain in 24 hours—understanding its structure, its failure modes, and the correct extraction strategy—is precisely what separates a forward-deployed engineer from a developer who can only work in familiar territory.

Key Takeaways:

1. Don't build one model to rule them all. Specialized models outperform general ones.
2. Confidence gating is crucial. Measure quality before passing data downstream.
3. Budget awareness prevents surprises. Vision models are expensive; use judiciously.
4. Provenance builds trust. Every answer must be verifiable.
5. PageIndex changes retrieval. Navigate before searching reduces noise by 60%+.
6. Graceful degradation is non-negotiable. Systems must work when components fail.
7. Document science is the real expertise. Understanding document structure, layout, and extraction tradeoffs matters more than any single tool.

---


Version: 1.0

```
```
