"""
Vision-augmented extraction using OpenRouter with Gemma 3 27B.
FIXED: Handle None values in table cells by converting to empty strings.
"""

import os
import base64
import asyncio
import time
import tempfile
import json
import re
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import hashlib
from pathlib import Path

import aiohttp
import pdf2image
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ExtractionStrategy
from ..models.extracted_document import (
    ExtractedDocument, ContentBlock, BlockType,
    BoundingBox, Table, Figure
)
from ..models.document_profile import DocumentProfile
from ..utils.budget_guard import BudgetGuard

poppler_path = r"C:\Users\hp\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"
class VisionExtractor(ExtractionStrategy):
    """
    Vision-augmented extraction using multimodal models via OpenRouter.
    FIXED: Handle None values in table cells and ensure all data is properly typed.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "google/gemma-3-27b-it",
        max_tokens_per_request: int = 4000,
        max_pages_per_batch: int = 5,
        temp_dir: Optional[str] = None
    ):
        """
        Initialize vision extractor.
        """
        super().__init__("vision_augmented")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens_per_request
        self.max_pages_per_batch = max_pages_per_batch
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.page_cost = 0.05
        self.budget_guard = BudgetGuard()
        
        # Setup temp directory
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            system_temp = Path(tempfile.gettempdir())
            self.temp_dir = system_temp / "refinery_vision"
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = asyncio.Semaphore(2)
        self.temp_files = []
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        for file_path in self.temp_files:
            try:
                if Path(file_path).exists():
                    Path(file_path).unlink()
            except:
                pass
        self.temp_files.clear()
    
    def _safe_parse_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Safely parse JSON from LLM response with multiple fallback strategies"""
        # Strategy 1: Try direct JSON parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON from markdown code blocks
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}'
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    cleaned = match.strip()
                    cleaned = re.sub(r'}(\s*){', r'},\1{', cleaned)
                    cleaned = re.sub(r',\s*}', '}', cleaned)
                    cleaned = re.sub(r',\s*\]', ']', cleaned)
                    return json.loads(cleaned)
                except:
                    continue
        
        return None
    
    def _clean_table_data(self, table_data: Dict) -> Dict:
        """
        Clean table data to ensure all values are strings and no None values.
        
        Args:
            table_data: Raw table data from vision model
            
        Returns:
            Cleaned table data with all values converted to strings
        """
        cleaned = {
            'headers': [],
            'rows': [],
            'caption': table_data.get('caption', '')
        }
        
        # Clean headers - ensure all are strings
        headers = table_data.get('headers', [])
        if isinstance(headers, list):
            cleaned['headers'] = [str(h) if h is not None else '' for h in headers]
        else:
            cleaned['headers'] = []
        
        # Clean rows - ensure all cells are strings
        rows = table_data.get('rows', [])
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, list):
                    cleaned_row = [str(cell) if cell is not None else '' for cell in row]
                    cleaned['rows'].append(cleaned_row)
                else:
                    cleaned['rows'].append([])
        else:
            cleaned['rows'] = []
        
        return cleaned
    
    def _clean_page_result(self, page_result: Dict) -> Dict:
        """
        Clean page result to ensure all data meets Pydantic model requirements.
        
        Args:
            page_result: Raw page result from vision model
            
        Returns:
            Cleaned page result with proper types
        """
        cleaned = {
            'page_number': page_result.get('page_number', 1),
            'text_blocks': [],
            'tables': [],
            'figures': [],
            'has_handwriting': bool(page_result.get('has_handwriting', False)),
            'extraction_confidence': float(page_result.get('extraction_confidence', 0.5))
        }
        
        # Clean text blocks
        text_blocks = page_result.get('text_blocks', [])
        if isinstance(text_blocks, list):
            for block in text_blocks:
                if isinstance(block, dict):
                    cleaned_block = {
                        'content': str(block.get('content', '')) if block.get('content') is not None else '',
                        'block_type': str(block.get('block_type', 'paragraph')),
                        'position': str(block.get('position', '')) if block.get('position') is not None else ''
                    }
                    cleaned['text_blocks'].append(cleaned_block)
        
        # Clean tables
        tables = page_result.get('tables', [])
        if isinstance(tables, list):
            for table in tables:
                if isinstance(table, dict):
                    cleaned['tables'].append(self._clean_table_data(table))
        
        # Clean figures
        figures = page_result.get('figures', [])
        if isinstance(figures, list):
            for figure in figures:
                if isinstance(figure, dict):
                    cleaned_figure = {
                        'caption': str(figure.get('caption', '')) if figure.get('caption') is not None else '',
                        'description': str(figure.get('description', '')) if figure.get('description') is not None else ''
                    }
                    cleaned['figures'].append(cleaned_figure)
        
        return cleaned
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _call_openrouter(self, messages: List[Dict], max_tokens: int = 2000) -> Dict:
        """Call OpenRouter API with retry logic"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/document-intelligence-refinery",
            "X-Title": "Document Intelligence Refinery"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": min(max_tokens, self.max_tokens),
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        
        async with self.rate_limit:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise Exception(f"OpenRouter API error: {response.status} - {text}")
                    return await response.json()
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _create_extraction_prompt(self, page_num: int, total_pages: int) -> str:
        """Create prompt for page extraction"""
        return f"""You are a document extraction specialist. Extract all content from page {page_num} of {total_pages} of this document.

Return a valid JSON object with the following structure:
{{
    "page_number": {page_num},
    "text_blocks": [
        {{
            "content": "text content",
            "block_type": "paragraph",
            "position": "approximate location description"
        }}
    ],
    "tables": [
        {{
            "headers": ["col1", "col2"],
            "rows": [["row1col1", "row1col2"]],
            "caption": "optional table caption"
        }}
    ],
    "figures": [
        {{
            "caption": "figure caption if present",
            "description": "description of figure content"
        }}
    ],
    "has_handwriting": false,
    "extraction_confidence": 0.95
}}

Important rules:
- Return ONLY valid JSON, no other text
- All string values must be strings, never null
- If a cell is empty, use empty string "" not null
- If no content exists, return empty arrays
- Keep all numbers and financial figures exactly as shown"""
    
    async def extract_page(self, image_path: str, page_num: int, total_pages: int) -> Dict[str, Any]:
        """Extract content from a single page image"""
        try:
            base64_image = self._encode_image(image_path)
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a precise document extraction system. Output only valid JSON with no null values."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._create_extraction_prompt(page_num, total_pages)
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            estimated_tokens = 1000 + len(base64_image) // 4
            budget_check = self.budget_guard.check_budget(self.name, 1, estimated_tokens)
            
            if not budget_check["approved"]:
                raise Exception(f"Budget limit exceeded: {budget_check['warnings']}")
            
            response = await self._call_openrouter(messages, max_tokens=4000)
            content = response["choices"][0]["message"]["content"]
            raw_result = self._safe_parse_json(content)
            
            if not raw_result:
                print(f"⚠️ Failed to parse JSON for page {page_num}, using default")
                return self._create_default_page_result(page_num)
            
            # Clean the result to ensure proper types
            cleaned_result = self._clean_page_result(raw_result)
            
            # Track cost
            usage = response.get("usage", {})
            total_tokens = usage.get("total_tokens", estimated_tokens)
            cost = total_tokens * self.budget_guard.token_costs["input"]
            self.budget_guard.record_spend(cost)
            
            cleaned_result["_cost_usd"] = cost
            cleaned_result["_tokens_used"] = total_tokens
            
            return cleaned_result
            
        except Exception as e:
            print(f"⚠️ Error processing page {page_num}: {e}")
            return self._create_default_page_result(page_num)
    
    def _create_default_page_result(self, page_num: int) -> Dict[str, Any]:
        """Create a default page result when parsing fails"""
        return {
            "page_number": page_num,
            "text_blocks": [],
            "tables": [],
            "figures": [],
            "has_handwriting": False,
            "extraction_confidence": 0.3,
            "_cost_usd": 0.0,
            "_tokens_used": 0
        }
    
    def _convert_page_result(self, page_result: Dict, doc_id: str) -> Tuple[List[ContentBlock], List[Table], List[Figure]]:
        """Convert page extraction result to models with proper type handling"""
        blocks = []
        tables = []
        figures = []
        
        page_num = page_result.get("page_number", 1)
        
        # Convert text blocks
        for i, text_block in enumerate(page_result.get("text_blocks", [])):
            block_type_map = {
                "paragraph": BlockType.TEXT,
                "header": BlockType.HEADER,
                "list": BlockType.LIST,
                "caption": BlockType.CAPTION,
                "footer": BlockType.FOOTER
            }
            
            block = ContentBlock(
                block_id=f"{doc_id}_vision_text_{page_num}_{i}",
                block_type=block_type_map.get(
                    text_block.get("block_type", "paragraph"),
                    BlockType.TEXT
                ),
                content=text_block.get("content", ""),
                bbox=None,
                metadata={
                    "position_hint": text_block.get("position", ""),
                    "source": "vision",
                    "page": page_num
                }
            )
            blocks.append(block)
        
        # Convert tables with cleaned data
        for i, table_data in enumerate(page_result.get("tables", [])):
            # Table requires bbox
            bbox = BoundingBox(
                x0=50, y0=100 + i*100,
                x1=550, y1=200 + i*100,
                page_number=page_num
            )
            
            table = Table(
                headers=table_data.get("headers", []),
                rows=table_data.get("rows", []),
                caption=table_data.get("caption"),
                bbox=bbox
            )
            tables.append(table)
            
            # Add as block
            table_block = ContentBlock(
                block_id=f"{doc_id}_vision_table_{page_num}_{i}",
                block_type=BlockType.TABLE,
                content=table.to_markdown(),
                bbox=None,
                metadata={
                    "caption": table.caption,
                    "headers": table.headers,
                    "source": "vision"
                }
            )
            blocks.append(table_block)
        
        # Convert figures
        for i, figure_data in enumerate(page_result.get("figures", [])):
            bbox = BoundingBox(
                x0=50, y0=300 + i*100,
                x1=550, y1=400 + i*100,
                page_number=page_num
            )
            
            figure = Figure(
                caption=figure_data.get("caption", ""),
                bbox=bbox,
                description=figure_data.get("description", "")
            )
            figures.append(figure)
        
        return blocks, tables, figures
    
    async def extract(self, pdf_path: str, profile: Optional[DocumentProfile] = None) -> ExtractedDocument:
        """Extract content using vision model with proper type handling"""
        start_time = time.time()
        doc_id = hashlib.md5(pdf_path.encode()).hexdigest()[:12]
        
        try:
            print(f"🖼️ Converting PDF to images...")
            images = pdf2image.convert_from_path(pdf_path,poppler_path=poppler_path)
            total_pages = len(images)
            print(f"✅ Converted {total_pages} pages")
            
            blocks: List[ContentBlock] = []
            tables: List[Table] = []
            figures: List[Figure] = []
            total_cost = 0.0
            confidences = []
            
            self._cleanup_temp_files()
            
            for batch_start in range(0, total_pages, self.max_pages_per_batch):
                batch_end = min(batch_start + self.max_pages_per_batch, total_pages)
                batch_tasks = []
                batch_temp_files = []
                
                print(f"📄 Processing pages {batch_start + 1}-{batch_end}...")
                
                for page_num in range(batch_start + 1, batch_end + 1):
                    safe_filename = f"page_{doc_id}_{page_num}.jpg"
                    temp_path = self.temp_dir / safe_filename
                    
                    try:
                        images[page_num - 1].save(temp_path, "JPEG", quality=95)
                        batch_temp_files.append(str(temp_path))
                        self.temp_files.append(str(temp_path))
                        
                        task = self.extract_page(str(temp_path), page_num, total_pages)
                        batch_tasks.append(task)
                    except Exception as e:
                        print(f"⚠️ Error saving page {page_num}: {e}")
                        continue
                
                if not batch_tasks:
                    continue
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        print(f"⚠️ Error in batch: {result}")
                        continue
                    
                    if result:
                        page_blocks, page_tables, page_figures = self._convert_page_result(result, doc_id)
                        blocks.extend(page_blocks)
                        tables.extend(page_tables)
                        figures.extend(page_figures)
                        
                        total_cost += result.get("_cost_usd", 0)
                        confidences.append(result.get("extraction_confidence", 0.5))
                
                # Clean up batch temp files
                for temp_file in batch_temp_files:
                    try:
                        Path(temp_file).unlink()
                        if temp_file in self.temp_files:
                            self.temp_files.remove(temp_file)
                    except:
                        pass
            
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            extraction_time = time.time() - start_time
            
            print(f"✅ Vision extraction complete: {len(blocks)} blocks, {len(tables)} tables, {len(figures)} figures")
            
            return ExtractedDocument(
                doc_id=doc_id,
                filename=Path(pdf_path).name,
                page_count=total_pages,
                blocks=blocks,
                tables=tables,
                figures=figures,
                extraction_strategy=self.name,
                extraction_timestamp=datetime.now(),
                confidence_score=overall_confidence,
                extraction_time_seconds=extraction_time,
                cost_estimate_usd=total_cost
            )
            
        except Exception as e:
            print(f"❌ Vision extraction failed: {e}")
            raise
        finally:
            self._cleanup_temp_files()
    
    def estimate_cost(self, pdf_path: str) -> float:
        """Estimate cost based on page count"""
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
            return page_count * self.page_cost
        except:
            return 1.0 * self.page_cost