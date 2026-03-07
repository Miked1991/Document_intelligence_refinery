"""
Vector store implementation for semantic search using ChromaDB.
Handles embedding generation, storage, and retrieval of LDUs.
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
from datetime import datetime

# Vector DB
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Local imports
from ..models.ldu import LDU, ChunkCollection
from ..models.provenancechain import SourceCitation


class VectorStore:
    """
    Vector store for semantic search with provenance tracking.
    Uses ChromaDB with sentence-transformers embeddings.
    """
    
    def __init__(
        self,
        persist_directory: str = ".refinery/vectors",
        collection_name: str = "document_chunks",
        embedding_model: str = "all-MiniLM-L6-v2",
        distance_metric: str = "cosine"
    ):
        """
        Initialize vector store.
        
        Args:
            persist_directory: Directory to persist vector data
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence transformer model for embeddings
            distance_metric: Distance metric for similarity search
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
            print(f"✅ Loaded existing collection: {collection_name}")
        except ValueError:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": distance_metric}
            )
            print(f"✅ Created new collection: {collection_name}")
    
    def add_chunks(self, chunks: Union[List[LDU], ChunkCollection]) -> int:
        """
        Add chunks to vector store.
        
        Args:
            chunks: List of LDUs or ChunkCollection
            
        Returns:
            Number of chunks added
        """
        if isinstance(chunks, ChunkCollection):
            chunks_list = chunks.chunks
        else:
            chunks_list = chunks
        
        if not chunks_list:
            return 0
        
        # Prepare batch data
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks_list:
            # Generate unique ID
            chunk_id = f"{chunk.doc_id}_{chunk.chunk_id}"
            ids.append(chunk_id)
            
            # Store content
            documents.append(chunk.content)
            
            # Store metadata for filtering and provenance
            metadata = {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "chunk_type": chunk.chunk_type.value,
                "page_refs": json.dumps(chunk.page_refs),
                "token_count": chunk.token_count,
                "content_hash": chunk.content_hash,
                "parent_section": chunk.parent_section or "",
                "section_hierarchy": json.dumps(chunk.section_hierarchy),
                "has_table": "true" if chunk.chunk_type.value == "table" else "false",
                "has_figure": "true" if chunk.chunk_type.value == "figure" else "false"
            }
            
            # Add bounding boxes if available
            if chunk.bounding_boxes:
                metadata["bbox_count"] = str(len(chunk.bounding_boxes))
                metadata["first_bbox"] = json.dumps(chunk.bounding_boxes[0])
            
            # Add any additional metadata
            for key, value in chunk.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[f"meta_{key}"] = str(value)
                elif value is not None:
                    metadata[f"meta_{key}"] = json.dumps(value)
            
            metadatas.append(metadata)
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
        
        print(f"✅ Added {len(ids)} chunks to vector store")
        return len(ids)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_dict: Metadata filters
            doc_id: Filter by document ID
            
        Returns:
            List of search results with metadata and scores
        """
        # Build where clause
        where_clause = {}
        if doc_id:
            where_clause["doc_id"] = doc_id
        if filter_dict:
            where_clause.update(filter_dict)
        
        # Execute search
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'score': 1 - results['distances'][0][i] if results['distances'] else 0.0,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                }
                
                # Parse JSON fields
                for key in ['page_refs', 'section_hierarchy', 'first_bbox']:
                    if key in result['metadata']:
                        try:
                            result['metadata'][key] = json.loads(result['metadata'][key])
                        except:
                            pass
                
                formatted_results.append(result)
        
        return formatted_results
    
    def search_with_provenance(
        self,
        query: str,
        n_results: int = 5,
        doc_id: Optional[str] = None
    ) -> List[SourceCitation]:
        """
        Search and return results as SourceCitations with provenance.
        
        Args:
            query: Search query
            n_results: Number of results
            doc_id: Optional document filter
            
        Returns:
            List of SourceCitation objects
        """
        results = self.search(query, n_results, doc_id=doc_id)
        citations = []
        
        for result in results:
            metadata = result['metadata']
            
            # Parse page references
            page_refs = metadata.get('page_refs', [1])
            if isinstance(page_refs, str):
                try:
                    page_refs = json.loads(page_refs)
                except:
                    page_refs = [1]
            
            # Get first page
            first_page = page_refs[0] if page_refs else 1
            
            # Get bounding box
            bbox = None
            if 'first_bbox' in metadata:
                try:
                    bbox_data = json.loads(metadata['first_bbox'])
                    bbox = {
                        'x0': bbox_data.get('x0', 0),
                        'y0': bbox_data.get('y0', 0),
                        'x1': bbox_data.get('x1', 0),
                        'y1': bbox_data.get('y1', 0)
                    }
                except:
                    pass
            
            citation = SourceCitation(
                document_name=f"{metadata.get('doc_id', 'unknown')}.pdf",
                page_number=first_page,
                bbox=bbox,
                content_hash=metadata.get('content_hash', ''),
                extracted_text=result['content'][:500],  # Limit text length
                confidence=result['score']
            )
            citations.append(citation)
        
        return citations
    
    def get_chunks_by_document(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        results = self.collection.get(
            where={"doc_id": doc_id}
        )
        
        chunks = []
        if results['ids']:
            for i in range(len(results['ids'])):
                chunk = {
                    'id': results['ids'][i],
                    'content': results['documents'][i] if results['documents'] else '',
                    'metadata': results['metadatas'][i] if results['metadatas'] else {}
                }
                chunks.append(chunk)
        
        return chunks
    
    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks for a document"""
        chunks = self.get_chunks_by_document(doc_id)
        if chunks:
            ids = [c['id'] for c in chunks]
            self.collection.delete(ids=ids)
            print(f"✅ Deleted {len(ids)} chunks for document {doc_id}")
            return len(ids)
        return 0
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        
        # Get sample metadata for stats
        if count > 0:
            sample = self.collection.get(limit=min(100, count))
            doc_ids = set()
            chunk_types = {}
            
            for metadata in sample['metadatas']:
                if metadata:
                    doc_ids.add(metadata.get('doc_id', 'unknown'))
                    ctype = metadata.get('chunk_type', 'unknown')
                    chunk_types[ctype] = chunk_types.get(ctype, 0) + 1
            
            stats = {
                'total_chunks': count,
                'documents': len(doc_ids),
                'chunk_types': chunk_types,
                'persist_directory': self.persist_directory
            }
        else:
            stats = {
                'total_chunks': 0,
                'documents': 0,
                'chunk_types': {},
                'persist_directory': self.persist_directory
            }
        
        return stats
    
    def reset(self):
        """Reset the collection (dangerous - deletes all data)"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )
        print("✅ Reset vector store collection")


class VectorStoreManager:
    """
    Manages multiple vector stores for different collections/projects.
    """
    
    def __init__(self, base_dir: str = ".refinery/vectors"):
        self.base_dir = base_dir
        self.stores: Dict[str, VectorStore] = {}
    
    def get_store(
        self,
        collection_name: str,
        embedding_model: str = "all-MiniLM-L6-v2"
    ) -> VectorStore:
        """Get or create a vector store for a collection"""
        if collection_name not in self.stores:
            persist_dir = os.path.join(self.base_dir, collection_name)
            self.stores[collection_name] = VectorStore(
                persist_directory=persist_dir,
                collection_name=collection_name,
                embedding_model=embedding_model
            )
        return self.stores[collection_name]
    
    def list_collections(self) -> List[str]:
        """List all available collections"""
        collections = []
        if os.path.exists(self.base_dir):
            collections = [d for d in os.listdir(self.base_dir) 
                          if os.path.isdir(os.path.join(self.base_dir, d))]
        return collections