"""
RAG engine combining ChromaDB vector store with BM25 keyword search.
"""

import llm
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import fcntl
import time
import pickle
import hashlib

from .config import RAGConfig, get_rag_config_dir
from .loaders import DocumentLoader
from .chunking import RecursiveCharacterTextSplitter, create_splitter_for_file
from .dedup import Deduplicator
from .search import create_hybrid_searcher
from .progress import progress_status, print_warning


class RAGEngine:
    """
    Main RAG engine combining vector and keyword search.
    """

    def __init__(self, collection_name: str, embedding_model: Optional[str] = None):
        """
        Initialize RAG engine for a collection.

        Args:
            collection_name: Name of the RAG collection
            embedding_model: Optional embedding model ID (uses llm default if not specified)

        Raises:
            ValueError: If collection_name is invalid or contains path traversal sequences
        """
        # Validate collection name to prevent path traversal attacks
        from .config import validate_collection_name
        validate_collection_name(collection_name)

        self.collection_name = collection_name
        self.last_search_sources: List[str] = []

        # Load global configuration (singleton)
        self.config = RAGConfig()

        # Override embedding model if specified (temporary, not saved)
        self.embedding_model_override = embedding_model

        # Initialize ChromaDB
        config_dir = get_rag_config_dir()
        chroma_dir = config_dir / collection_name / "chromadb"
        chroma_dir.mkdir(parents=True, exist_ok=True)

        # Setup file lock for concurrent access protection
        self.lock_file = config_dir / collection_name / ".lock"
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fd = None

        # BM25 cache file for persistence
        self.bm25_cache_file = config_dir / collection_name / "bm25_cache.pkl"

        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": f"RAG collection: {collection_name}"}
        )

        # Initialize components
        self.loader = DocumentLoader(self.config.get_loaders())
        self.deduplicator = Deduplicator()
        self.hybrid_search = create_hybrid_searcher(self.config.to_dict())

        # Track BM25 index availability (set to True after successful rebuild)
        self.bm25_available = False

        # BM25/ChromaDB ID mapping
        # BM25 uses sequential integer IDs (0, 1, 2...)
        # ChromaDB uses string IDs ("{hash}_{index}")
        # We need bidirectional mapping for hybrid search
        self.bm25_to_chroma: Dict[int, str] = {}  # BM25 int ID -> ChromaDB string ID
        self.chroma_to_bm25: Dict[str, int] = {}  # ChromaDB string ID -> BM25 int ID
        self._max_bm25_id: int = -1  # Track highest BM25 ID used (prevents collisions after deletions)

        # Get embedding model with consistency check
        # Priority: override > global config > llm default
        model_id = self.embedding_model_override or self.config.get_embedding_model()
        if not model_id:
            # Fall back to llm's default embedding model
            model_id = llm.get_default_embedding_model()

        if not model_id:
            raise ValueError(
                "No embedding model available. Configure one with:\n"
                "  llm embed-models default <model>  # Set default\n"
                "  llm keys set <provider>           # Set API key\n"
                "Or specify a model with --model. See: llm embed-models list"
            )

        try:
            self.embedding_model = llm.get_embedding_model(model_id)
        except Exception as e:
            raise ValueError(
                f"Failed to load embedding model '{model_id}'. "
                "Check API keys and model availability."
            ) from e

        # Check embedding model consistency with existing collection
        collection_metadata = self.chroma_collection.metadata or {}
        stored_model = collection_metadata.get("embedding_model")

        # Get the actual model ID being used (not just what was explicitly specified)
        # This prevents silent model mismatches when user doesn't specify --model
        # Try multiple attributes to get a stable identifier
        actual_model_id = None
        if self.embedding_model:
            # Try model_id first (most embedding models have this)
            actual_model_id = getattr(self.embedding_model, 'model_id', None)
            # Try 'name' attribute if model_id is not available
            if not actual_model_id:
                actual_model_id = getattr(self.embedding_model, 'name', None)
            # Last resort: use the class name (better than None for detection)
            if not actual_model_id:
                actual_model_id = type(self.embedding_model).__name__
        # Note: Do NOT fall back to model_id variable - it may be None for default models
        # which would bypass the consistency check

        if stored_model and actual_model_id and stored_model != actual_model_id:
            raise ValueError(
                f"Collection '{collection_name}' uses embedding model '{stored_model}' "
                f"but you specified '{actual_model_id}'. Cannot mix embedding models in one collection. "
                f"Create a new collection or use the same model."
            )

        # Get embedding dimension by generating a test embedding
        # This is done once at initialization to enable dimension validation
        self._embedding_dimension = None
        try:
            test_embedding = self.embedding_model.embed("test")
            # Handle both flat and nested embedding formats
            if hasattr(test_embedding, '__len__') and len(test_embedding) > 0:
                first_elem = test_embedding[0]
                # Check if nested (list of embeddings) vs flat (single embedding)
                if hasattr(first_elem, '__len__') and not isinstance(first_elem, (str, bytes)):
                    self._embedding_dimension = len(first_elem)
                else:
                    self._embedding_dimension = len(test_embedding)
        except Exception:
            pass  # _embedding_dimension remains None

        # Check embedding dimension consistency with existing collection
        stored_dimension = collection_metadata.get("embedding_dimension")
        if stored_dimension and self._embedding_dimension:
            try:
                stored_dim_int = int(stored_dimension)
            except (ValueError, TypeError):
                # Corrupted metadata - log warning and skip dimension check
                print_warning(f"Invalid embedding_dimension in metadata: {stored_dimension!r}, skipping validation")
                stored_dim_int = None

            if stored_dim_int is not None and stored_dim_int != self._embedding_dimension:
                raise ValueError(
                    f"Collection '{collection_name}' uses {stored_dim_int}-dimensional embeddings "
                    f"but model '{actual_model_id}' produces {self._embedding_dimension}-dimensional embeddings. "
                    f"Cannot mix embedding dimensions in one collection. "
                    f"Create a new collection or use a compatible model."
                )

        # Store embedding model and dimension in collection metadata
        # Update if: (1) new collection, or (2) existing collection missing dimension
        needs_metadata_update = False
        metadata = dict(collection_metadata) if collection_metadata else {}
        metadata["description"] = f"RAG collection: {collection_name}"

        if not stored_model and actual_model_id:
            metadata["embedding_model"] = actual_model_id
            needs_metadata_update = True

        if not stored_dimension and self._embedding_dimension:
            metadata["embedding_dimension"] = str(self._embedding_dimension)
            needs_metadata_update = True

        if needs_metadata_update:
            self.chroma_collection.modify(metadata=metadata)

        # Load existing document hashes for deduplication
        self._load_existing_hashes()

        # Try to load BM25 index from cache, rebuild if cache is stale or missing
        # Use lock to prevent concurrent rebuilds from corrupting cache
        if not self._acquire_lock(timeout=30.0):
            raise RuntimeError(f"Failed to acquire lock for collection '{collection_name}' during initialization")
        try:
            if not self._load_bm25_cache():
                self._rebuild_bm25_from_chromadb()
                # Save the freshly rebuilt cache
                self._save_bm25_cache()
        finally:
            self._release_lock()

    def _load_existing_hashes(self):
        """Load existing document hashes from ChromaDB metadata."""
        try:
            # Get all documents to populate deduplicator
            result = self.chroma_collection.get(include=["metadatas"])
            if result and result.get("metadatas"):
                for metadata in result["metadatas"]:
                    if metadata and "hash" in metadata:
                        self.deduplicator.add_hash(metadata["hash"])
        except Exception as e:
            print_warning(f"Failed to load existing hashes: {e}")

    def _rebuild_bm25_from_chromadb(self):
        """Rebuild BM25 index and ID mappings from existing ChromaDB documents."""
        try:
            # Clear existing mappings and BM25 index FIRST
            # This must happen before any early returns to prevent stale data
            self.bm25_to_chroma.clear()
            self.chroma_to_bm25.clear()
            self.hybrid_search.clear()

            # Get all documents from ChromaDB with metadata
            result = self.chroma_collection.get(include=["documents", "metadatas"])

            if not result or not result.get("ids") or not result.get("documents"):
                # Empty collection - mark BM25 as unavailable and return
                self.bm25_available = False
                return

            chroma_ids = result["ids"]
            documents = result["documents"]
            metadatas = result.get("metadatas", [])

            # Sort by IDs for deterministic ordering (prevents BM25 score variations across rebuilds)
            # Create list of (id, doc, metadata) tuples and sort by ID
            sorted_data = sorted(
                zip(chroma_ids, documents, metadatas if metadatas else [{}] * len(chroma_ids)),
                key=lambda x: x[0]
            )
            chroma_ids, documents, metadatas = zip(*sorted_data) if sorted_data else ([], [], [])

            # First pass: collect existing BM25 IDs to prevent collisions
            used_bm25_ids = set()
            documents_with_ids = []
            documents_without_ids = []

            for chroma_id, doc, metadata in zip(chroma_ids, documents, metadatas):
                if metadata and "bm25_id" in metadata:
                    # Document already has a BM25 ID
                    try:
                        bm25_id = int(metadata["bm25_id"])
                        used_bm25_ids.add(bm25_id)
                        documents_with_ids.append((chroma_id, doc, metadata, bm25_id))
                    except (ValueError, TypeError):
                        # Corrupted bm25_id metadata - treat as needing new ID
                        documents_without_ids.append((chroma_id, doc, metadata or {}))
                else:
                    # Document needs a BM25 ID assigned
                    documents_without_ids.append((chroma_id, doc, metadata or {}))

            # Second pass: assign BM25 IDs to documents without them using gap-finding
            next_candidate_id = 0
            need_backfill = []

            for chroma_id, doc, metadata in documents_without_ids:
                # Find next available ID (fill gaps first, then extend)
                while next_candidate_id in used_bm25_ids:
                    next_candidate_id += 1

                bm25_id = next_candidate_id
                used_bm25_ids.add(bm25_id)
                next_candidate_id += 1

                # Add to documents list and mark for backfill
                documents_with_ids.append((chroma_id, doc, metadata, bm25_id))
                need_backfill.append((chroma_id, bm25_id, metadata))

            # Third pass: build BM25 index and mappings with all documents
            bm25_docs = []
            max_bm25_id = -1
            for chroma_id, doc, metadata, bm25_id in documents_with_ids:

                # Store bidirectional mapping
                self.bm25_to_chroma[bm25_id] = chroma_id
                self.chroma_to_bm25[chroma_id] = bm25_id
                # Track maximum BM25 ID
                max_bm25_id = max(max_bm25_id, bm25_id)
                # Add to BM25 with integer ID
                bm25_docs.append((bm25_id, doc))

            # Validate no ID collisions occurred (detect corrupted metadata)
            if len(self.bm25_to_chroma) != len(set(self.bm25_to_chroma.values())):
                raise RuntimeError(
                    "BM25 ID collision detected - collection metadata may be corrupted. "
                    "Consider deleting and recreating the collection."
                )

            # Backfill missing bm25_id metadata for old collections
            if need_backfill:
                with progress_status(f"Backfilling metadata for {len(need_backfill)} documents...") as status:
                    for idx, (chroma_id, bm25_id, metadata) in enumerate(need_backfill):
                        status.update(f"[cyan]Backfilling metadata: {idx + 1}/{len(need_backfill)}[/]")
                        # Update metadata with bm25_id
                        metadata["bm25_id"] = str(bm25_id)
                        try:
                            # ChromaDB update requires all fields
                            self.chroma_collection.update(
                                ids=[chroma_id],
                                metadatas=[metadata]
                            )
                        except Exception as e:
                            print_warning(f"Failed to backfill bm25_id for {chroma_id}: {e}")

            if bm25_docs:
                self.hybrid_search.add_documents(bm25_docs)
                # Finalize to rebuild BM25 index once after all documents added
                self.hybrid_search.finalize()

            # Update max BM25 ID tracker
            self._max_bm25_id = max_bm25_id

            # Mark BM25 as available after successful rebuild
            self.bm25_available = True

        except Exception as e:
            print_warning(f"Failed to rebuild BM25 index: {e}")
            self.bm25_available = False

    def _get_collection_state_hash(self) -> str:
        """
        Compute a hash representing the current state of the ChromaDB collection.
        Used to detect when BM25 cache is stale.

        For large collections (>10k docs), uses sampling strategy to avoid memory issues.

        Returns:
            SHA256 hash of collection state
        """
        try:
            # Get collection count as primary state indicator
            count = self.chroma_collection.count()

            if count == 0:
                return hashlib.sha256(b"empty").hexdigest()

            # For small collections (<= 10k), use all IDs
            if count <= 10000:
                result = self.chroma_collection.get(limit=count, include=[])
                ids = result.get("ids", [])
                state_str = f"{count}:{','.join(sorted(ids))}"
                return hashlib.sha256(state_str.encode()).hexdigest()

            # For large collections, use count + sample of IDs
            # Get first 1000 and last 1000 IDs for efficient change detection
            first_batch = self.chroma_collection.get(limit=1000, offset=0, include=[])
            first_ids = first_batch.get("ids", [])

            # Get last batch (offset = count - 1000)
            last_offset = max(0, count - 1000)
            last_batch = self.chroma_collection.get(limit=1000, offset=last_offset, include=[])
            last_ids = last_batch.get("ids", [])

            # Create hash from count + sampled IDs
            # This detects: additions (count changes), deletions (count changes),
            # and modifications to beginning/end of collection
            state_str = f"{count}:first={','.join(sorted(first_ids))}:last={','.join(sorted(last_ids))}"
            return hashlib.sha256(state_str.encode()).hexdigest()
        except Exception:
            # If we can't get state, return empty hash (will force rebuild)
            return ""

    def _get_cache_signing_key(self) -> bytes:
        """
        Generate a signing key for cache files based on collection name and user config.
        This prevents tampering with cache files by requiring a valid signature.

        Returns:
            Signing key bytes
        """
        # Use collection name as salt to make keys unique per collection
        key_material = f"llm-rag-cache-v1:{self.collection_name}".encode('utf-8')
        return hashlib.sha256(key_material).digest()

    def _save_bm25_cache(self):
        """Save BM25 index, ID mappings, and state hash to disk with HMAC signature."""
        try:
            # Compute state hash while we have consistent data
            cache_data = {
                "state_hash": self._get_collection_state_hash(),
                "hybrid_search": self.hybrid_search,
                "bm25_to_chroma": self.bm25_to_chroma,
                "chroma_to_bm25": self.chroma_to_bm25,
                "max_bm25_id": self._max_bm25_id,
            }

            # Serialize data
            serialized = pickle.dumps(cache_data, protocol=pickle.HIGHEST_PROTOCOL)

            # Sign with HMAC to prevent tampering
            signing_key = self._get_cache_signing_key()
            signature = hashlib.sha256(signing_key + serialized).digest()

            # Write signature + data atomically
            with open(self.bm25_cache_file, 'wb') as f:
                f.write(signature)  # First 32 bytes = signature
                f.write(serialized)  # Rest = pickled data

        except Exception as e:
            # Non-fatal - just log warning
            print_warning(f"Failed to save BM25 cache: {e}")

    def _load_bm25_cache(self) -> bool:
        """
        Load BM25 index from cache if it exists, is not stale, and has valid signature.

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        if not self.bm25_cache_file.exists():
            return False

        try:
            # Read entire cache file
            cache_bytes = self.bm25_cache_file.read_bytes()

            # Check minimum size (32 bytes for signature + some data)
            if len(cache_bytes) < 33:
                print_warning("BM25 cache file too small, rebuilding...")
                return False

            # Split signature and data
            stored_signature = cache_bytes[:32]
            serialized = cache_bytes[32:]

            # Verify signature to prevent tampering/corruption
            signing_key = self._get_cache_signing_key()
            expected_signature = hashlib.sha256(signing_key + serialized).digest()

            if stored_signature != expected_signature:
                print_warning("BM25 cache signature invalid (corrupted or tampered), rebuilding...")
                return False

            # Signature valid - deserialize
            cache_data = pickle.loads(serialized)

            # Verify cache is not stale
            cached_state = cache_data.get("state_hash", "")
            current_state = self._get_collection_state_hash()

            if cached_state != current_state:
                return False  # Cache stale, needs rebuild

            # Load cached data
            self.hybrid_search = cache_data["hybrid_search"]
            self.bm25_to_chroma = cache_data["bm25_to_chroma"]
            self.chroma_to_bm25 = cache_data["chroma_to_bm25"]
            # Load max_bm25_id if present (backward compatibility for old caches)
            self._max_bm25_id = cache_data.get("max_bm25_id", -1)

            # Mark BM25 as available after successful cache load
            self.bm25_available = True
            return True
        except Exception as e:
            print_warning(f"Failed to load BM25 cache: {e}")
            return False

    def _embed_chunks(self, chunks: List[str], max_retries: int = 3, max_memory_mb: int = 500) -> List[List[float]]:
        """
        Generate embeddings for chunks with batching support and retry logic.

        Args:
            chunks: List of text chunks to embed
            max_retries: Maximum number of retry attempts for failed embeddings
            max_memory_mb: Maximum estimated memory usage in MB (default 500MB)

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If estimated memory usage exceeds limit
        """
        # Estimate memory usage for input chunks
        total_chunk_bytes = sum(len(chunk.encode('utf-8')) for chunk in chunks)
        total_chunk_mb = total_chunk_bytes / 1_000_000

        # Estimate memory for embeddings (assume 1536 dimensions * 4 bytes per float)
        # This is conservative - actual size depends on embedding model
        estimated_embedding_mb = (len(chunks) * 1536 * 4) / 1_000_000

        # Total estimated memory
        estimated_total_mb = total_chunk_mb + estimated_embedding_mb

        if estimated_total_mb > max_memory_mb:
            raise ValueError(
                f"Estimated memory usage ({estimated_total_mb:.1f}MB) exceeds limit ({max_memory_mb}MB). "
                f"Document has {len(chunks)} chunks totaling {total_chunk_mb:.1f}MB of text. "
                f"Consider splitting the document into smaller files or processing in batches."
            )

        embeddings = []

        # Try to use batch embedding if available
        if hasattr(self.embedding_model, 'embed_batch') and callable(self.embedding_model.embed_batch):
            # Batch embedding supported - use optimal batch size
            batch_size = self._calculate_batch_size(len(chunks))

            with progress_status(f"Embedding {len(chunks)} chunks...") as status:
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]

                    # Retry logic with exponential backoff
                    for retry in range(max_retries):
                        try:
                            batch_embeddings = self.embedding_model.embed_batch(batch)
                            embeddings.extend(batch_embeddings)
                            status.update(f"[cyan]Embedding: {min(i + batch_size, len(chunks))}/{len(chunks)} chunks[/]")
                            break  # Success, exit retry loop
                        except Exception as e:
                            if retry < max_retries - 1:
                                wait_time = 2 ** retry  # Exponential backoff: 1s, 2s, 4s
                                print_warning(f"Batch embedding failed (attempt {retry + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                                time.sleep(wait_time)
                            else:
                                # Final attempt failed, fallback to one-by-one
                                print_warning(f"Batch embedding failed after {max_retries} attempts, falling back to sequential")
                                for chunk_idx, chunk in enumerate(batch):
                                    chunk_num = i + chunk_idx + 1  # Global chunk number
                                    for seq_retry in range(max_retries):
                                        try:
                                            embeddings.append(self.embedding_model.embed(chunk))
                                            status.update(f"[cyan]Embedding: {chunk_num}/{len(chunks)} chunks[/]")
                                            break
                                        except Exception as seq_e:
                                            if seq_retry < max_retries - 1:
                                                time.sleep(2 ** seq_retry)
                                            else:
                                                # Provide detailed error context
                                                chunk_preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                                                raise RuntimeError(
                                                    f"Failed to embed chunk {chunk_num}/{len(chunks)} "
                                                    f"after {max_retries} attempts. "
                                                    f"Content preview: '{chunk_preview}'. "
                                                    f"Error: {seq_e}"
                                                ) from seq_e
        else:
            # No batch support - embed one by one with progress indicator and retry
            with progress_status(f"Embedding {len(chunks)} chunks...") as status:
                for i, chunk in enumerate(chunks):
                    for retry in range(max_retries):
                        try:
                            embeddings.append(self.embedding_model.embed(chunk))
                            status.update(f"[cyan]Embedding: {i + 1}/{len(chunks)} chunks[/]")
                            break
                        except Exception as e:
                            if retry < max_retries - 1:
                                wait_time = 2 ** retry
                                time.sleep(wait_time)
                            else:
                                raise RuntimeError(f"Failed to embed chunk {i + 1} after {max_retries} attempts: {e}")

        return embeddings

    def _calculate_batch_size(self, num_chunks: int) -> int:
        """
        Calculate optimal batch size for embedding.

        Args:
            num_chunks: Number of chunks to embed

        Returns:
            Optimal batch size
        """
        # Default batch size
        default_batch_size = 100

        # Try to get model-specific limits
        try:
            # Check if model has max_batch_size attribute
            if hasattr(self.embedding_model, 'max_batch_size'):
                max_batch = self.embedding_model.max_batch_size
                return min(max_batch, num_chunks, default_batch_size)

            # Check if model has max_tokens and we can calculate batch size
            if hasattr(self.embedding_model, 'max_input_tokens'):
                chunk_size = self.config.get_chunk_size()  # Characters
                max_tokens = self.embedding_model.max_input_tokens  # Tokens
                # Convert tokens to approximate characters (1 token ≈ 4 chars for English)
                CHARS_PER_TOKEN = 4
                max_chars = max_tokens * CHARS_PER_TOKEN
                calculated_batch = max(1, max_chars // chunk_size)
                return min(calculated_batch, num_chunks, default_batch_size)
        except Exception:
            pass

        # Fallback to reasonable default
        return min(default_batch_size, num_chunks)

    def add_document(self, path: str, refresh: bool = False) -> Dict[str, Any]:
        """
        Add a document to the RAG collection.

        Args:
            path: Document path or protocol path (e.g., git:https://...)
                 Supports recursive URL crawling with ** pattern
            refresh: If True, reindex even if document exists

        Returns:
            Dictionary with status information
        """
        # Acquire lock for write operation
        if not self._acquire_lock():
            return {"status": "error", "path": path, "error": "Failed to acquire collection lock (timeout)"}

        try:
            # Check if this is a multi-document path (recursive crawl)
            if self.loader.is_recursive_url(path):
                return self._add_multi_documents(path, refresh)

            # Load single document content
            content = self.loader.load(path)

            # Use internal method for actual processing
            return self._add_single_document(path, content, refresh)

        except Exception as e:
            return {"status": "error", "path": path, "error": str(e)}
        finally:
            self._release_lock()

    def _add_multi_documents(self, path_pattern: str, refresh: bool = False) -> Dict[str, Any]:
        """
        Add multiple documents from a recursive crawl.

        Args:
            path_pattern: URL pattern with ** for recursive crawling
            refresh: If True, reindex even if document exists

        Returns:
            Dictionary with status information
        """
        try:
            # Load all documents from the pattern
            documents = self.loader.load_multi(path_pattern)

            if not documents:
                return {"status": "skipped", "reason": "no documents found", "path": path_pattern}

            # Track overall statistics
            total_added = 0
            total_skipped = 0
            total_chunks = 0
            errors = []

            # Add each document using internal method (avoids recursive check)
            for doc in documents:
                result = self._add_single_document(doc.path, doc.content, refresh)
                if result["status"] == "success":
                    total_added += 1
                    total_chunks += result.get("chunks", 0)
                elif result["status"] == "skipped":
                    total_skipped += 1
                elif result["status"] == "error":
                    errors.append(f"{doc.path}: {result.get('error', 'unknown')}")

            return {
                "status": "success" if total_added > 0 else "skipped",
                "path": path_pattern,
                "documents_added": total_added,
                "documents_skipped": total_skipped,
                "chunks": total_chunks,
                "errors": errors
            }

        except Exception as e:
            return {"status": "error", "path": path_pattern, "error": str(e)}

    def _add_single_document(self, path: str, content: str, refresh: bool = False) -> Dict[str, Any]:
        """
        Internal method to add a single document with pre-loaded content.

        Args:
            path: Document path (for metadata)
            content: Pre-loaded document content
            refresh: If True, reindex even if document exists

        Returns:
            Dictionary with status information
        """
        try:
            if not content.strip():
                return {"status": "skipped", "reason": "empty content", "path": path}

            # Check for duplicate (unless refreshing)
            content_hash = self.deduplicator.hash_content(content)
            if not refresh and self.deduplicator.contains_hash(content_hash):
                return {"status": "skipped", "reason": "duplicate", "path": path, "hash": content_hash}

            # If refreshing and document exists, delete old chunks first
            if refresh and self.deduplicator.contains_hash(content_hash):
                try:
                    # Find all chunks with this document hash
                    existing = self.chroma_collection.get(
                        where={"doc_hash": content_hash},
                        include=["metadatas"]
                    )
                    if existing and existing.get("ids"):
                        old_ids = existing["ids"]
                        # Remove from ChromaDB
                        self.chroma_collection.delete(ids=old_ids)
                        # Remove from BM25 mappings
                        for old_id in old_ids:
                            if old_id in self.chroma_to_bm25:
                                old_bm25_id = self.chroma_to_bm25.pop(old_id)
                                self.bm25_to_chroma.pop(old_bm25_id, None)
                        # Rebuild BM25 index without these documents
                        self._rebuild_bm25_from_chromadb()
                        self._save_bm25_cache()
                except Exception as e:
                    print_warning(f"Failed to delete old document chunks during refresh: {e}")

            # Create appropriate splitter based on file type
            if self.loader.is_protocol_path(path) or not Path(path).exists():
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.get_chunk_size(),
                    chunk_overlap=self.config.get_chunk_overlap()
                )
            else:
                splitter = create_splitter_for_file(
                    path,
                    chunk_size=self.config.get_chunk_size(),
                    chunk_overlap=self.config.get_chunk_overlap()
                )

            # Split into chunks
            chunks = splitter.split_text(content)

            if not chunks:
                return {"status": "skipped", "reason": "no chunks", "path": path}

            # Deduplicate chunks while tracking original indices
            unique_chunks = []
            chunk_hashes = []
            original_indices = []

            for i, chunk in enumerate(chunks):
                # Hash the chunk to check if it's duplicate
                chunk_hash = self.deduplicator.hash_content(chunk)
                is_duplicate = self.deduplicator.contains_hash(chunk_hash)

                # Add to unique chunks if new or refreshing
                if not is_duplicate or refresh:
                    unique_chunks.append(chunk)
                    chunk_hashes.append(chunk_hash)
                    original_indices.append(i)  # i is always the correct original position

                # Update deduplicator only for truly new chunks
                if not is_duplicate:
                    self.deduplicator.add_hash(chunk_hash)

            if not unique_chunks:
                return {"status": "skipped", "reason": "all chunks duplicate", "path": path}

            # Apply contextual headers if enabled (improves embedding quality)
            # We embed with headers for context, but store original chunks for clean display
            if self.config.get_contextual_headers():
                # Get source name for header
                source_name = Path(path).name if not self.loader.is_protocol_path(path) else path
                chunks_for_embedding = [
                    f"Source: {source_name}\n\n{chunk}"
                    for chunk in unique_chunks
                ]
            else:
                chunks_for_embedding = unique_chunks

            # Generate embeddings with batching support
            # Uses header-prefixed chunks if contextual_headers enabled
            embeddings = self._embed_chunks(chunks_for_embedding)

            # Get next available BM25 ID (max + 1) to assign new sequential IDs
            # This prevents collisions after deletions
            next_bm25_id = self._max_bm25_id + 1

            # Prepare metadata and IDs with BM25 IDs
            doc_ids = [f"{content_hash}_{i}" for i in range(len(unique_chunks))]
            metadatas = [
                {
                    "source": path,
                    "hash": chunk_hash,
                    "doc_hash": content_hash,
                    "chunk_index": original_idx,  # Use original position from document
                    "bm25_id": str(next_bm25_id + i)  # Store as string for ChromaDB compatibility
                }
                for i, (chunk_hash, original_idx) in enumerate(zip(chunk_hashes, original_indices))
            ]

            # Add to ChromaDB
            self.chroma_collection.add(
                ids=doc_ids,
                embeddings=embeddings,
                documents=unique_chunks,
                metadatas=metadatas
            )

            # Add to BM25 index with proper ID mapping
            # Wrap in try-except to rollback ChromaDB on failure
            # Save mapping state before modifications for rollback
            saved_bm25_to_chroma = self.bm25_to_chroma.copy()
            saved_chroma_to_bm25 = self.chroma_to_bm25.copy()
            saved_max_bm25_id = self._max_bm25_id

            try:
                bm25_docs = []
                for i, (doc_id, chunk) in enumerate(zip(doc_ids, unique_chunks)):
                    bm25_id = next_bm25_id + i
                    # Store bidirectional mapping
                    self.bm25_to_chroma[bm25_id] = doc_id
                    self.chroma_to_bm25[doc_id] = bm25_id
                    bm25_docs.append((bm25_id, chunk))

                self.hybrid_search.add_documents(bm25_docs)
                # Finalize to rebuild BM25 index once after all documents added
                self.hybrid_search.finalize()

                # Update max BM25 ID tracker after successful addition
                self._max_bm25_id = next_bm25_id + len(unique_chunks) - 1
            except Exception as e:
                # Rollback both ChromaDB and ID mappings to maintain consistency
                try:
                    self.chroma_collection.delete(ids=doc_ids)
                except Exception as rollback_error:
                    # Log rollback failure but raise original error
                    print_warning(f"Failed to rollback ChromaDB after BM25 error: {rollback_error}")

                # Restore mapping state
                self.bm25_to_chroma = saved_bm25_to_chroma
                self.chroma_to_bm25 = saved_chroma_to_bm25
                self._max_bm25_id = saved_max_bm25_id

                raise ValueError(f"Failed to add documents to BM25 index: {e}") from e

            # Add document hash to deduplicator for future duplicate detection
            self.deduplicator.add_hash(content_hash)

            # Update BM25 cache after successful addition
            self._save_bm25_cache()

            return {
                "status": "success",
                "path": path,
                "chunks": len(unique_chunks),
                "hash": content_hash
            }

        except Exception as e:
            return {"status": "error", "path": path, "error": str(e)}

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the RAG collection.

        Args:
            query: Search query
            top_k: Number of results (uses config default if not specified)
            mode: Search mode override (vector, keyword, hybrid)
            filters: Metadata filters (e.g., {"source": "file.py"} or {"source_contains": "src/"})

        Returns:
            List of result dictionaries with content and metadata

        Raises:
            ValueError: If query is empty or whitespace-only
        """
        # Validate query is not empty
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")

        # Validate query length to prevent memory issues and API failures
        max_query_length = 10_000  # 10k characters
        if len(query) > max_query_length:
            raise ValueError(
                f"Query length ({len(query)} characters) exceeds maximum allowed length ({max_query_length} characters). "
                f"Please shorten your query or split it into multiple searches."
            )

        top_k = top_k or self.config.get_top_k()
        mode = mode or self.config.get_search_mode()

        # Fall back to vector-only if BM25 is not available
        if mode in ("hybrid", "keyword") and not self.bm25_available:
            print_warning("BM25 index not available, falling back to vector-only search")
            mode = "vector"

        # Get query embedding
        query_embedding = self.embedding_model.embed(query)

        # Vector search via ChromaDB
        # Note: ChromaDB expects query_embeddings as list of lists (2D array)
        # Validate and normalize embedding format - handle various types (list, numpy array, etc.)

        # First, ensure we have a sequence (convert numpy arrays, etc. to list)
        if not isinstance(query_embedding, (list, tuple)):
            try:
                query_embedding = list(query_embedding)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Embedding model returned non-iterable type {type(query_embedding)} "
                    f"for query: {query[:50]}..."
                )

        if len(query_embedding) == 0:
            raise ValueError(
                f"Embedding model returned empty embedding for query: {query[:50]}..."
            )

        # Check if already nested (2D array)
        # Exclude strings - they are iterable but not valid embedding containers
        first_elem = query_embedding[0]
        is_nested = (
            isinstance(first_elem, (list, tuple)) or
            (hasattr(first_elem, '__iter__') and not isinstance(first_elem, (str, bytes)))
        )
        if is_nested:
            # Nested structure - validate it's properly formatted
            try:
                # Convert to list of lists and validate numbers
                query_embeddings = []
                for sublist in query_embedding:
                    if isinstance(sublist, (str, bytes)):
                        raise ValueError(
                            f"Expected list of numbers, got {type(sublist).__name__}. "
                            f"Strings/bytes are not valid embedding values."
                        )
                    if hasattr(sublist, '__iter__'):
                        flat_sublist = [float(x) for x in sublist]  # Convert to float, validates numeric
                        if len(flat_sublist) == 0:
                            raise ValueError("Empty sublist in nested embedding")
                        query_embeddings.append(flat_sublist)
                    else:
                        raise ValueError(
                            f"Expected list of numbers, got non-iterable {type(sublist).__name__}"
                        )
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Embedding model returned invalid nested structure: {e} "
                    f"for query: {query[:50]}..."
                )
        else:
            # Flat list of numbers - validate and wrap it
            try:
                flat_embedding = [float(x) for x in query_embedding]  # Convert to float, validates numeric
                query_embeddings = [flat_embedding]
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Embedding model returned invalid flat embedding: {e} "
                    f"for query: {query[:50]}..."
                )

        # Build where clause for filtering
        where_clause = self._build_where_clause(filters) if filters else None

        vector_results = self.chroma_collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k * 2,  # Get more for fusion
            where=where_clause
        )

        # Comprehensive ChromaDB response validation
        if not vector_results or not isinstance(vector_results, dict):
            return []

        if not vector_results.get("ids"):
            return []

        # Validate nested list structure (ChromaDB returns {"ids": [[...]], ...})
        ids_result = vector_results["ids"]
        if not isinstance(ids_result, list) or len(ids_result) == 0:
            return []

        if not isinstance(ids_result[0], list):
            raise ValueError(
                f"ChromaDB returned malformed 'ids' structure. "
                f"Expected list of lists, got list of {type(ids_result[0]).__name__}"
            )

        # Extract document IDs from first result list
        vector_doc_ids = ids_result[0]

        # Apply search mode
        if mode == "vector":
            final_ids = vector_doc_ids[:top_k]
            # Apply post-filtering for substring filters (ChromaDB doesn't support $contains on metadata)
            if filters and self._has_post_filters(filters):
                final_ids = self._filter_chroma_ids(final_ids, filters)
        elif mode == "keyword":
            # BM25 only - perform pure keyword search
            keyword_results = self.hybrid_search.bm25_index.search(query, top_k=top_k * 2)
            # Filter BM25 results by metadata if filters specified
            if filters:
                keyword_results = self._filter_bm25_results(keyword_results, filters)
            # Map BM25 IDs back to ChromaDB IDs
            final_ids = []
            for bm25_id in keyword_results[:top_k]:
                if bm25_id in self.bm25_to_chroma:
                    final_ids.append(self.bm25_to_chroma[bm25_id])
        else:  # hybrid
            # Apply query-aware weight adjustment if enabled
            if self.config.get_query_aware_weights():
                from llm_tools_rag.query_analyzer import analyze_query
                vector_weight, keyword_weight = analyze_query(query)
                # Temporarily override weights for this search
                saved_vector_weight = self.hybrid_search.vector_weight
                saved_keyword_weight = self.hybrid_search.keyword_weight
                self.hybrid_search.vector_weight = vector_weight
                self.hybrid_search.keyword_weight = keyword_weight
            else:
                saved_vector_weight = None  # Flag to skip restoration

            # Convert ChromaDB string IDs to BM25 integer IDs for hybrid search
            vector_bm25_ids = []
            for chroma_id in vector_doc_ids:
                if chroma_id in self.chroma_to_bm25:
                    vector_bm25_ids.append(self.chroma_to_bm25[chroma_id])

            try:
                if not vector_bm25_ids:
                    # Fallback to vector-only if no BM25 mappings exist
                    final_ids = vector_doc_ids[:top_k]
                else:
                    # Perform hybrid search with BM25 IDs
                    fused_bm25_ids = self.hybrid_search.search(
                        query=query,
                        vector_results=vector_bm25_ids,
                        top_k=top_k
                    )

                    # Map BM25 IDs back to ChromaDB IDs
                    final_ids = []
                    for bm25_id in fused_bm25_ids:
                        if bm25_id in self.bm25_to_chroma:
                            final_ids.append(self.bm25_to_chroma[bm25_id])

                    # Filter hybrid results by metadata (BM25 component wasn't filtered)
                    if filters:
                        final_ids = self._filter_chroma_ids(final_ids, filters)
            finally:
                # Restore original weights if they were modified (always, even on exception)
                if saved_vector_weight is not None:
                    self.hybrid_search.vector_weight = saved_vector_weight
                    self.hybrid_search.keyword_weight = saved_keyword_weight

        # Retrieve full documents
        if not final_ids:
            return []

        # Check if MMR (diversity) is enabled
        diversity_lambda = self.config.get("diversity_lambda", 1.0)
        use_mmr = diversity_lambda < 1.0

        # Fetch with embeddings if MMR is enabled
        include_fields = ["documents", "metadatas"]
        if use_mmr:
            include_fields.append("embeddings")

        result_docs = self.chroma_collection.get(
            ids=final_ids,
            include=include_fields
        )

        # ChromaDB doesn't guarantee order preservation, so we need to re-sort
        # Create mapping from ID to result data
        ids = result_docs.get("ids", [])
        documents = result_docs.get("documents", [])
        metadatas = result_docs.get("metadatas", [])
        embeddings_list = result_docs.get("embeddings", []) if use_mmr else []

        # Build ID -> (document, metadata) mapping
        id_to_data = {}
        id_to_embedding = {}
        for i, doc_id in enumerate(ids):
            content = documents[i] if documents else ""
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            id_to_data[doc_id] = (content, metadata if metadata else {})
            if embeddings_list and i < len(embeddings_list):
                id_to_embedding[doc_id] = embeddings_list[i]

        # Apply cross-encoder reranker FIRST (scores all candidates accurately)
        reranker_model = self.config.get_reranker_model()
        if reranker_model and len(final_ids) > 1:
            try:
                from llm_tools_rag.reranker import rerank

                # Collect documents to rerank
                docs_to_rerank = []
                valid_ids_for_rerank = []
                for doc_id in final_ids:
                    if doc_id in id_to_data:
                        content, _ = id_to_data[doc_id]
                        docs_to_rerank.append(content)
                        valid_ids_for_rerank.append(doc_id)

                if docs_to_rerank:
                    reranker_top_k = self.config.get_reranker_top_k() or top_k
                    final_ids = rerank(
                        query=query,
                        documents=docs_to_rerank,
                        ids=valid_ids_for_rerank,
                        model_name=reranker_model,
                        top_k=reranker_top_k
                    )
            except ImportError:
                # flashrank not installed - skip reranking
                pass
            except Exception as e:
                # Reranking failed - log warning and continue with original order
                print_warning(f"Reranking failed: {e}")

        # Apply MMR AFTER reranking (diversify the top-k from accurately scored results)
        # This ensures final results are both relevant AND diverse
        if use_mmr and id_to_embedding and len(final_ids) > 1:
            from llm_tools_rag.diversity import maximal_marginal_relevance

            # Get embeddings in final_ids order (only for IDs that have embeddings)
            ordered_embeddings = []
            valid_ids = []
            for doc_id in final_ids:
                if doc_id in id_to_embedding:
                    ordered_embeddings.append(id_to_embedding[doc_id])
                    valid_ids.append(doc_id)

            if ordered_embeddings:
                # Reorder using MMR
                final_ids = maximal_marginal_relevance(
                    query_embedding=query_embeddings[0],
                    embeddings=ordered_embeddings,
                    ids=valid_ids,
                    lambda_mult=diversity_lambda,
                    k=top_k
                )

        # Re-order results according to final_ids (preserves RRF/reranker/MMR ranking)
        results = []
        sources_set = set()

        for doc_id in final_ids:
            if doc_id in id_to_data:
                content, metadata = id_to_data[doc_id]
                results.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata
                })
                # Track source
                if metadata and "source" in metadata:
                    sources_set.add(metadata["source"])

        # Store sources for .sources command
        self.last_search_sources = sorted(list(sources_set))

        return results

    def _build_where_clause(self, filters: Dict[str, Any]) -> Optional[Dict]:
        """
        Convert filters to ChromaDB where clause.

        Note: ChromaDB only supports $contains on document content (#document),
        not on metadata fields. Substring filters like 'source_contains' must be
        handled via post-filtering in _filter_chroma_ids().

        Args:
            filters: Dictionary of filter conditions

        Returns:
            ChromaDB where clause dictionary or None
        """
        if not filters:
            return None

        clauses = []
        for key, value in filters.items():
            if key.endswith("_contains"):
                # Skip substring filters - ChromaDB doesn't support $contains on metadata
                # These are handled in post-filtering (_filter_chroma_ids, _matches_filter)
                continue
            else:
                # Exact match - supported by ChromaDB
                clauses.append({key: {"$eq": value}})

        if len(clauses) > 1:
            return {"$and": clauses}
        elif clauses:
            return clauses[0]
        return None

    def _has_post_filters(self, filters: Dict[str, Any]) -> bool:
        """Check if filters contain conditions that require post-filtering."""
        if not filters:
            return False
        return any(key.endswith("_contains") for key in filters.keys())

    def _filter_bm25_results(self, bm25_ids: List[int], filters: Dict[str, Any]) -> List[int]:
        """
        Filter BM25 results by metadata (BM25 doesn't support where clauses).

        Args:
            bm25_ids: List of BM25 integer IDs
            filters: Dictionary of filter conditions

        Returns:
            Filtered list of BM25 IDs (preserving order)
        """
        if not filters or not bm25_ids:
            return bm25_ids

        # Convert BM25 IDs to ChromaDB IDs
        chroma_ids = [self.bm25_to_chroma.get(bid) for bid in bm25_ids]
        chroma_ids = [cid for cid in chroma_ids if cid]

        if not chroma_ids:
            return []

        # Batch fetch metadata for efficiency
        result = self.chroma_collection.get(ids=chroma_ids, include=["metadatas"])

        # Build set of passing ChromaDB IDs
        passing_chroma_ids = set()
        for cid, meta in zip(result.get("ids", []), result.get("metadatas", [])):
            if meta and self._matches_filter(meta, filters):
                passing_chroma_ids.add(cid)

        # Return BM25 IDs that pass filter (preserving order)
        return [bid for bid in bm25_ids
                if self.bm25_to_chroma.get(bid) in passing_chroma_ids]

    def _matches_filter(self, metadata: Dict, filters: Dict[str, Any]) -> bool:
        """
        Check if metadata matches all filters.

        Supports:
        - Exact match: {"field": "value"}
        - Substring match: {"field_contains": "substring"}

        Args:
            metadata: Document metadata dictionary
            filters: Dictionary of filter conditions

        Returns:
            True if all filters match, False otherwise
        """
        for key, value in filters.items():
            if key.endswith("_contains"):
                # Substring match: "field_contains" checks if value is in metadata["field"]
                field_name = key[:-9]  # Remove "_contains" suffix
                if value not in metadata.get(field_name, ""):
                    return False
            elif metadata.get(key) != value:
                # Exact match
                return False
        return True

    def _filter_chroma_ids(self, chroma_ids: List[str], filters: Dict[str, Any]) -> List[str]:
        """
        Filter ChromaDB IDs by metadata.

        Args:
            chroma_ids: List of ChromaDB string IDs
            filters: Dictionary of filter conditions

        Returns:
            Filtered list of ChromaDB IDs (preserving order)
        """
        if not filters or not chroma_ids:
            return chroma_ids

        # Batch fetch metadata
        result = self.chroma_collection.get(ids=chroma_ids, include=["metadatas"])

        # Build set of passing IDs
        passing_ids = set()
        for cid, meta in zip(result.get("ids", []), result.get("metadatas", [])):
            if meta and self._matches_filter(meta, filters):
                passing_ids.add(cid)

        # Return IDs that pass filter (preserving order)
        return [cid for cid in chroma_ids if cid in passing_ids]

    def list_documents(self) -> List[str]:
        """Get list of document paths in collection from ChromaDB metadata."""
        try:
            result = self.chroma_collection.get(include=["metadatas"])
            if result and result.get("metadatas"):
                # Extract unique source paths
                sources = set()
                for metadata in result["metadatas"]:
                    if metadata and "source" in metadata:
                        sources.add(metadata["source"])
                return sorted(list(sources))
        except Exception:
            pass
        return []

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        count = self.chroma_collection.count()
        return {
            "collection": self.collection_name,
            "total_chunks": count,
            "unique_documents": len(self.list_documents()),
            "config": self.config.to_dict()
        }

    def get_last_sources(self) -> List[str]:
        """
        Get the list of source documents from the last search.

        Returns:
            List of source paths used in last search
        """
        return self.last_search_sources

    def rebuild_index(self):
        """Rebuild BM25 index and ID mappings from ChromaDB."""
        # Acquire lock for write operation to prevent concurrent modifications
        if not self._acquire_lock():
            raise RuntimeError("Failed to acquire collection lock (timeout)")

        try:
            # Check if embedding model has changed
            collection_metadata = self.chroma_collection.metadata or {}
            stored_model = collection_metadata.get("embedding_model")

            # Get actual model ID (same logic as __init__)
            actual_model_id = None
            if self.embedding_model:
                actual_model_id = getattr(self.embedding_model, 'model_id', None)
                if not actual_model_id:
                    actual_model_id = getattr(self.embedding_model, 'name', None)
                if not actual_model_id:
                    actual_model_id = type(self.embedding_model).__name__
            # Note: Do NOT fall back to model_id variable

            if stored_model and actual_model_id and stored_model != actual_model_id:
                raise ValueError(
                    f"Cannot rebuild index: embedding model has changed from '{stored_model}' to '{actual_model_id}'. "
                    f"Rebuilding only recreates the BM25 index, not embeddings. "
                    f"To fix this, you must create a new collection and re-add all documents with the new model."
                )

            # Use the same logic as initialization
            self._rebuild_bm25_from_chromadb()
            # Update cache after rebuild
            self._save_bm25_cache()
        finally:
            self._release_lock()

    def _acquire_lock(self, timeout: float = 30.0) -> bool:
        """
        Acquire exclusive lock on collection using flock.

        The flock mechanism automatically releases locks when a process dies
        (file descriptors are closed), so we don't need stale lock detection.
        We simply retry until timeout.

        Note: We intentionally do NOT delete lock files. Deleting a lock file
        while another process holds an flock on it creates a race condition
        where both processes think they have exclusive access.

        Args:
            timeout: Maximum seconds to wait for lock

        Returns:
            True if lock acquired, False if timeout
        """
        if self._lock_fd is not None:
            return True  # Already locked

        start_time = time.time()

        while True:
            try:
                self._lock_fd = open(self.lock_file, 'w')
                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Write PID and timestamp to lock file for debugging
                import os
                self._lock_fd.write(f"pid={os.getpid()},time={time.time()}\n")
                self._lock_fd.flush()
                return True
            except (IOError, OSError):
                # Close the file descriptor to prevent leak on retry
                if self._lock_fd:
                    self._lock_fd.close()
                    self._lock_fd = None

                if time.time() - start_time > timeout:
                    return False
                time.sleep(0.1)
            except Exception:
                # Clean up fd on any unexpected exception and re-raise
                if self._lock_fd:
                    self._lock_fd.close()
                    self._lock_fd = None
                raise

    def _release_lock(self):
        """Release collection lock."""
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
                self._lock_fd.close()
            except Exception:
                pass
            finally:
                self._lock_fd = None

    def delete_collection(self):
        """Delete the entire collection."""
        # Acquire lock before deletion
        if not self._acquire_lock():
            raise RuntimeError(f"Failed to acquire lock for collection '{self.collection_name}' (timeout)")

        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception as e:
            print_warning(f"Failed to delete ChromaDB collection: {e}")
        finally:
            self._release_lock()

        # Delete config
        from .config import delete_collection
        delete_collection(self.collection_name)


def get_or_create_engine(collection_name: str, embedding_model: Optional[str] = None) -> RAGEngine:
    """
    Get or create a RAG engine for a collection.

    Args:
        collection_name: Name of the collection
        embedding_model: Optional embedding model override

    Returns:
        RAGEngine instance
    """
    return RAGEngine(collection_name, embedding_model)
