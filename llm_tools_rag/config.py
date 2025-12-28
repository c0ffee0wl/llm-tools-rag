"""
YAML-based global configuration management for RAG.
All collections share the same global configuration.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from copy import deepcopy


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    Values in 'override' take precedence over 'base'.
    Nested dicts are merged recursively instead of replaced.

    Args:
        base: Base dictionary with default values
        override: Dictionary with override values

    Returns:
        Merged dictionary
    """
    # Shallow copy base first (only top level)
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = deep_merge(result[key], value)
        else:
            # Deep copy only leaf values to prevent mutation
            result[key] = deepcopy(value)

    return result


# Default configuration matching aichat's structure
DEFAULT_CONFIG = {
    "embedding_model": None,  # Will use llm's default embedding model
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "top_k": 5,
    "search_mode": "hybrid",  # vector | keyword | hybrid
    "rrf_k": 60,
    "vector_weight": 0.7,
    "keyword_weight": 0.3,
    "diversity_lambda": 1.0,  # MMR: 1.0=relevance-only (disabled), 0.5=balanced, 0.0=diversity-only
    "document_loaders": {
        # Git loader: yek with jq transform to aichat format (matches aichat exactly)
        "git": """sh -c "yek $1 --json | jq '[.[] | { path: .filename, contents: .content }]'" """,
        "pdf": "pdftotext $1 -",
        "docx": "pandoc --to plain $1",
        "odt": "pandoc --to plain $1",
        "rtf": "pandoc --to plain $1",
        "epub": "pandoc --to plain $1",
        "rst": "pandoc --to plain $1",
        "org": "pandoc --to plain $1",
    },
}


class RAGConfig:
    """Global configuration manager for all RAG collections."""

    _instance = None
    _config: Dict[str, Any] = {}
    _config_file: Optional[Path] = None

    def __new__(cls):
        """Singleton pattern - only one global config instance."""
        if cls._instance is None:
            cls._instance = super(RAGConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize global configuration."""
        if self._config_file is None:
            config_dir = get_rag_config_dir()
            self._config_file = config_dir / "config.yaml"
            self.load()

    def load(self) -> Dict[str, Any]:
        """
        Load global configuration from file.

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If configuration contains invalid values
        """
        if self._config_file.exists():
            try:
                with open(self._config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f) or {}
                # Deep merge with defaults to ensure all keys exist and preserve nested defaults
                self._config = deep_merge(DEFAULT_CONFIG, loaded_config)
            except Exception as e:
                print(f"Warning: Failed to load global config from {self._config_file}: {e}")
                self._config = deepcopy(DEFAULT_CONFIG)
        else:
            self._config = deepcopy(DEFAULT_CONFIG)

        # Validate configuration values
        self._validate_config()

        return self._config

    def _validate_config(self):
        """Validate configuration values for consistency."""
        # Validate chunk_size and chunk_overlap
        chunk_size = self._config.get("chunk_size", DEFAULT_CONFIG["chunk_size"])
        chunk_overlap = self._config.get("chunk_overlap", DEFAULT_CONFIG["chunk_overlap"])

        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got: {chunk_size}")

        if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be a non-negative integer, got: {chunk_overlap}")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size}). "
                f"This would cause an infinite loop during text splitting."
            )

        # Validate top_k
        top_k = self._config.get("top_k", DEFAULT_CONFIG["top_k"])
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {top_k}")

        # Validate search_mode
        search_mode = self._config.get("search_mode", DEFAULT_CONFIG["search_mode"])
        valid_modes = ["vector", "keyword", "hybrid"]
        if search_mode not in valid_modes:
            raise ValueError(f"search_mode must be one of {valid_modes}, got: {search_mode}")

        # Validate rrf_k (RRF constant) - reasonable bounds to prevent numerical precision issues
        rrf_k = self._config.get("rrf_k", DEFAULT_CONFIG["rrf_k"])
        if not isinstance(rrf_k, int) or not (1 <= rrf_k <= 100):
            raise ValueError(
                f"rrf_k must be an integer between 1 and 100, got: {rrf_k}. "
                f"Higher values reduce score differentiation between ranks."
            )

        # Validate vector_weight and keyword_weight are non-negative
        vector_weight = self._config.get("vector_weight", DEFAULT_CONFIG["vector_weight"])
        keyword_weight = self._config.get("keyword_weight", DEFAULT_CONFIG["keyword_weight"])
        if not isinstance(vector_weight, (int, float)) or vector_weight < 0:
            raise ValueError(f"vector_weight must be a non-negative number, got: {vector_weight}")
        if not isinstance(keyword_weight, (int, float)) or keyword_weight < 0:
            raise ValueError(f"keyword_weight must be a non-negative number, got: {keyword_weight}")

        # Ensure at least one search method is active
        if vector_weight == 0 and keyword_weight == 0:
            raise ValueError(
                "At least one of vector_weight or keyword_weight must be greater than 0"
            )

        # Validate diversity_lambda (MMR parameter)
        diversity_lambda = self._config.get("diversity_lambda", DEFAULT_CONFIG["diversity_lambda"])
        if not isinstance(diversity_lambda, (int, float)) or not (0 <= diversity_lambda <= 1):
            raise ValueError(
                f"diversity_lambda must be a number between 0 and 1, got: {diversity_lambda}. "
                f"Use 1.0 to disable diversity (relevance-only), 0.5 for balanced, 0.0 for max diversity."
            )

    def save(self):
        """Save global configuration to file."""
        # Create directory if it doesn't exist
        self._config_file.parent.mkdir(parents=True, exist_ok=True)

        # Write config
        with open(self._config_file, 'w') as f:
            yaml.safe_dump(self._config, f, default_flow_style=False, sort_keys=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value."""
        self._config[key] = value

    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values."""
        self._config.update(updates)

    def get_chunk_size(self) -> int:
        """Get chunk size for text splitting."""
        return self._config.get("chunk_size", DEFAULT_CONFIG["chunk_size"])

    def get_chunk_overlap(self) -> int:
        """Get chunk overlap for text splitting."""
        return self._config.get("chunk_overlap", DEFAULT_CONFIG["chunk_overlap"])

    def get_top_k(self) -> int:
        """Get number of results to return from search."""
        return self._config.get("top_k", DEFAULT_CONFIG["top_k"])

    def get_search_mode(self) -> str:
        """Get search mode: vector, keyword, or hybrid."""
        return self._config.get("search_mode", DEFAULT_CONFIG["search_mode"])

    def get_embedding_model(self) -> Optional[str]:
        """Get embedding model identifier."""
        return self._config.get("embedding_model")

    def get_loaders(self) -> Dict[str, str]:
        """Get document loaders configuration."""
        return self._config.get("document_loaders", DEFAULT_CONFIG["document_loaders"].copy())

    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return deepcopy(self._config)

    @property
    def exists(self) -> bool:
        """Check if configuration file exists."""
        return self._config_file.exists()


def get_rag_config_dir() -> Path:
    """
    Get the RAG configuration directory.

    Uses llm's config directory structure:
    ~/.config/io.datasette.llm/rag/

    Returns:
        Path to RAG config directory
    """
    import llm
    base_dir = llm.user_dir()
    rag_dir = base_dir / "rag"
    rag_dir.mkdir(parents=True, exist_ok=True)
    return rag_dir


def list_collections() -> List[str]:
    """
    List all RAG collections.

    Returns:
        List of collection names
    """
    config_dir = get_rag_config_dir()
    if not config_dir.exists():
        return []

    collections = []
    for item in config_dir.iterdir():
        if item.is_dir() and (item / "chromadb").exists():
            collections.append(item.name)

    return sorted(collections)


def validate_collection_name(collection_name: str):
    """
    Validate collection name to prevent path traversal attacks.

    Args:
        collection_name: Name to validate

    Raises:
        ValueError: If collection name is invalid
    """
    if not collection_name:
        raise ValueError("Collection name cannot be empty")

    # Prevent path traversal
    if ".." in collection_name or "/" in collection_name or "\\" in collection_name:
        raise ValueError(
            f"Invalid collection name: '{collection_name}'. "
            "Collection names cannot contain path separators or '..' sequences."
        )

    # Prevent hidden directories
    if collection_name.startswith("."):
        raise ValueError(
            f"Invalid collection name: '{collection_name}'. "
            "Collection names cannot start with '.'"
        )

    # Ensure it's a reasonable name (alphanumeric, hyphens, underscores)
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', collection_name):
        raise ValueError(
            f"Invalid collection name: '{collection_name}'. "
            "Collection names must contain only alphanumeric characters, hyphens, and underscores."
        )


def delete_collection(collection_name: str):
    """
    Delete a RAG collection's data directory.

    Args:
        collection_name: Name of collection to delete

    Raises:
        ValueError: If collection name is invalid
    """
    # Validate collection name to prevent path traversal
    validate_collection_name(collection_name)

    config_dir = get_rag_config_dir()
    collection_dir = config_dir / collection_name

    # Additional safety check: ensure resolved path is within config_dir
    try:
        collection_dir_resolved = collection_dir.resolve()
        config_dir_resolved = config_dir.resolve()
        # Use is_relative_to() for proper path containment check
        # This correctly handles cases like "/config-evil" not being inside "/config"
        if not collection_dir_resolved.is_relative_to(config_dir_resolved):
            raise ValueError(
                f"Security error: Collection directory '{collection_dir_resolved}' "
                f"is outside config directory '{config_dir_resolved}'"
            )
    except ValueError:
        # Re-raise ValueError (our security error) as-is
        raise
    except Exception as e:
        raise ValueError(f"Failed to validate collection path: {e}")

    if collection_dir.exists():
        import shutil
        shutil.rmtree(collection_dir)
