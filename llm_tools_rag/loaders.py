"""
Protocol-based document loaders matching aichat's architecture.
Supports: git: (via yek), pdf (via pdftotext), docx (via pandoc), and plain text.
Also supports recursive URL crawling with ** syntax.
"""

import subprocess
import shutil
import shlex
import re
import time
import os
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from urllib.parse import urlparse, urljoin, urldefrag
from dataclasses import dataclass


# Default document loaders (matches aichat configuration)
# $1 = input file, $2 = output file (optional, for loaders that can't use stdout)
# If loader outputs JSON array of {path, contents}, creates multiple documents
DEFAULT_LOADERS: Dict[str, str] = {
    # Git loader: defined for compatibility but actually uses built-in Python implementation
    # (supports remote URLs via clone + yek, local paths via direct yek)
    "git": """sh -c "yek $1 --json | jq '[.[] | { path: .filename, contents: .content }]'" """,
    "pdf": "pdftotext $1 -",
    "docx": "pandoc --to plain $1",
    # Additional formats supported by pandoc
    "odt": "pandoc --to plain $1",    # OpenDocument Text
    "rtf": "pandoc --to plain $1",    # Rich Text Format
    "epub": "pandoc --to plain $1",   # E-book format
    "rst": "pandoc --to plain $1",    # reStructuredText
    "org": "pandoc --to plain $1",    # Org-mode
}

# Browser-like User-Agent for web requests (avoids blocks from python-requests default)
DEFAULT_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'


@dataclass
class LoadedDocument:
    """Represents a loaded document with path and content."""
    path: str
    content: str


class DocumentLoader:
    """Protocol-based document loader."""

    def __init__(self, loaders: Optional[Dict[str, str]] = None):
        """
        Initialize document loader with custom loaders.

        Args:
            loaders: Dictionary mapping protocol/extension to shell command
                    Use $1 as placeholder for the path argument
        """
        self.loaders = loaders if loaders is not None else DEFAULT_LOADERS.copy()

    def is_protocol_path(self, path: str) -> bool:
        """Check if path uses a protocol loader (e.g., git:/path)."""
        if ':' not in path or path.startswith('/'):
            return False

        protocol = path.split(':', 1)[0]
        return protocol in self.loaders

    def is_recursive_url(self, path: str) -> bool:
        """Check if path is a recursive URL pattern (contains **)."""
        return '**' in path and path.startswith('http')

    def is_single_url(self, path: str) -> bool:
        """Check if path is a single URL (http/https without **)."""
        return (path.startswith('http://') or path.startswith('https://')) and '**' not in path

    def load(self, path: str) -> str:
        """
        Load document content from path.

        Args:
            path: File path or protocol path (e.g., git:https://github.com/user/repo)
                 Supports recursive URL crawling with ** pattern (e.g., https://example.com/**)

        Returns:
            Document content as string (single document)

        Raises:
            ValueError: If loader is not available or command fails
        """
        # Check for recursive URL pattern
        if self.is_recursive_url(path):
            # For backward compatibility, load_multi handles this
            # But load() should return first document's content
            docs = self.load_multi(path)
            if docs:
                return docs[0].content
            return ""

        # Check for single URL (fetch and process based on content-type)
        if self.is_single_url(path):
            return self._load_url(path)

        # Check for protocol-based path
        if self.is_protocol_path(path):
            protocol, actual_path = path.split(':', 1)
            return self._load_via_protocol(protocol, actual_path)

        # Regular file path - detect by extension
        file_path = Path(path)

        if not file_path.exists():
            raise ValueError(f"File not found: {path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a regular file: {path} (is directory, device, or special file)")

        # Get extension and check for loader
        ext = file_path.suffix.lstrip('.')

        if ext in self.loaders:
            return self._load_via_protocol(ext, str(file_path))

        # Default: read as plain text
        return self._load_text(file_path)

    def load_multi(self, path: str) -> List[LoadedDocument]:
        """
        Load documents from path, supporting multiple documents from recursive crawling.

        Args:
            path: File path, protocol path, or recursive URL pattern

        Returns:
            List of LoadedDocument objects

        Raises:
            ValueError: If loader is not available or command fails
        """
        # Check for recursive URL pattern
        if self.is_recursive_url(path):
            return self._crawl_recursive_url(path)

        # Check for protocol-based path (may return multiple documents for JSON output)
        if self.is_protocol_path(path):
            protocol, actual_path = path.split(':', 1)
            return self._load_via_protocol_multi(protocol, actual_path, path)

        # Single document load
        content = self.load(path)
        return [LoadedDocument(path=path, content=content)]

    def _load_via_protocol_multi(self, protocol: str, path: str, full_path: str) -> List[LoadedDocument]:
        """
        Load documents via protocol, supporting JSON array output for multiple documents.

        Supports two JSON formats:
        - aichat format: {path, contents} (from external loaders with jq transform)
        - yek native format: {filename, content} (from direct yek calls)

        If loader outputs JSON array in either format, creates a separate
        LoadedDocument for each item. Otherwise returns single document.
        """
        import json

        raw_output = self._load_via_protocol_raw(protocol, path)

        # Try to parse as JSON array of documents
        try:
            data = json.loads(raw_output)
            if isinstance(data, list) and len(data) > 0:
                documents = []
                for item in data:
                    if isinstance(item, dict):
                        # Support both formats:
                        # 1. aichat format: {path, contents}
                        # 2. yek native format: {filename, content}
                        doc_path = item.get('path') or item.get('filename')
                        doc_content = item.get('contents') or item.get('content')
                        if doc_path and doc_content:
                            # Prefix path with protocol if not already prefixed
                            if not doc_path.startswith(full_path):
                                doc_path = f"{full_path}/{doc_path}"
                            documents.append(LoadedDocument(path=doc_path, content=doc_content))
                if documents:
                    return documents
        except (json.JSONDecodeError, TypeError):
            pass  # Not JSON, treat as plain text

        # Fallback: single document with raw output
        return [LoadedDocument(path=full_path, content=raw_output)]

    def _load_via_protocol(self, protocol: str, path: str) -> str:
        """Load via protocol, returning concatenated content (for load() method)."""
        docs = self._load_via_protocol_multi(protocol, path, f"{protocol}:{path}")
        if len(docs) == 1:
            return docs[0].content
        # Multiple documents: concatenate with file headers
        parts = []
        for doc in docs:
            parts.append(f"=== {doc.path} ===\n{doc.content}")
        return "\n\n".join(parts)

    def _is_remote_git_url(self, path: str) -> bool:
        """Check if path is a remote git URL."""
        return (path.startswith('http://') or
                path.startswith('https://') or
                path.startswith('git@'))

    def _load_git_repo(self, path: str) -> str:
        """
        Load git repository content using yek.
        For remote URLs, clones to temp dir first.
        Returns JSON array of {filename, content} objects.
        """
        import tempfile
        import json

        if self._is_remote_git_url(path):
            # Clone to temp directory, run yek, clean up
            with tempfile.TemporaryDirectory(prefix='rag-git-') as temp_dir:
                clone_path = os.path.join(temp_dir, 'repo')

                # Clone with depth=1 for speed
                clone_cmd = ['git', 'clone', '--depth', '1', '--single-branch', path, clone_path]
                try:
                    subprocess.run(
                        clone_cmd,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=300  # 5 min timeout for large repos
                    )
                except subprocess.CalledProcessError as e:
                    raise ValueError(f"Failed to clone git repository: {e.stderr}")
                except subprocess.TimeoutExpired:
                    raise ValueError(f"Git clone timed out after 5 minutes: {path}")

                # Run yek on the cloned repo
                return self._run_yek_json(clone_path)
        else:
            # Local path - run yek directly
            return self._run_yek_json(path)

    def _run_yek_json(self, local_path: str) -> str:
        """Run yek with --json on a local path."""
        if not shutil.which('yek'):
            raise ValueError(
                "yek command not found. Install with: cargo install yek"
            )

        try:
            result = subprocess.run(
                ['yek', local_path, '--json'],
                capture_output=True,
                text=True,
                check=True,
                timeout=300
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise ValueError(f"yek failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise ValueError(f"yek timed out after 5 minutes")

    def _load_via_protocol_raw(self, protocol: str, path: str) -> str:
        """Run loader command and return raw output."""
        import tempfile

        if protocol not in self.loaders:
            raise ValueError(f"No loader configured for protocol: {protocol}")

        # Special handling for git protocol - use yek with clone support
        if protocol == 'git':
            return self._load_git_repo(path)

        # Get command template and build safe argument list
        cmd_template = self.loaders[protocol]

        # Parse template safely with shlex, then substitute $1 and $2
        try:
            cmd_parts = shlex.split(cmd_template)
        except ValueError as e:
            raise ValueError(f"Invalid command template for {protocol}: {e}")

        # Sanitize path to prevent flag injection
        # If path starts with '-' or '--', prefix with './' to treat it as a file
        sanitized_path = path
        if path.startswith('-'):
            # Absolute paths cannot start with '-' in Unix (they start with '/')
            # If we somehow encounter this, it's either impossible or malicious
            if path.startswith('/'):
                raise ValueError(
                    f"Invalid path: absolute paths cannot start with '-': {path}"
                )
            # For relative paths, prefix with ./ to prevent flag injection
            sanitized_path = './' + path

        # Check if $2 (output file) is used - some loaders can't write to stdout
        use_stdout = True
        output_file = None
        if any('$2' in part for part in cmd_parts):
            use_stdout = False
            # Create temp file for output
            fd, output_file = tempfile.mkstemp(prefix='rag-loader-', suffix='.txt')
            os.close(fd)

        # Replace $1 and $2 placeholders
        def replace_placeholders(part: str) -> str:
            result = part
            if '$1' in result:
                result = result.replace('$1', sanitized_path)
            if '$2' in result and output_file:
                result = result.replace('$2', output_file)
            return result

        cmd_parts = [replace_placeholders(part) for part in cmd_parts]

        # Check if command is available
        cmd_binary = cmd_parts[0]
        if not shutil.which(cmd_binary):
            raise ValueError(
                f"Loader command '{cmd_binary}' not found. "
                f"Please install it to use {protocol}: protocol."
            )

        try:
            if use_stdout:
                result = subprocess.run(
                    cmd_parts,  # Use list, not string - prevents shell injection
                    shell=False,  # SAFE: no shell interpretation
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=300  # 5 minute timeout
                )
                return result.stdout
            else:
                # Output goes to file, not stdout
                result = subprocess.run(
                    cmd_parts,
                    shell=False,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=300
                )
                # Read output from file
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        return f.read()
                except UnicodeDecodeError:
                    with open(output_file, 'r', encoding='latin-1') as f:
                        return f.read()
        except subprocess.TimeoutExpired:
            raise ValueError(f"Loader command timed out after 5 minutes: {' '.join(cmd_parts)}")
        except subprocess.CalledProcessError as e:
            raise ValueError(
                f"Loader command failed (exit code {e.returncode}): {' '.join(cmd_parts)}\n"
                f"stderr: {e.stderr}"
            )
        finally:
            # Clean up temp file
            if output_file and os.path.exists(output_file):
                try:
                    os.unlink(output_file)
                except OSError:
                    pass

    def _load_text(self, file_path: Path) -> str:
        """Load plain text file."""
        # First, check if file appears to be binary
        # Read first 8192 bytes to detect null bytes (binary indicator)
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(8192)
                if b'\x00' in sample:
                    raise ValueError(
                        f"File appears to be binary (contains null bytes): {file_path}. "
                        f"Use a protocol loader (pdf:, docx:, etc.) for binary formats."
                    )
        except OSError as e:
            raise ValueError(f"Failed to read file: {file_path}") from e

        # Now try to read as text
        try:
            return file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Try with latin-1 as fallback for extended ASCII
            try:
                return file_path.read_text(encoding='latin-1')
            except Exception as e:
                raise ValueError(f"Failed to read file as text: {file_path}") from e

    def _load_url(self, url: str) -> str:
        """
        Load document from URL, detecting content-type and using appropriate loader.

        Similar to aichat's fetch_with_loaders - handles PDF, DOCX, HTML, etc.
        """
        import tempfile
        import requests
        from bs4 import BeautifulSoup

        # Content-type to extension mapping (matches aichat)
        CONTENT_TYPE_MAP = {
            'application/pdf': 'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
            'application/vnd.oasis.opendocument.text': 'odt',
            'application/vnd.oasis.opendocument.spreadsheet': 'ods',
            'application/vnd.oasis.opendocument.presentation': 'odp',
            'application/rtf': 'rtf',
            'application/epub+zip': 'epub',
        }

        try:
            response = requests.get(url, timeout=30, headers={'User-Agent': DEFAULT_USER_AGENT})
            response.raise_for_status()
        except Exception as e:
            raise ValueError(f"Failed to fetch URL: {url}: {e}")

        # Parse content-type (strip charset and parameters)
        content_type = response.headers.get('Content-Type', '')
        if ';' in content_type:
            content_type = content_type.split(';')[0].strip()

        # HTML: extract text directly
        if 'text/html' in content_type:
            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()
            return soup.get_text(separator='\n', strip=True)

        # Plain text: return directly
        if content_type.startswith('text/'):
            return response.text

        # Check if we have a loader for this content-type
        extension = CONTENT_TYPE_MAP.get(content_type)

        # Fallback: try to detect from URL extension
        if not extension:
            parsed_url = urlparse(url)
            url_path = parsed_url.path
            if '.' in url_path:
                extension = url_path.rsplit('.', 1)[-1].lower()

        # If we have a loader for this extension, download and process
        if extension and extension in self.loaders:
            fd, temp_path = tempfile.mkstemp(prefix='rag-download-', suffix=f'.{extension}')
            try:
                os.write(fd, response.content)
                os.close(fd)
                return self._load_via_protocol(extension, temp_path)
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass

        # Unknown content type - try to decode as text
        try:
            return response.text
        except Exception:
            raise ValueError(
                f"Cannot process URL with content-type '{content_type}': {url}. "
                f"No loader configured for this type."
            )

    def add_loader(self, protocol: str, command: str):
        """
        Add or override a loader.

        Args:
            protocol: Protocol name or file extension
            command: Shell command with $1 placeholder for path
        """
        self.loaders[protocol] = command

    def remove_loader(self, protocol: str):
        """Remove a loader."""
        self.loaders.pop(protocol, None)

    def _crawl_recursive_url(
        self,
        url_pattern: str,
        max_depth: int = 3,
        max_pages: int = 50,
        max_memory_bytes: int = 100_000_000  # 100MB default limit
    ) -> List[LoadedDocument]:
        """
        Crawl a website recursively following links with memory protection.

        Args:
            url_pattern: URL with ** pattern (e.g., https://example.com/**)
            max_depth: Maximum depth to crawl
            max_pages: Maximum number of pages to fetch
            max_memory_bytes: Maximum total content size in bytes (default 100MB)

        Returns:
            List of LoadedDocument objects

        Raises:
            ValueError: If memory limit is exceeded during crawling
        """
        import requests
        from bs4 import BeautifulSoup

        # Extract base URL (remove ** pattern)
        base_url = url_pattern.replace('/**', '/').replace('**', '')

        # Normalize start URL (similar to aichat's normalize_start_url)
        parsed_base = urlparse(base_url)
        # Strip query string and fragment
        normalized_path = parsed_base.path
        # Ensure path ends at directory (at last /)
        if '/' in normalized_path:
            last_slash = normalized_path.rfind('/')
            normalized_path = normalized_path[:last_slash + 1]
        if not normalized_path.endswith('/'):
            normalized_path += '/'

        base_url = f"{parsed_base.scheme}://{parsed_base.netloc}{normalized_path}"
        # Store base path prefix for restricting crawl scope
        base_path_prefix = normalized_path

        visited = set()
        to_visit = [(base_url, 0)]  # (url, depth)
        documents = []
        total_content_bytes = 0  # Track total memory usage

        print(f"Crawling {base_url} recursively (max depth: {max_depth}, max pages: {max_pages})...")

        while to_visit and len(visited) < max_pages:
            url, depth = to_visit.pop(0)

            if url in visited or depth > max_depth:
                continue

            visited.add(url)

            try:
                # Fetch page with timeout
                response = requests.get(url, timeout=10, headers={'User-Agent': DEFAULT_USER_AGENT})
                response.raise_for_status()

                # Check content-length header before processing body
                content_length = response.headers.get('Content-Length')
                if content_length:
                    size = int(content_length)
                    if total_content_bytes + size > max_memory_bytes:
                        continue  # Skip oversized pages

                # Only process HTML content
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type:
                    continue

                # Rate limiting: wait 1 second between HTML requests (after content-type check)
                time.sleep(1)

                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract text content (remove scripts, styles, and navigation boilerplate)
                for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                    element.decompose()
                text = soup.get_text(separator='\n', strip=True)

                # Check memory limit before adding document
                content_bytes = len(text.encode('utf-8'))
                if total_content_bytes + content_bytes > max_memory_bytes:
                    raise ValueError(
                        f"Memory limit exceeded during crawling. "
                        f"Crawled {len(documents)} pages totaling "
                        f"{total_content_bytes / 1_000_000:.1f}MB before hitting "
                        f"{max_memory_bytes / 1_000_000:.0f}MB limit at '{url}'. "
                        f"Consider crawling a smaller site or increasing max_memory_bytes parameter."
                    )

                # Add document and update memory counter
                documents.append(LoadedDocument(path=url, content=text))
                total_content_bytes += content_bytes
                print(f"  Crawled: {url} ({len(text)} chars, {total_content_bytes / 1_000_000:.1f}MB total)")

                # Find links to follow
                if depth < max_depth:
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        # Convert relative URLs to absolute
                        absolute_url = urljoin(url, href)
                        # Strip fragment (#section) - same page, different scroll position
                        absolute_url, _ = urldefrag(absolute_url)

                        # Parse and normalize the URL
                        parsed_url = urlparse(absolute_url)
                        # Strip query string
                        normalized_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
                        # Normalize index pages: /path/index.html -> /path/
                        normalized_path = parsed_url.path
                        for index_suffix in ['/index.html', '/index.htm']:
                            if normalized_path.endswith(index_suffix):
                                normalized_path = normalized_path[:-len(index_suffix) + 1]  # Keep trailing /
                                normalized_url = f"{parsed_url.scheme}://{parsed_url.netloc}{normalized_path}"
                                break

                        # Only follow links that:
                        # 1. Are on the same domain
                        # 2. Start with the base path prefix (stay within crawl scope)
                        # 3. Haven't been visited
                        if (parsed_url.netloc == parsed_base.netloc and
                            normalized_path.startswith(base_path_prefix) and
                            normalized_url not in visited):
                            to_visit.append((normalized_url, depth + 1))

            except ValueError:
                # Re-raise memory limit errors
                raise
            except Exception as e:
                print(f"  Warning: Failed to crawl {url}: {e}")
                continue

        print(f"Crawled {len(documents)} pages ({total_content_bytes / 1_000_000:.1f}MB total)")
        return documents


def check_loader_dependencies() -> Dict[str, bool]:
    """
    Check which loader commands are available.

    Returns:
        Dictionary mapping loader name to availability status
    """
    # Map: command binary -> list of formats it enables
    commands = {
        "pdftotext": ["pdf"],
        "pandoc": ["docx", "odt", "rtf", "epub", "rst", "org"],
    }

    status = {}
    for cmd, loader_types in commands.items():
        available = shutil.which(cmd) is not None
        for loader_type in loader_types:
            status[loader_type] = available

    # git loader requires both yek AND jq (piped together)
    yek_available = shutil.which("yek") is not None
    jq_available = shutil.which("jq") is not None
    status["git"] = yek_available and jq_available

    return status


def get_missing_dependencies() -> Dict[str, str]:
    """
    Get missing loader dependencies with installation instructions.

    Returns:
        Dictionary mapping loader type to installation command
    """
    missing = {}

    # Check git loader (requires both yek and jq)
    yek_available = shutil.which("yek") is not None
    jq_available = shutil.which("jq") is not None
    if not yek_available or not jq_available:
        parts = []
        if not yek_available:
            parts.append("yek (install via cargo: cargo install yek)")
        if not jq_available:
            parts.append("jq (install via: sudo apt-get install jq)")
        missing["git"] = f"Missing: {', '.join(parts)}"

    # Check other loaders
    install_commands = {
        "pdf": "sudo apt-get install poppler-utils",
        "docx": "sudo apt-get install pandoc",
        "odt": "sudo apt-get install pandoc",
        "rtf": "sudo apt-get install pandoc",
        "epub": "sudo apt-get install pandoc",
        "rst": "sudo apt-get install pandoc",
        "org": "sudo apt-get install pandoc",
    }

    status = check_loader_dependencies()
    for loader_type, available in status.items():
        if not available and loader_type in install_commands and loader_type not in missing:
            missing[loader_type] = install_commands[loader_type]

    return missing
