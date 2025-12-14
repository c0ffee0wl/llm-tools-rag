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
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass


# Default document loaders (matches aichat configuration)
DEFAULT_LOADERS: Dict[str, str] = {
    "git": "yek $1",
    "pdf": "pdftotext $1 -",
    "docx": "pandoc --to plain $1",
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

        # Single document load
        content = self.load(path)
        return [LoadedDocument(path=path, content=content)]

    def _load_via_protocol(self, protocol: str, path: str) -> str:
        """Load document using protocol-specific loader command."""
        if protocol not in self.loaders:
            raise ValueError(f"No loader configured for protocol: {protocol}")

        # Get command template and build safe argument list
        cmd_template = self.loaders[protocol]

        # Parse template safely with shlex, then substitute $1
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

        # Replace $1 placeholder with sanitized path
        cmd_parts = [sanitized_path if part == '$1' else part for part in cmd_parts]

        # Check if command is available
        cmd_binary = cmd_parts[0]
        if not shutil.which(cmd_binary):
            raise ValueError(
                f"Loader command '{cmd_binary}' not found. "
                f"Please install it to use {protocol}: protocol."
            )

        try:
            result = subprocess.run(
                cmd_parts,  # Use list, not string - prevents shell injection
                shell=False,  # SAFE: no shell interpretation
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5 minute timeout
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            raise ValueError(f"Loader command timed out after 5 minutes: {' '.join(cmd_parts)}")
        except subprocess.CalledProcessError as e:
            raise ValueError(
                f"Loader command failed (exit code {e.returncode}): {' '.join(cmd_parts)}\n"
                f"stderr: {e.stderr}"
            )

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
        base_url = url_pattern.replace('/**', '').replace('**', '')
        parsed_base = urlparse(base_url)
        base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

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

                # Rate limiting: wait 1 second between requests
                time.sleep(1)

                # Only process HTML content
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type:
                    continue

                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract text content (remove scripts and styles)
                for script in soup(["script", "style"]):
                    script.decompose()
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

                        # Only follow links on the same domain (ignore protocol differences)
                        parsed_url = urlparse(absolute_url)
                        if parsed_url.netloc == parsed_base.netloc and absolute_url not in visited:
                            to_visit.append((absolute_url, depth + 1))

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
    commands = {
        "yek": "git",
        "pdftotext": "pdf",
        "pandoc": "docx",
    }

    status = {}
    for cmd, loader_type in commands.items():
        status[loader_type] = shutil.which(cmd) is not None

    return status


def get_missing_dependencies() -> Dict[str, str]:
    """
    Get missing loader dependencies with installation instructions.

    Returns:
        Dictionary mapping loader type to installation command
    """
    install_commands = {
        "git": "yek is already installed via install-llm-tools.sh",
        "pdf": "sudo apt-get install poppler-utils",
        "docx": "sudo apt-get install pandoc",
    }

    status = check_loader_dependencies()
    missing = {}

    for loader_type, available in status.items():
        if not available:
            missing[loader_type] = install_commands[loader_type]

    return missing
