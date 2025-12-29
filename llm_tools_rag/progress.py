"""Progress display utilities using Rich."""

from contextlib import contextmanager
from rich.console import Console
from rich.status import Status

# Use stderr to avoid interfering with stdout pipe
_console = Console(stderr=True)


@contextmanager
def progress_status(message: str):
    """
    Context manager for inline status updates.

    Usage:
        with progress_status("Processing...") as status:
            for i in range(100):
                status.update(f"[cyan]Processing: {i}/100[/]")
    """
    with Status(f"[cyan]{message}[/]", console=_console, spinner="dots", spinner_style="cyan") as status:
        yield status


def print_warning(message: str):
    """Print a warning message to stderr (yellow)."""
    _console.print(f"[yellow]Warning:[/] {message}", highlight=False)


def print_info(message: str):
    """Print an info message to stderr (dim)."""
    _console.print(f"[dim]{message}[/]", highlight=False)
