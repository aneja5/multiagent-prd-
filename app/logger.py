"""Logging setup for the multi-agent PRD generator.

This module configures logging to both console (with rich formatting) and file.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

from app.config import get_config


class Logger:
    """Centralized logging manager with console and file output."""

    _instance: Optional['Logger'] = None
    _initialized: bool = False

    def __new__(cls) -> 'Logger':
        """Ensure singleton pattern for logger."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the logger (only once)."""
        if Logger._initialized:
            return

        self.config = get_config()
        self.console = Console()

        # Set up root logger
        self._setup_logging()

        Logger._initialized = True

    def _setup_logging(self) -> None:
        """Configure logging handlers and formatters."""
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config.log_level)

        # Remove existing handlers to avoid duplicates
        root_logger.handlers.clear()

        # Console handler with Rich formatting
        console_handler = RichHandler(
            console=self.console,
            rich_tracebacks=True,
            show_time=True,
            show_path=True,
            markup=True,
            log_time_format="[%X]"
        )
        console_handler.setLevel(self.config.log_level)
        root_logger.addHandler(console_handler)

        # File handler
        log_file = self.config.log_dir / "app.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file

        # File formatter (more detailed than console)
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance for a specific module.

        Args:
            name: The logger name (typically __name__)

        Returns:
            A configured logger instance
        """
        return logging.getLogger(name)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.

    This is a convenience function that creates the Logger singleton
    and returns a module-specific logger.

    Args:
        name: The logger name (typically __name__)

    Returns:
        A configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting process")
    """
    logger_manager = Logger()
    return logger_manager.get_logger(name)


def log_section(title: str, console: Optional[Console] = None) -> None:
    """Print a formatted section header.

    Args:
        title: The section title
        console: Optional console instance (creates new if not provided)
    """
    if console is None:
        console = Console()

    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold cyan]{title.center(60)}[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")


def log_success(message: str, console: Optional[Console] = None) -> None:
    """Print a success message.

    Args:
        message: The success message
        console: Optional console instance
    """
    if console is None:
        console = Console()

    console.print(f"[bold green]✓[/bold green] {message}")


def log_error(message: str, console: Optional[Console] = None) -> None:
    """Print an error message.

    Args:
        message: The error message
        console: Optional console instance
    """
    if console is None:
        console = Console()

    console.print(f"[bold red]✗[/bold red] {message}")


def log_warning(message: str, console: Optional[Console] = None) -> None:
    """Print a warning message.

    Args:
        message: The warning message
        console: Optional console instance
    """
    if console is None:
        console = Console()

    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def log_info(message: str, console: Optional[Console] = None) -> None:
    """Print an info message.

    Args:
        message: The info message
        console: Optional console instance
    """
    if console is None:
        console = Console()

    console.print(f"[bold blue]ℹ[/bold blue] {message}")
