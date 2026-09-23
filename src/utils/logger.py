import sys
from dataclasses import dataclass, field


@dataclass
class Logger:
    tag: str
    _prefix: str = field(init=False)

    def __post_init__(self) -> None:
        self._prefix = f"[{self.tag}]"

    def info(self, message: str) -> None:
        """Print an informational message to stdout."""
        print(f"{self._prefix} {message}")

    def error(self, message: str) -> None:
        """Print an error message to stderr."""
        print(f"{self._prefix} {message}", file=sys.stderr)

    def blank(self) -> None:
        """Print a blank line to stdout (for visual spacing)."""
        print()
