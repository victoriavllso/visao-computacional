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

    def skip(self, message):
        print(f"[SKIPPING] {message}")

    def summary(self, image_type, stats):

        downloaded_count = (
            stats["aws"] + stats["off"]
        )

        self.blank()
        print(f"Download finished: {image_type}")

        print(
            f"Downloaded from AWS:       "
            f"{stats['aws']}"
        )

        print(
            f"Downloaded from OFF:       "
            f"{stats['off']}"
        )

        print(
            f"Already existed:            "
            f"{stats['exists']}"
        )

        print(
            f"No image metadata:          "
            f"{stats['no_image']}"
        )

        print(
            f"Invalid image metadata:     "
            f"{stats['invalid_image_data']}"
        )

        print(
            f"Failed:                     "
            f"{stats['failed']}"
        )

        print(
            f"New images downloaded:      "
            f"{downloaded_count}"
        )

        self.blank()
