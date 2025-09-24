"""
Lightweight Progress Tracking for XLFusion V1.1

Simple, efficient progress bars that minimize performance impact during
large SDXL model merging operations.
"""
import sys
import time
from typing import Optional


class SimpleProgress:
    """
    Lightweight progress tracker optimized for minimal overhead.
    Updates display at most every 100ms to avoid performance impact.
    """

    def __init__(self, total: int, description: str = "Progress", width: int = 30):
        self.total = max(total, 1)  # Avoid division by zero
        self.current = 0
        self.description = description[:40]  # Limit description length
        self.width = width
        self.start_time = time.time()
        self.last_update = 0
        self.update_interval = 0.1  # 100ms minimum between updates

    def update(self, increment: int = 1) -> None:
        """Update progress by increment amount"""
        self.current = min(self.current + increment, self.total)

        current_time = time.time()
        # Only update display if enough time has passed or we're complete
        if (current_time - self.last_update >= self.update_interval or
            self.current == self.total):
            self._render()
            self.last_update = current_time

    def _render(self) -> None:
        """Render progress bar with minimal string operations"""
        try:
            percent = self.current / self.total
            filled = int(self.width * percent)
            bar = "█" * filled + "░" * (self.width - filled)

            # Simple time calculation
            elapsed = time.time() - self.start_time
            if self.current > 0 and self.current < self.total:
                rate = self.current / elapsed
                eta = (self.total - self.current) / rate
                eta_str = f"{int(eta//60):02d}:{int(eta%60):02d}"
            else:
                eta_str = "00:00"

            # Single string format operation
            line = f"\r{self.description} [{bar}] {self.current}/{self.total} ({percent*100:.0f}%) ETA:{eta_str}"

            # Ensure line doesn't exceed reasonable length
            if len(line) > 100:
                line = line[:97] + "..."

            sys.stdout.write(line)
            sys.stdout.flush()

            if self.current == self.total:
                sys.stdout.write("\n")
                sys.stdout.flush()

        except (OSError, IOError):
            # Silently handle stdout errors (e.g., broken pipe)
            pass

    def finish(self) -> None:
        """Mark as complete"""
        self.current = self.total
        self._render()


def track_merge_progress(models: list, mode: str) -> Optional['SimpleProgress']:
    """
    Create a simple progress tracker for merge operations.
    Returns None if progress tracking is disabled or not possible.
    """
    try:
        if not sys.stdout.isatty():
            # Don't show progress bars in non-interactive environments
            return None

        print(f"\n{mode.upper()} MERGE: {len(models)} models")
        return SimpleProgress(len(models), f"{mode} merge")

    except (AttributeError, OSError):
        # Gracefully handle environments where stdout isn't available
        return None


def track_tensor_progress(total: int, description: str) -> Optional['SimpleProgress']:
    """
    Create progress tracker for tensor operations.
    Returns None if not in interactive environment.
    """
    try:
        if not sys.stdout.isatty() or total < 100:
            # Skip progress for small operations or non-interactive
            return None
        return SimpleProgress(total, description)
    except (AttributeError, OSError):
        return None


def show_phase_start(phase_name: str) -> float:
    """Start a phase and return start time"""
    try:
        print(f"\n{phase_name}...")
        return time.time()
    except (OSError, IOError):
        return time.time()


def show_phase_complete(phase_name: str, start_time: float) -> None:
    """Show phase completion with timing"""
    try:
        duration = time.time() - start_time
        if duration < 1:
            print(f"  {phase_name} completed in {duration*1000:.0f}ms")
        elif duration < 60:
            print(f"  {phase_name} completed in {duration:.1f}s")
        else:
            mins = int(duration // 60)
            secs = duration % 60
            print(f"  {phase_name} completed in {mins}m {secs:.1f}s")
    except (OSError, IOError):
        pass


def show_merge_complete(start_time: float, mode: str) -> None:
    """Show final merge completion summary"""
    try:
        total_time = time.time() - start_time
        if total_time < 60:
            print(f"\n{mode} merge completed in {total_time:.1f}s")
        else:
            mins = int(total_time // 60)
            secs = total_time % 60
            print(f"\n{mode} merge completed in {mins}m {secs:.1f}s")
    except (OSError, IOError):
        pass