"""Shared execution options and progress helpers for merge runtimes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None
    TQDM_AVAILABLE = False

from .blocks import BlockMapping, get_block_mapping


EXECUTION_MODES = {"standard", "low-memory"}
PROGRESS_MODES = {"auto", "tqdm", "simple", "quiet"}
_GROUP_ORDER = {"down": 0, "mid": 1, "up": 2, "other": 3}


@dataclass(frozen=True)
class MergeExecutionOptions:
    """Runtime behaviour for merge execution."""

    mode: str = "low-memory"
    progress: str = "auto"
    sort_keys: bool = False
    log_every: int = 250


def normalize_execution_options(raw: Optional[Any] = None) -> MergeExecutionOptions:
    """Normalize user or batch execution options to a stable structure."""
    if raw is None:
        options = MergeExecutionOptions()
    elif isinstance(raw, MergeExecutionOptions):
        options = raw
    elif isinstance(raw, dict):
        mode = str(raw.get("mode", MergeExecutionOptions.mode)).strip() or MergeExecutionOptions.mode
        progress = str(raw.get("progress", MergeExecutionOptions.progress)).strip() or MergeExecutionOptions.progress
        sort_keys = bool(raw.get("sort_keys", False))
        log_every_raw = raw.get("log_every", MergeExecutionOptions.log_every)
        try:
            log_every = max(1, int(log_every_raw))
        except (TypeError, ValueError):
            log_every = MergeExecutionOptions.log_every
        options = MergeExecutionOptions(
            mode=mode,
            progress=progress,
            sort_keys=sort_keys,
            log_every=log_every,
        )
    else:
        raise TypeError("execution options must be a mapping or MergeExecutionOptions")

    mode = options.mode if options.mode in EXECUTION_MODES else MergeExecutionOptions.mode
    progress = options.progress if options.progress in PROGRESS_MODES else MergeExecutionOptions.progress
    sort_keys = options.sort_keys or mode == "low-memory"
    return MergeExecutionOptions(
        mode=mode,
        progress=progress,
        sort_keys=sort_keys,
        log_every=max(1, int(options.log_every)),
    )


def execution_options_to_dict(raw: Optional[Any]) -> Dict[str, Any]:
    """Serialize normalized execution options for metadata and presets."""
    return asdict(normalize_execution_options(raw))


def build_processing_order(
    base_keys: Sequence[str],
    extra_key_sets: Iterable[Iterable[str]],
    *,
    sort_keys: bool,
    block_mapping: object = "sdxl",
) -> List[str]:
    """Build the ordered list of unique tensor keys that will be processed."""
    seen = set()
    ordered: List[str] = []

    for key in base_keys:
        if key not in seen:
            ordered.append(key)
            seen.add(key)

    for keys in extra_key_sets:
        for key in keys:
            if key not in seen:
                ordered.append(key)
                seen.add(key)

    if not sort_keys:
        return ordered

    mapping = _resolve_block_mapping(block_mapping)
    return sorted(ordered, key=lambda key: _sort_key(key, mapping))

def _resolve_block_mapping(value: object) -> BlockMapping:
    if isinstance(value, BlockMapping):
        return value
    if value is None:
        return get_block_mapping("sdxl")
    if isinstance(value, str):
        return get_block_mapping(value)
    raise TypeError("block_mapping must be a BlockMapping, a string name, or None")


def _sort_key(key: str, mapping: BlockMapping) -> tuple[int, str]:
    block_group = mapping.get_block_assignment(key)
    coarse_group = block_group.split("_", 1)[0] if block_group else (mapping.group_for_key(key) or "other")
    return (_GROUP_ORDER.get(coarse_group, _GROUP_ORDER["other"]), key)


class ProgressReporter:
    """Single progress abstraction for TTY, logs and callbacks."""

    def __init__(
        self,
        total: int,
        desc: str,
        options: MergeExecutionOptions,
        *,
        progress_cb: Optional[Any] = None,
    ) -> None:
        self.total = max(0, int(total))
        self.desc = desc
        self.options = normalize_execution_options(options)
        self.progress_cb = progress_cb
        self.current = 0
        self._tqdm_bar = None

    def __enter__(self) -> "ProgressReporter":
        if self.progress_cb:
            try:
                self.progress_cb("total", self.total)
            except Exception:
                pass

        mode = self._resolved_progress_mode()
        if mode == "tqdm" and TQDM_AVAILABLE:
            self._tqdm_bar = tqdm(total=self.total, desc=self.desc, unit="tensor")
        elif mode == "simple":
            print(f"{self.desc}: 0/{self.total}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._tqdm_bar is not None:
            self._tqdm_bar.close()
            self._tqdm_bar = None
        elif exc is None and self._resolved_progress_mode() == "simple" and self.total:
            if self.current < self.total:
                print(f"{self.desc}: {self.current}/{self.total}")

    def step(self, increment: int = 1) -> None:
        increment = max(0, int(increment))
        if increment == 0:
            return
        self.current += increment
        if self.progress_cb:
            try:
                self.progress_cb("tick", increment)
            except Exception:
                pass

        mode = self._resolved_progress_mode()
        if self._tqdm_bar is not None:
            self._tqdm_bar.update(increment)
            return
        if mode != "simple":
            return
        if (
            self.current == self.total
            or self.current % self.options.log_every == 0
            or self.current == 1
        ):
            if self.total > 0:
                percent = (self.current / self.total) * 100.0
                print(f"{self.desc}: {self.current}/{self.total} ({percent:.1f}%)")
            else:
                print(f"{self.desc}: {self.current}")

    def _resolved_progress_mode(self) -> str:
        if self.options.progress != "auto":
            return self.options.progress
        if TQDM_AVAILABLE and sys.stderr.isatty():
            return "tqdm"
        return "simple"
