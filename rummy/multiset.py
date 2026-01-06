from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .tiles import MULTISET_SIZE


def _validate_counts(counts: Sequence[int]) -> None:
    if len(counts) != MULTISET_SIZE:
        raise ValueError(f"multiset length must be {MULTISET_SIZE}")
    if any(c < 0 for c in counts):
        raise ValueError("multiset counts must be non-negative")


@dataclass
class TileMultiset:
    counts: List[int]

    def __post_init__(self) -> None:
        _validate_counts(self.counts)

    @classmethod
    def empty(cls) -> "TileMultiset":
        return cls([0] * MULTISET_SIZE)

    @classmethod
    def from_iterable(cls, tiles: Iterable[int]) -> "TileMultiset":
        counts = [0] * MULTISET_SIZE
        for tile in tiles:
            counts[tile] += 1
        return cls(counts)

    @classmethod
    def from_compact(cls, data: Iterable[Tuple[int, int]]) -> "TileMultiset":
        counts = [0] * MULTISET_SIZE
        for tile_id, count in data:
            counts[tile_id] = count
        return cls(counts)

    def to_compact(self) -> List[Tuple[int, int]]:
        return [(idx, c) for idx, c in enumerate(self.counts) if c]

    def add(self, other: "TileMultiset") -> "TileMultiset":
        return TileMultiset([a + b for a, b in zip(self.counts, other.counts)])

    def sub(self, other: "TileMultiset") -> "TileMultiset":
        if any(a < b for a, b in zip(self.counts, other.counts)):
            raise ValueError("cannot subtract: negative counts")
        return TileMultiset([a - b for a, b in zip(self.counts, other.counts)])

    def leq(self, other: "TileMultiset") -> bool:
        return all(a >= b for a, b in zip(self.counts, other.counts))

    def total(self) -> int:
        return sum(self.counts)

    def copy(self) -> "TileMultiset":
        return TileMultiset(list(self.counts))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TileMultiset) and self.counts == other.counts
