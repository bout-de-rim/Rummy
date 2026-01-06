from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

JOKER_ID = 52
MAX_TILE_ID = 52
MULTISET_SIZE = 53


def color_of(tile_id: int) -> int:
    if tile_id == JOKER_ID:
        raise ValueError("Joker has no inherent color")
    return tile_id // 13


def value_of(tile_id: int) -> int:
    if tile_id == JOKER_ID:
        raise ValueError("Joker has no inherent value")
    return (tile_id % 13) + 1


@dataclass(frozen=True)
class TileSlot:
    tile_id: int
    assigned_color: int | None = None
    assigned_value: int | None = None

    def effective_color(self) -> int:
        if self.tile_id == JOKER_ID:
            if self.assigned_color is None or self.assigned_value is None:
                raise ValueError("Unassigned joker")
            return self.assigned_color
        return color_of(self.tile_id)

    def effective_value(self) -> int:
        if self.tile_id == JOKER_ID:
            if self.assigned_color is None or self.assigned_value is None:
                raise ValueError("Unassigned joker")
            return self.assigned_value
        return value_of(self.tile_id)

    def is_joker(self) -> bool:
        return self.tile_id == JOKER_ID

    def signature(self) -> Tuple[int, int, int]:
        return (
            self.tile_id,
            -1 if self.assigned_color is None else self.assigned_color,
            -1 if self.assigned_value is None else self.assigned_value,
        )

    @classmethod
    def from_effective(cls, color: int, value: int, use_joker: bool = False) -> "TileSlot":
        if use_joker:
            return cls(JOKER_ID, assigned_color=color, assigned_value=value)
        return cls(color * 13 + (value - 1))


def iter_full_deck(colors: int, values: int, copies: int, num_jokers: int) -> Iterable[int]:
    for _ in range(copies):
        for color in range(colors):
            for value in range(1, values + 1):
                yield color * 13 + (value - 1)
    for _ in range(num_jokers):
        yield JOKER_ID


def compact_multiset_from_slots(slots: Iterable[TileSlot]) -> List[Tuple[int, int]]:
    counts = [0] * MULTISET_SIZE
    for slot in slots:
        counts[slot.tile_id] += 1
    return [(idx, count) for idx, count in enumerate(counts) if count]
