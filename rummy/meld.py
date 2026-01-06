from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Tuple

from .tiles import TileSlot


class MeldKind(str, Enum):
    RUN = "RUN"
    GROUP = "GROUP"


@dataclass
class Meld:
    kind: MeldKind
    slots: List[TileSlot]

    def canonicalize(self) -> "Meld":
        sorted_slots = sorted(
            self.slots,
            key=lambda s: (
                s.effective_value() if self.kind == MeldKind.RUN else s.effective_color(),
                s.effective_color() if self.kind == MeldKind.RUN else s.effective_value(),
                s.tile_id,
            ),
        )
        return Meld(self.kind, sorted_slots)

    def is_valid(self) -> Tuple[bool, str]:
        if len(self.slots) < 3:
            return False, "meld too short"
        try:
            effective_colors = [s.effective_color() for s in self.slots]
            effective_values = [s.effective_value() for s in self.slots]
        except ValueError as exc:
            return False, str(exc)

        if self.kind == MeldKind.RUN:
            if len(set(effective_colors)) != 1:
                return False, "run must have same color"
            sorted_vals = sorted(effective_values)
            if sorted_vals != list(range(sorted_vals[0], sorted_vals[0] + len(sorted_vals))):
                return False, "run must be consecutive"
            if len(set(sorted_vals)) != len(sorted_vals):
                return False, "run must not duplicate value"
            return True, ""

        if self.kind == MeldKind.GROUP:
            if len(self.slots) not in (3, 4):
                return False, "group must have length 3 or 4"
            if len(set(effective_values)) != 1:
                return False, "group must share value"
            if len(set(effective_colors)) != len(effective_colors):
                return False, "group colors must be distinct"
            return True, ""

        return False, "unknown meld kind"

    def effective_signature(self) -> Tuple:
        canon = self.canonicalize()
        return (
            canon.kind.value,
            tuple((slot.effective_color(), slot.effective_value(), slot.tile_id) for slot in canon.slots),
        )

    @classmethod
    def from_effective_run(cls, color: int, start: int, length: int) -> "Meld":
        slots = [TileSlot.from_effective(color, start + i) for i in range(length)]
        return cls(MeldKind.RUN, slots)

    @classmethod
    def from_effective_group(cls, value: int, colors: Iterable[int]) -> "Meld":
        slots = [TileSlot.from_effective(color, value) for color in colors]
        return cls(MeldKind.GROUP, slots)
