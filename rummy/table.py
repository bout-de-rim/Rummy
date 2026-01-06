from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .meld import Meld
from .multiset import TileMultiset
from .tiles import TileSlot


@dataclass
class Table:
    melds: List[Meld]

    def canonicalize(self) -> "Table":
        canon_melds = []
        for m in self.melds:
            if not m.slots:
                raise ValueError("meld cannot be empty")
            canon_melds.append(m.canonicalize())
        canon_melds.sort(
            key=lambda m: (
                m.kind.value,
                m.slots[0].effective_color(),
                m.slots[0].effective_value(),
                len(m.slots),
                m.effective_signature(),
            )
        )
        return Table(canon_melds)

    def multiset(self) -> TileMultiset:
        slots = (slot for meld in self.melds for slot in meld.slots)
        return TileMultiset.from_iterable(slot.tile_id for slot in slots)

    def all_slots(self) -> Iterable[TileSlot]:
        for meld in self.melds:
            for slot in meld.slots:
                yield slot
