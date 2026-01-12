from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Iterable, List, Tuple

from .meld import Meld
from .multiset import TileMultiset
from .tiles import TileSlot


@dataclass
class Table:
    melds: List[Meld]
    _multiset_cache: TileMultiset | None = field(default=None, init=False, repr=False)

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
        if self._multiset_cache is None:
            slots = (slot for meld in self.melds for slot in meld.slots)
            self._multiset_cache = TileMultiset.from_iterable(slot.tile_id for slot in slots)
        return self._multiset_cache

    def canonical_key(self) -> Tuple:
        canon = self.canonicalize()
        return tuple(meld.effective_signature() for meld in canon.melds)

    def stable_hash(self) -> str:
        key = self.canonical_key()
        return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()

    def all_slots(self) -> Iterable[TileSlot]:
        for meld in self.melds:
            for slot in meld.slots:
                yield slot
