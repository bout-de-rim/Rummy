from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .multiset import TileMultiset
from .table import Table


class MoveKind(str, Enum):
    DRAW = "DRAW"
    PASS = "PASS"
    PLAY = "PLAY"


@dataclass(frozen=True)
class PlayPayload:
    delta_from_hand: TileMultiset
    new_table: Table


@dataclass(frozen=True)
class Move:
    kind: MoveKind
    payload: Optional[PlayPayload] = None

    @staticmethod
    def draw() -> "Move":
        return Move(MoveKind.DRAW)

    @staticmethod
    def skip() -> "Move":
        return Move(MoveKind.PASS)

    @staticmethod
    def play(delta_from_hand: TileMultiset, new_table: Table) -> "Move":
        return Move(MoveKind.PLAY, PlayPayload(delta_from_hand, new_table))
