from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .multiset import TileMultiset
from .rules import Ruleset
from .table import Table
from .tiles import iter_full_deck


@dataclass
class GameEvent:
    player: int
    move_kind: str
    payload: dict


@dataclass
class GameState:
    ruleset: Ruleset
    current_player: int
    hands: List[TileMultiset]
    table: Table
    deck_order: List[int]
    deck_index: int
    initial_meld_done: List[bool]
    turn_number: int
    rng_seed: Optional[int] = None
    event_log: List[GameEvent] = field(default_factory=list)
    winner: Optional[int] = None

    def copy(self) -> "GameState":
        return GameState(
            ruleset=self.ruleset,
            current_player=self.current_player,
            hands=[h.copy() for h in self.hands],
            table=copy.deepcopy(self.table),
            deck_order=list(self.deck_order),
            deck_index=self.deck_index,
            initial_meld_done=list(self.initial_meld_done),
            turn_number=self.turn_number,
            rng_seed=self.rng_seed,
            event_log=list(self.event_log),
            winner=self.winner,
        )

    def state_key(self) -> Tuple:
        return (
            self.current_player,
            self.deck_index,
            self.turn_number,
            tuple(tuple(h.counts) for h in self.hands),
            self.table.canonical_key(),
            tuple(self.initial_meld_done),
            self.winner,
        )

    def stable_hash(self) -> str:
        return hashlib.sha256(repr(self.state_key()).encode("utf-8")).hexdigest()


def _deal_initial_hands(deck: List[int], ruleset: Ruleset, rng: random.Random) -> Tuple[List[TileMultiset], int, List[int]]:
    deck_copy = deck[:]
    rng.shuffle(deck_copy)
    hands = [TileMultiset.empty() for _ in range(ruleset.num_players)]
    idx = 0
    for _ in range(ruleset.initial_hand_size):
        for player in range(ruleset.num_players):
            tile = deck_copy[idx]
            hands[player].counts[tile] += 1
            idx += 1
    return hands, idx, deck_copy


def new_game(ruleset: Ruleset | None = None, rng_seed: Optional[int] = None) -> GameState:
    ruleset = ruleset or Ruleset()
    rng = random.Random(rng_seed)
    deck = list(iter_full_deck(ruleset.colors, ruleset.values, ruleset.copies_per_tiletype, ruleset.num_jokers))
    hands, deck_index, deck_order = _deal_initial_hands(deck, ruleset, rng)
    return GameState(
        ruleset=ruleset,
        current_player=0,
        hands=hands,
        table=Table([]),
        deck_order=deck_order,
        deck_index=deck_index,
        initial_meld_done=[False] * ruleset.num_players,
        turn_number=0,
        rng_seed=rng_seed,
    )
