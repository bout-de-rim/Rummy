from __future__ import annotations

import argparse
import random
from typing import Optional

from .engine import apply_move, is_legal_move
from .meld import Meld, MeldKind
from .move import Move
from .rules import Ruleset
from .state import GameState, new_game
from .tiles import TileSlot
from .multiset import TileMultiset


def _find_run_in_hand(state: GameState) -> Optional[Meld]:
    hand = state.hands[state.current_player].counts
    for color in range(state.ruleset.colors):
        tiles = [color * 13 + i for i in range(state.ruleset.values)]
        streak: list[int] = []
        for tile_id in tiles:
            if hand[tile_id]:
                streak.append(tile_id)
            else:
                if len(streak) >= 3:
                    return Meld(MeldKind.RUN, [TileSlot(t) for t in streak[:3]])
                streak = []
        if len(streak) >= 3:
            return Meld(MeldKind.RUN, [TileSlot(t) for t in streak[:3]])
    return None


def _find_group_in_hand(state: GameState) -> Optional[Meld]:
    hand = state.hands[state.current_player].counts
    for value in range(1, state.ruleset.values + 1):
        available = []
        for color in range(state.ruleset.colors):
            tile_id = color * 13 + (value - 1)
            if hand[tile_id]:
                available.append(tile_id)
        if len(available) >= 3:
            return Meld(MeldKind.GROUP, [TileSlot(t) for t in available[:3]])
    return None


def _choose_move(state: GameState, rng: random.Random) -> Move:
    run = _find_run_in_hand(state)
    if run:
        delta_tiles = [slot.tile_id for slot in run.slots]
        delta = TileMultiset.from_iterable(delta_tiles)
        new_table = state.table.canonicalize()
        new_table.melds.append(run)
        candidate = Move.play(delta_from_hand=delta, new_table=new_table)
        legal, _ = is_legal_move(state, candidate)
        if legal:
            return candidate

    group = _find_group_in_hand(state)
    if group:
        delta_tiles = [slot.tile_id for slot in group.slots]
        delta = TileMultiset.from_iterable(delta_tiles)
        new_table = state.table.canonicalize()
        new_table.melds.append(group)
        candidate = Move.play(delta_from_hand=delta, new_table=new_table)
        legal, _ = is_legal_move(state, candidate)
        if legal:
            return candidate

    if state.deck_index < len(state.deck_order):
        return Move.draw()
    return Move.skip()


def run_game(seed: Optional[int] = None) -> GameState:
    rules = Ruleset()
    rng = random.Random(seed)
    state = new_game(ruleset=rules, rng_seed=seed)
    for _ in range(500):
        if state.winner is not None:
            break
        move = _choose_move(state, rng)
        state = apply_move(state, move)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight Rummikub CLI simulation.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible deck order.")
    args = parser.parse_args()
    state = run_game(seed=args.seed)
    print(f"Game finished after {state.turn_number} turns")
    if state.winner is not None:
        print(f"Winner: player {state.winner}")
    else:
        print("No winner (turn limit reached)")
    print("Hands sizes:", [h.total() for h in state.hands])
    print("Table melds:", len(state.table.melds))


if __name__ == "__main__":
    main()
