import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rummy.engine import apply_move, is_legal_move, replay_event_log
from rummy.meld import Meld
from rummy.move import Move
from rummy.multiset import TileMultiset
from rummy.rules import Ruleset
from rummy.state import new_game
from rummy.table import Table
from rummy.tiles import TileSlot


def test_draw_updates_hand_and_deck():
    rules = Ruleset(initial_hand_size=1)
    state = new_game(ruleset=rules, rng_seed=42)
    initial_hand_size = state.hands[state.current_player].total()
    initial_deck_index = state.deck_index

    move = Move.draw()
    legal, reason = is_legal_move(state, move)
    assert legal, reason

    next_state = apply_move(state, move)
    assert next_state.deck_index == initial_deck_index + 1
    previous_player = (next_state.current_player - 1) % rules.num_players
    assert next_state.hands[previous_player].total() == initial_hand_size + 1


def test_play_respects_conservation_and_valid_meld():
    rules = Ruleset(initial_hand_size=0, initial_meld_min_points=0)
    state = new_game(ruleset=rules, rng_seed=1)
    state.hands[0] = TileMultiset.from_iterable([0, 1, 2])

    meld = Meld.from_effective_run(color=0, start=1, length=3)
    delta = TileMultiset.from_iterable([0, 1, 2])
    move = Move.play(delta_from_hand=delta, new_table=Table([meld]))
    legal, reason = is_legal_move(state, move)
    assert legal, reason

    next_state = apply_move(state, move)
    assert next_state.table.melds, "meld added to table"
    assert next_state.hands[0].total() == 0
    assert next_state.initial_meld_done[0] is True


def test_illegal_meld_is_rejected():
    rules = Ruleset(initial_hand_size=0, initial_meld_min_points=0)
    state = new_game(ruleset=rules, rng_seed=2)
    state.hands[0] = TileMultiset.from_iterable([0, 2, 3])

    invalid_meld = Meld(Meld.from_effective_run(color=0, start=1, length=2).kind, [TileSlot(0), TileSlot(2), TileSlot(3)])
    delta = TileMultiset.from_iterable([0, 2, 3])
    move = Move.play(delta_from_hand=delta, new_table=Table([invalid_meld]))
    legal, _ = is_legal_move(state, move)
    assert not legal


def test_replay_is_deterministic():
    rules = Ruleset(initial_hand_size=1, initial_meld_min_points=0)
    base_state = new_game(ruleset=rules, rng_seed=123)
    state = base_state
    state = apply_move(state, Move.draw())
    state = apply_move(state, Move.draw())

    replayed = replay_event_log(base_state, state.event_log)
    assert replayed.state_key() == state.state_key()
