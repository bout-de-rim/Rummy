import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rummy.candidates import generate_candidate_moves
from rummy.engine import is_legal_move
from rummy.meld import Meld
from rummy.move import MoveKind
from rummy.multiset import TileMultiset
from rummy.rules import Ruleset
from rummy.state import new_game
from rummy.table import Table


def test_generate_candidates_are_legal():
    rules = Ruleset(initial_hand_size=0, initial_meld_min_points=0)
    state = new_game(ruleset=rules, rng_seed=7)
    state.hands[0] = TileMultiset.from_iterable([0, 1, 2, 13, 26])

    moves = generate_candidate_moves(state, k=15, budget=0)
    assert moves, "expected at least one candidate move"
    for move in moves:
        legal, reason = is_legal_move(state, move)
        assert legal, reason


def test_generate_candidates_includes_run_play():
    rules = Ruleset(initial_hand_size=0, initial_meld_min_points=0)
    state = new_game(ruleset=rules, rng_seed=11)
    state.hands[0] = TileMultiset.from_iterable([0, 1, 2])

    moves = generate_candidate_moves(state, k=10, budget=0)
    expected_delta = TileMultiset.from_iterable([0, 1, 2])
    assert any(
        move.kind == MoveKind.PLAY
        and move.payload is not None
        and move.payload.delta_from_hand == expected_delta
        for move in moves
    )


def test_generate_candidates_extend_run_with_budget():
    rules = Ruleset(initial_hand_size=0, initial_meld_min_points=0)
    state = new_game(ruleset=rules, rng_seed=21)
    state.table = Table([Meld.from_effective_run(color=0, start=1, length=3)])
    state.hands[0] = TileMultiset.from_iterable([3])
    state.initial_meld_done[0] = True

    moves = generate_candidate_moves(state, k=10, budget=1)
    assert any(
        move.kind == MoveKind.PLAY
        and move.payload is not None
        and any(
            meld.kind.value == "RUN"
            and [slot.effective_value() for slot in meld.slots] == [1, 2, 3, 4]
            for meld in move.payload.new_table.canonicalize().melds
        )
        for move in moves
    )
