import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rummy.gui import (
    TimelineController,
    build_play_move,
    compute_delta_from_tables,
    remaining_hand_after_edit,
    resolve_screenshot_path,
)
from rummy.meld import Meld
from rummy.move import MoveKind
from rummy.multiset import TileMultiset
from rummy.rules import Ruleset
from rummy.state import new_game
from rummy.table import Table
from rummy.tiles import TileSlot


def test_compute_delta_rejects_missing_table_tile():
    base = Table([Meld.from_effective_run(color=0, start=1, length=3)])
    edited = Table([Meld.from_effective_run(color=0, start=1, length=2)])

    delta, reason = compute_delta_from_tables(base, edited)
    assert delta is None
    assert "missing tiles" in reason


def test_build_play_move_success_and_remaining_hand_updates():
    rules = Ruleset(initial_hand_size=0, initial_meld_min_points=0)
    state = new_game(ruleset=rules, rng_seed=1)
    state.hands[0] = TileMultiset.from_iterable([0, 1, 2])

    edited_table = Table([Meld.from_effective_run(color=0, start=1, length=3)])
    move, reason = build_play_move(state, edited_table)

    assert move is not None, reason
    assert move.kind == MoveKind.PLAY
    assert move.payload is not None
    assert move.payload.delta_from_hand == TileMultiset.from_iterable([0, 1, 2])

    remaining = remaining_hand_after_edit(state, edited_table)
    assert remaining.total() == 0


def test_timeline_controller_appends_and_jumps():
    rules = Ruleset(initial_hand_size=0, initial_meld_min_points=0)
    state = new_game(ruleset=rules, rng_seed=2)

    controller = TimelineController.from_state(state)
    assert controller.current is state
    assert controller.at_end()

    controller.append(state)
    assert controller.index == 1
    assert controller.current is state

    controller.jump(0)
    assert controller.index == 0


def test_resolve_screenshot_path_reads_env(tmp_path, monkeypatch):
    path = tmp_path / "shot.png"
    monkeypatch.setenv("RUMMY_GUI_SCREENSHOT_PATH", str(path))
    resolved = resolve_screenshot_path()
    assert resolved == path
