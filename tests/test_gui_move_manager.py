import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rummy.gui import DraftMoveManager
from rummy.meld import Meld
from rummy.rules import Ruleset
from rummy.state import new_game
from rummy.table import Table
from rummy.tiles import JOKER_ID, TileSlot


def _table_signature(table: Table):
    return [m.effective_signature() for m in table.canonicalize().melds]


def _state_with_opening_done() -> Ruleset:
    rules = Ruleset(initial_hand_size=0, initial_meld_min_points=0)
    state = new_game(ruleset=rules, rng_seed=1)
    state.initial_meld_done[state.current_player] = True
    return state


def test_run_insert_extends_without_joker():
    state = _state_with_opening_done()
    table = Table([Meld.from_effective_run(color=0, start=1, length=3)])
    manager = DraftMoveManager(state, table)

    slot = TileSlot.from_effective(color=0, value=4)
    ok, reason = manager.insert_slot_into_target(slot, ("run", 0, -1))

    assert ok, reason
    assert len(manager.edited_table.melds) == 1
    meld = manager.edited_table.melds[0]
    valid, reason = meld.is_valid()
    assert valid, reason
    assert len(meld.slots) == 4


def test_run_insert_joker_bridges_runs():
    state = _state_with_opening_done()
    table = Table(
        [
            Meld.from_effective_run(color=0, start=1, length=2),
            Meld.from_effective_run(color=0, start=4, length=2),
        ]
    )
    manager = DraftMoveManager(state, table, hand_joker_value=1)

    slot = TileSlot(JOKER_ID, assigned_color=None, assigned_value=None)
    ok, reason = manager.insert_slot_into_target(slot, ("run", 0, -1))

    assert ok, reason
    assert len(manager.edited_table.melds) == 1
    meld = manager.edited_table.melds[0]
    valid, reason = meld.is_valid()
    assert valid, reason
    joker_slots = [s for s in meld.slots if s.tile_id == JOKER_ID]
    assert joker_slots
    assert joker_slots[0].assigned_color == 0
    assert joker_slots[0].assigned_value == 3


def test_group_insert_joker_selects_missing_color():
    state = _state_with_opening_done()
    table = Table([Meld.from_effective_group(value=7, colors=[0, 1])])
    manager = DraftMoveManager(state, table)

    slot = TileSlot(JOKER_ID, assigned_color=None, assigned_value=None)
    slot = manager.adapt_slot_for_target(slot, ("group", 0, 6))
    ok, reason = manager.insert_slot_into_target(slot, ("group", 0, 6))

    assert ok, reason
    meld = manager.edited_table.melds[0]
    valid, reason = meld.is_valid()
    assert valid, reason
    joker_slots = [s for s in meld.slots if s.tile_id == JOKER_ID]
    assert joker_slots
    assert joker_slots[0].assigned_value == 7
    assert joker_slots[0].assigned_color == 2


def test_opening_blocks_existing_group_insert():
    rules = Ruleset(initial_hand_size=0, initial_meld_min_points=30)
    state = new_game(ruleset=rules, rng_seed=2)
    table = Table([Meld.from_effective_group(value=5, colors=[0, 1, 2])])
    state.table = table
    manager = DraftMoveManager(state, table)
    original = _table_signature(manager.edited_table)

    slot = TileSlot.from_effective(color=3, value=5)
    ok, reason = manager.insert_slot_into_target(slot, ("group", 0, 4))

    assert not ok
    assert "ouverture" in reason
    assert _table_signature(manager.edited_table) == original


def test_move_table_slot_rolls_back_on_invalid_drop():
    state = _state_with_opening_done()
    table = Table(
        [
            Meld.from_effective_run(color=0, start=1, length=3),
            Meld.from_effective_group(value=5, colors=[0, 1, 2]),
        ]
    )
    manager = DraftMoveManager(state, table)
    original = _table_signature(manager.edited_table)

    ok, reason = manager.move_table_slot(0, 1, ("group", 0, 4))

    assert not ok
    assert "Déplacement refusé" in reason
    assert _table_signature(manager.edited_table) == original
