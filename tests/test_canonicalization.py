import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rummy.meld import Meld, MeldKind
from rummy.rules import Ruleset
from rummy.state import new_game
from rummy.table import Table
from rummy.tiles import TileSlot


def test_table_canonicalization_idempotent_and_order_invariant():
    run_slots = [TileSlot(2), TileSlot(0), TileSlot(1)]
    group_slots = [TileSlot(13), TileSlot(26), TileSlot(39)]
    run = Meld(MeldKind.RUN, run_slots)
    group = Meld(MeldKind.GROUP, group_slots)

    table_a = Table([group, run])
    table_b = Table([run, group])

    canon_a = table_a.canonicalize()
    canon_b = table_b.canonicalize()

    assert canon_a.canonicalize().canonical_key() == canon_a.canonical_key()
    assert canon_a.canonical_key() == canon_b.canonical_key()
    assert table_a.stable_hash() == table_b.stable_hash()


def test_table_multiset_is_cached():
    meld = Meld.from_effective_run(color=0, start=1, length=3)
    table = Table([meld])

    first = table.multiset()
    second = table.multiset()

    assert first is second


def test_state_hash_equivalent_tables_match():
    rules = Ruleset(initial_hand_size=0, initial_meld_min_points=0)
    state_a = new_game(ruleset=rules, rng_seed=1)
    state_b = new_game(ruleset=rules, rng_seed=1)

    run = Meld.from_effective_run(color=0, start=1, length=3)
    group = Meld.from_effective_group(value=7, colors=[0, 1, 2])

    state_a.table = Table([group, run])
    state_b.table = Table([run, group])

    assert state_a.state_key() == state_b.state_key()
    assert state_a.stable_hash() == state_b.stable_hash()
