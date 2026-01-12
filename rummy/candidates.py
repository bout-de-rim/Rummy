from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Optional, Sequence, Tuple

from .engine import is_legal_move
from .meld import Meld, MeldKind
from .move import Move, MoveKind
from .multiset import TileMultiset
from .state import GameState
from .table import Table
from .tiles import TileSlot


@dataclass(frozen=True)
class _MeldCandidate:
    meld: Meld
    delta: TileMultiset
    tile_count: int


def _can_use(available: Sequence[int], needed: Sequence[int]) -> bool:
    return all(a >= b for a, b in zip(available, needed))


def _run_melds_from_hand(hand: TileMultiset, colors: int, values: int) -> Iterable[_MeldCandidate]:
    counts = hand.counts
    for color in range(colors):
        present = [value for value in range(1, values + 1) if counts[color * 13 + (value - 1)] > 0]
        if not present:
            continue
        streak: List[int] = []
        for value in range(1, values + 1):
            if value in present:
                streak.append(value)
            else:
                if len(streak) >= 3:
                    yield from _subruns(color, streak)
                streak = []
        if len(streak) >= 3:
            yield from _subruns(color, streak)


def _subruns(color: int, streak: List[int]) -> Iterable[_MeldCandidate]:
    for i in range(len(streak)):
        for j in range(i + 3, len(streak) + 1):
            values = streak[i:j]
            slots = [TileSlot.from_effective(color, value) for value in values]
            meld = Meld(MeldKind.RUN, slots)
            delta = TileMultiset.from_iterable(slot.tile_id for slot in slots)
            yield _MeldCandidate(meld, delta, len(slots))


def _group_melds_from_hand(hand: TileMultiset, colors: int, values: int) -> Iterable[_MeldCandidate]:
    counts = hand.counts
    for value in range(1, values + 1):
        available_colors = [
            color for color in range(colors) if counts[color * 13 + (value - 1)] > 0
        ]
        if len(available_colors) < 3:
            continue
        for size in (3, 4):
            if len(available_colors) < size:
                continue
            for combo in combinations(available_colors, size):
                slots = [TileSlot.from_effective(color, value) for color in combo]
                meld = Meld(MeldKind.GROUP, slots)
                delta = TileMultiset.from_iterable(slot.tile_id for slot in slots)
                yield _MeldCandidate(meld, delta, len(slots))


def _hand_meld_candidates(hand: TileMultiset, colors: int, values: int) -> List[_MeldCandidate]:
    melds = list(_run_melds_from_hand(hand, colors, values))
    melds.extend(_group_melds_from_hand(hand, colors, values))
    return melds


def _extend_run_candidates(
    table: Table, hand: TileMultiset, colors: int, values: int
) -> Iterable[Tuple[Table, TileMultiset]]:
    for idx, meld in enumerate(table.melds):
        if meld.kind != MeldKind.RUN:
            continue
        effective_color = meld.slots[0].effective_color()
        effective_values = sorted(slot.effective_value() for slot in meld.slots)
        vmin, vmax = effective_values[0], effective_values[-1]

        left_values: List[int] = []
        value = vmin - 1
        while value >= 1 and hand.counts[effective_color * 13 + (value - 1)] > 0:
            left_values.append(value)
            value -= 1

        right_values: List[int] = []
        value = vmax + 1
        while value <= values and hand.counts[effective_color * 13 + (value - 1)] > 0:
            right_values.append(value)
            value += 1

        for left_count in range(len(left_values) + 1):
            for right_count in range(len(right_values) + 1):
                if left_count + right_count == 0:
                    continue
                new_values = (
                    list(reversed(left_values[:left_count])) + effective_values + right_values[:right_count]
                )
                new_slots = [TileSlot.from_effective(effective_color, value) for value in new_values]
                new_meld = Meld(MeldKind.RUN, new_slots)
                delta_slots = [
                    TileSlot.from_effective(effective_color, value)
                    for value in left_values[:left_count] + right_values[:right_count]
                ]
                delta = TileMultiset.from_iterable(slot.tile_id for slot in delta_slots)
                new_table = Table(
                    table.melds[:idx] + [new_meld] + table.melds[idx + 1 :],
                )
                yield new_table, delta


def _extend_group_candidates(
    table: Table, hand: TileMultiset, colors: int
) -> Iterable[Tuple[Table, TileMultiset]]:
    for idx, meld in enumerate(table.melds):
        if meld.kind != MeldKind.GROUP:
            continue
        if len(meld.slots) >= 4:
            continue
        value = meld.slots[0].effective_value()
        existing_colors = {slot.effective_color() for slot in meld.slots}
        for color in range(colors):
            if color in existing_colors:
                continue
            tile_id = color * 13 + (value - 1)
            if hand.counts[tile_id] <= 0:
                continue
            new_slots = list(meld.slots) + [TileSlot.from_effective(color, value)]
            new_meld = Meld(MeldKind.GROUP, new_slots)
            delta = TileMultiset.from_iterable([tile_id])
            new_table = Table(table.melds[:idx] + [new_meld] + table.melds[idx + 1 :])
            yield new_table, delta


def _beam_combine_melds(
    hand: TileMultiset, melds: List[_MeldCandidate], max_melds: int, beam_width: int
) -> Iterable[List[_MeldCandidate]]:
    @dataclass(frozen=True)
    class _BeamState:
        chosen: Tuple[_MeldCandidate, ...]
        remaining: TileMultiset
        tile_count: int

    beam: List[_BeamState] = [_BeamState((), hand, 0)]
    for _ in range(max_melds):
        next_beam: List[_BeamState] = []
        for state in beam:
            for candidate in melds:
                if not _can_use(state.remaining.counts, candidate.delta.counts):
                    continue
                remaining = state.remaining.sub(candidate.delta)
                chosen = state.chosen + (candidate,)
                next_beam.append(
                    _BeamState(
                        chosen=chosen,
                        remaining=remaining,
                        tile_count=state.tile_count + candidate.tile_count,
                    )
                )
        if not next_beam:
            break
        next_beam.sort(key=lambda s: s.tile_count, reverse=True)
        beam = next_beam[:beam_width]
        for state in beam:
            yield list(state.chosen)


def _move_key(move: Move) -> Tuple:
    if move.kind != MoveKind.PLAY or move.payload is None:
        return (move.kind.value,)
    payload = move.payload
    return (
        move.kind.value,
        tuple(payload.delta_from_hand.to_compact()),
        payload.new_table.canonical_key(),
    )


def generate_candidate_moves(state: GameState, k: int = 50, budget: int = 0) -> List[Move]:
    ruleset = state.ruleset
    hand = state.hands[state.current_player]
    table = state.table.canonicalize()
    candidates: List[Move] = []
    seen = set()

    def add_move(move: Move) -> None:
        if len(candidates) >= k:
            return
        legal, _ = is_legal_move(state, move)
        if not legal:
            return
        key = _move_key(move)
        if key in seen:
            return
        seen.add(key)
        candidates.append(move)

    meld_candidates = _hand_meld_candidates(hand, ruleset.colors, ruleset.values)
    meld_candidates.sort(key=lambda c: c.tile_count, reverse=True)

    for meld_candidate in meld_candidates:
        new_table = Table(table.melds + [meld_candidate.meld])
        move = Move.play(delta_from_hand=meld_candidate.delta, new_table=new_table)
        add_move(move)

    beam_width = min(max(1, k), 8)
    for melds in _beam_combine_melds(hand, meld_candidates, max_melds=3, beam_width=beam_width):
        if not melds:
            continue
        delta = TileMultiset.empty()
        new_melds = list(table.melds)
        for candidate in melds:
            delta = delta.add(candidate.delta)
            new_melds.append(candidate.meld)
        move = Move.play(delta_from_hand=delta, new_table=Table(new_melds))
        add_move(move)

    if budget >= 1:
        for new_table, delta in _extend_run_candidates(table, hand, ruleset.colors, ruleset.values):
            add_move(Move.play(delta_from_hand=delta, new_table=new_table))
        for new_table, delta in _extend_group_candidates(table, hand, ruleset.colors):
            add_move(Move.play(delta_from_hand=delta, new_table=new_table))

    if state.deck_index < len(state.deck_order):
        add_move(Move.draw())
    else:
        add_move(Move.skip())

    return candidates[:k]
