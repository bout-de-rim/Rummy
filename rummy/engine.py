from __future__ import annotations

from typing import List, Tuple

from .meld import Meld, MeldKind
from .move import Move, MoveKind, PlayPayload
from .multiset import TileMultiset
from .rules import Ruleset
from .state import GameEvent, GameState
from .table import Table
from .tiles import JOKER_ID, TileSlot


def _meld_points(meld: Meld) -> int:
    return sum(slot.effective_value() for slot in meld.slots)


def _validate_initial_meld(
    ruleset: Ruleset, player: int, old_table: Table, new_table: Table, delta_from_hand: TileMultiset
) -> Tuple[bool, str]:
    from collections import Counter

    old_canon = old_table.canonicalize().melds
    new_canon = new_table.canonicalize().melds

    old_counter = Counter(m.effective_signature() for m in old_canon)
    new_counter = Counter(m.effective_signature() for m in new_canon)
    for sig, count in old_counter.items():
        if new_counter[sig] < count:
            return False, "initial meld cannot rearrange existing table"

    added_melds: List[Meld] = []
    temp_counter = old_counter.copy()
    for meld in new_canon:
        sig = meld.effective_signature()
        if temp_counter[sig] > 0:
            temp_counter[sig] -= 1
        else:
            added_melds.append(meld)

    if ruleset.initial_meld_must_use_only_hand:
        if not added_melds:
            return False, "initial meld must introduce new melds"
        total_points = sum(_meld_points(m) for m in added_melds)
        if total_points < ruleset.initial_meld_min_points:
            return False, f"initial meld must score at least {ruleset.initial_meld_min_points}"
        if not ruleset.initial_meld_allow_joker and any(slot.tile_id == JOKER_ID for meld in added_melds for slot in meld.slots):
            return False, "joker not allowed in initial meld"
    return True, ""


def is_legal_move(state: GameState, move: Move) -> Tuple[bool, str]:
    if state.winner is not None:
        return False, "game already finished"

    ruleset = state.ruleset
    hand = state.hands[state.current_player]

    if move.kind == MoveKind.DRAW:
        if state.deck_index >= len(state.deck_order):
            return False, "deck is empty"
        return True, ""

    if move.kind == MoveKind.PASS:
        return True, ""

    if move.kind != MoveKind.PLAY or move.payload is None:
        return False, "invalid move payload"

    payload: PlayPayload = move.payload
    if any(a < b for a, b in zip(hand.counts, payload.delta_from_hand.counts)):
        return False, "cannot play tiles not in hand"

    try:
        new_table = payload.new_table.canonicalize()
    except ValueError as exc:
        return False, str(exc)
    new_table_ms = new_table.multiset()
    expected_table = state.table.multiset().add(payload.delta_from_hand)
    if new_table_ms != expected_table:
        return False, "table tiles must match previous table plus played tiles"

    for meld in new_table.melds:
        ok, reason = meld.is_valid()
        if not ok:
            return False, f"invalid meld: {reason}"

    if not state.initial_meld_done[state.current_player]:
        ok, reason = _validate_initial_meld(ruleset, state.current_player, state.table, new_table, payload.delta_from_hand)
        if not ok:
            return False, reason

    return True, ""


def _advance_player(state: GameState) -> None:
    state.current_player = (state.current_player + 1) % state.ruleset.num_players
    state.turn_number += 1


def _apply_draw(state: GameState) -> None:
    tile = state.deck_order[state.deck_index]
    state.deck_index += 1
    state.hands[state.current_player].counts[tile] += 1
    state.event_log.append(GameEvent(player=state.current_player, move_kind=MoveKind.DRAW.value, payload={"tile": tile}))
    if state.ruleset.draw_ends_turn:
        _advance_player(state)


def _apply_pass(state: GameState) -> None:
    state.event_log.append(GameEvent(player=state.current_player, move_kind=MoveKind.PASS.value, payload={}))
    _advance_player(state)


def _apply_play(state: GameState, payload: PlayPayload) -> None:
    for idx, count in enumerate(payload.delta_from_hand.counts):
        state.hands[state.current_player].counts[idx] -= count
    state.table = payload.new_table.canonicalize()
    state.event_log.append(
        GameEvent(
            player=state.current_player,
            move_kind=MoveKind.PLAY.value,
            payload={
                "delta": payload.delta_from_hand.to_compact(),
                "table": [
                    {
                        "kind": meld.kind.value,
                        "slots": [slot.signature() for slot in meld.slots],
                    }
                    for meld in state.table.melds
                ],
            },
        )
    )
    if not state.initial_meld_done[state.current_player]:
        state.initial_meld_done[state.current_player] = True
    if state.hands[state.current_player].total() == 0:
        state.winner = state.current_player
    _advance_player(state)


def apply_move(state: GameState, move: Move) -> GameState:
    legal, reason = is_legal_move(state, move)
    if not legal:
        raise ValueError(f"illegal move: {reason}")

    new_state = state.copy()
    if move.kind == MoveKind.DRAW:
        _apply_draw(new_state)
    elif move.kind == MoveKind.PASS:
        _apply_pass(new_state)
    elif move.kind == MoveKind.PLAY and move.payload:
        _apply_play(new_state, move.payload)
    return new_state


def replay_event_log(initial_state: GameState, events: List[GameEvent]) -> GameState:
    state = initial_state.copy()
    for event in events:
        if event.move_kind == MoveKind.DRAW.value:
            state = apply_move(state, Move.draw())
        elif event.move_kind == MoveKind.PASS.value:
            state = apply_move(state, Move.skip())
        elif event.move_kind == MoveKind.PLAY.value:
            table_melds: List[Meld] = []
            for meld_dict in event.payload["table"]:
                kind = MeldKind(meld_dict["kind"])
                slots = [
                    TileSlot(tile_id, None if assigned_color == -1 else assigned_color, None if assigned_value == -1 else assigned_value)
                    for tile_id, assigned_color, assigned_value in meld_dict["slots"]
                ]
                table_melds.append(Meld(kind=kind, slots=slots))
            delta = TileMultiset.from_compact(event.payload["delta"])
            state = apply_move(state, Move.play(delta_from_hand=delta, new_table=Table(table_melds)))
        else:
            raise ValueError(f"Unknown event kind {event.move_kind}")
    return state
