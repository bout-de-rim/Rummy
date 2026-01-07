from __future__ import annotations

"""Pygame GUI for Rummikub (Sprint 2).

This module contains two parts:
- pure helpers (`TimelineController`, `build_play_move`, ...), testable without
  pygame installed;
- the actual pygame UI (`launch_gui`), imported lazily to avoid making pygame a
  hard dependency for non-UI workflows.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from .engine import apply_move, is_legal_move
from .meld import Meld, MeldKind
from .move import Move
from .multiset import TileMultiset
from .rules import Ruleset
from .state import GameState, new_game
from .table import Table
from .tiles import JOKER_ID, TileSlot

try:  # pragma: no cover - runtime dependency only for the interactive UI
    import pygame
except Exception:  # pragma: no cover
    pygame = None


def _safe_canonical_table(table: Table) -> Tuple[Optional[Table], str]:
    try:
        return table.canonicalize(), ""
    except ValueError as exc:
        return None, str(exc)


def compute_delta_from_tables(base_table: Table, edited_table: Table) -> Tuple[Optional[TileMultiset], str]:
    """Return the multiset delta required from the hand to reach edited_table.

    The delta is defined by: new_table = base_table + delta_from_hand.
    If the edited table removes tiles that were already on the table, the delta
    is invalid.
    """

    base_ms = base_table.multiset().counts
    new_ms = edited_table.multiset().counts
    diff = [n - b for n, b in zip(new_ms, base_ms)]
    if any(d < 0 for d in diff):
        return None, "table is missing tiles from previous state"
    return TileMultiset(diff), ""


def build_play_move(state: GameState, edited_table: Table) -> Tuple[Optional[Move], str]:
    """Compute a PLAY move for the current player based on the edited table."""

    canonical_table, error = _safe_canonical_table(edited_table)
    if canonical_table is None:
        return None, error

    delta, error = compute_delta_from_tables(state.table.canonicalize(), canonical_table)
    if delta is None:
        return None, error

    if not state.hands[state.current_player].leq(delta):
        return None, "not enough tiles in hand"

    move = Move.play(delta_from_hand=delta, new_table=canonical_table)
    legal, reason = is_legal_move(state, move)
    if not legal:
        return None, reason
    return move, ""


def remaining_hand_after_edit(state: GameState, edited_table: Table) -> TileMultiset:
    """Compute the remaining hand after committing the edited_table.

    If the edited table is invalid, return the current hand unchanged. This is
    intended for UI rendering only.
    """

    canonical_table, error = _safe_canonical_table(edited_table)
    if canonical_table is None:
        return state.hands[state.current_player].copy()

    delta, error = compute_delta_from_tables(state.table.canonicalize(), canonical_table)
    if delta is None:
        return state.hands[state.current_player].copy()

    try:
        return state.hands[state.current_player].sub(delta)
    except ValueError:
        return state.hands[state.current_player].copy()


@dataclass
class TimelineController:
    """A lightweight timeline controller for stepping through GameState history."""

    history: List[GameState]
    index: int = 0

    @classmethod
    def from_state(cls, state: GameState) -> "TimelineController":
        return cls(history=[state], index=0)

    @property
    def current(self) -> GameState:
        return self.history[self.index]

    def at_end(self) -> bool:
        return self.index == len(self.history) - 1

    def append(self, state: GameState) -> None:
        if not self.at_end():
            self.history = self.history[: self.index + 1]
        self.history.append(state)
        self.index = len(self.history) - 1

    def jump(self, idx: int) -> None:
        idx = max(0, min(idx, len(self.history) - 1))
        self.index = idx


# --- Pygame UI -----------------------------------------------------------------


def launch_gui(seed: Optional[int] = None, ruleset: Optional[Ruleset] = None) -> None:  # pragma: no cover - interactive
    if pygame is None:
        raise ImportError("pygame is required for the GUI. Install it with `pip install pygame`." )

    ruleset = ruleset or Ruleset()
    timeline = TimelineController.from_state(new_game(ruleset=ruleset, rng_seed=seed))
    edited_table = timeline.current.table.canonicalize()

    pygame.init()
    pygame.display.set_caption("Rummikub — Sprint 2 GUI")
    screen = pygame.display.set_mode((1280, 768))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 18)
    small_font = pygame.font.SysFont("arial", 14)

    carried: Optional[TileSlot] = None
    carried_from: Optional[Tuple[str, int, int]] = None  # ("table"/"hand", meld_idx, slot_idx)
    selected_slot: Optional[Tuple[int, int]] = None
    message = "Left click to drag tiles. Right click a joker to tweak its assignment." \
              " Use <-/-> to navigate the timeline."
    show_godmode = False

    TILE_W, TILE_H = 50, 70
    TABLE_START_X = 30
    TABLE_START_Y = 80
    ROW_HEIGHT = TILE_H + 30

    def _slots_from_counts(counts: List[int]) -> List[TileSlot]:
        slots: List[TileSlot] = []
        for tile_id, count in enumerate(counts):
            for _ in range(count):
                if tile_id == JOKER_ID:
                    slots.append(TileSlot(JOKER_ID, assigned_color=0, assigned_value=1))
                else:
                    slots.append(TileSlot(tile_id))
        slots.sort(key=lambda s: (s.effective_color() if s.tile_id != JOKER_ID else s.assigned_color or 0, s.effective_value()))
        return slots

    def _slots_from_tile_ids(tile_ids: List[int]) -> List[TileSlot]:
        slots: List[TileSlot] = []
        for tile_id in tile_ids:
            if tile_id == JOKER_ID:
                slots.append(TileSlot(JOKER_ID, assigned_color=0, assigned_value=1))
            else:
                slots.append(TileSlot(tile_id))
        slots.sort(key=lambda s: (s.effective_color() if s.tile_id != JOKER_ID else s.assigned_color or 0, s.effective_value()))
        return slots

    color_palette = [
        (214, 72, 72),
        (75, 139, 190),
        (40, 168, 112),
        (214, 156, 76),
    ]
    bg_color = (26, 29, 38)
    panel_color = (40, 46, 60)
    text_color = (230, 232, 240)
    disabled_color = (110, 110, 120)
    accent_color = (109, 176, 255)

    def _tile_label(slot: TileSlot) -> str:
        if slot.tile_id == JOKER_ID:
            suffix = "?" if slot.assigned_value is None else str(slot.assigned_value)
            return f"J{suffix}"
        return str((slot.tile_id % 13) + 1)

    def _tile_color(slot: TileSlot) -> Tuple[int, int, int]:
        if slot.tile_id == JOKER_ID:
            return (180, 92, 200)
        return color_palette[slot.tile_id // 13 % len(color_palette)]

    def _draw_tile(surface, rect, slot: TileSlot, highlight: bool = False) -> None:
        pygame.draw.rect(surface, (245, 245, 245), rect, border_radius=6)
        pygame.draw.rect(surface, accent_color if highlight else (60, 60, 70), rect, width=2, border_radius=6)
        inner = rect.inflate(-10, -10)
        pygame.draw.rect(surface, _tile_color(slot), inner, border_radius=4)
        label = font.render(_tile_label(slot), True, (0, 0, 0))
        surface.blit(label, label.get_rect(center=inner.center))

    def _remove_empty_melds(table: Table) -> None:
        table.melds = [m for m in table.melds if m.slots]

    def _visible_hand_slots() -> List[TileSlot]:
        remaining = remaining_hand_after_edit(timeline.current, edited_table)
        return _slots_from_counts(remaining.counts)

    def _layout_table_tiles(start_x: int, start_y: int) -> List[Tuple[pygame.Rect, TileSlot, int, int]]:
        layout: List[Tuple[pygame.Rect, TileSlot, int, int]] = []
        x, y = start_x, start_y
        for meld_idx, meld in enumerate(edited_table.melds):
            header_rect = pygame.Rect(TABLE_START_X - 10, y - 20, 900, 18)
            pygame.draw.rect(screen, panel_color, header_rect)
            header_label = small_font.render(f"{meld.kind.value} #{meld_idx+1}", True, text_color)
            screen.blit(header_label, (header_rect.x + 4, header_rect.y + 1))
            for slot_idx, slot in enumerate(meld.slots):
                rect = pygame.Rect(x, y, TILE_W, TILE_H)
                _draw_tile(screen, rect, slot, highlight=selected_slot == (meld_idx, slot_idx))
                layout.append((rect, slot, meld_idx, slot_idx))
                x += TILE_W + 8
            x = start_x
            y += ROW_HEIGHT
        return layout

    def _layout_hand_tiles(
        slots: List[TileSlot],
        origin_y: int,
        per_row: int = 18,
        tile_w: int = TILE_W,
        tile_h: int = TILE_H,
        start_x: int = 30,
    ) -> List[Tuple[pygame.Rect, TileSlot]]:
        hand_layout: List[Tuple[pygame.Rect, TileSlot]] = []
        x, y = start_x, origin_y
        for idx, slot in enumerate(slots):
            row = idx // per_row
            col = idx % per_row
            rect = pygame.Rect(x + col * (tile_w + 6), y + row * (tile_h + 8), tile_w, tile_h)
            _draw_tile(screen, rect, slot)
            hand_layout.append((rect, slot))
        return hand_layout

    def _render_timeline_bar(rect: pygame.Rect) -> List[Tuple[pygame.Rect, int]]:
        pygame.draw.rect(screen, panel_color, rect, border_radius=6)
        segments: List[Tuple[pygame.Rect, int]] = []
        if len(timeline.history) == 1:
            pygame.draw.rect(screen, accent_color, rect.inflate(-4, -4), border_radius=6)
            segments.append((rect, 0))
            return segments
        seg_w = max(6, (rect.width - 20) // len(timeline.history))
        x = rect.x + 10
        for idx in range(len(timeline.history)):
            seg_rect = pygame.Rect(x, rect.y + 6, seg_w, rect.height - 12)
            color = accent_color if idx == timeline.index else (90, 94, 110)
            pygame.draw.rect(screen, color, seg_rect, border_radius=4)
            segments.append((seg_rect, idx))
            x += seg_w + 6
        return segments

    def _apply_move_and_refresh(move: Move, success_message: str) -> None:
        nonlocal edited_table, message
        next_state = apply_move(timeline.current, move)
        timeline.append(next_state)
        edited_table = next_state.table.canonicalize()
        message = success_message

    def _handle_drop(pos: Tuple[int, int], table_layout, hand_layout) -> None:
        nonlocal carried, carried_from, message, selected_slot
        if carried is None:
            return

        # Drop onto meld header
        for meld_idx, meld in enumerate(edited_table.melds):
            meld_rect = pygame.Rect(TABLE_START_X - 12, TABLE_START_Y + meld_idx * ROW_HEIGHT - 22, 1220, ROW_HEIGHT)
            if meld_rect.collidepoint(pos):
                edited_table.melds[meld_idx].slots.append(carried)
                _remove_empty_melds(edited_table)
                carried = None
                carried_from = None
                selected_slot = (meld_idx, len(edited_table.melds[meld_idx].slots) - 1)
                return

        # Drop on hand area => cancel drag
        hand_area = pygame.Rect(20, 520, 1240, 200)
        if hand_area.collidepoint(pos):
            if carried_from and carried_from[0] == "table":
                meld_idx, slot_idx = carried_from[1], carried_from[2]
                if meld_idx < len(edited_table.melds):
                    edited_table.melds[meld_idx].slots.insert(min(slot_idx, len(edited_table.melds[meld_idx].slots)), carried)
            carried = None
            carried_from = None
            return

        # Drop on empty space -> revert if from table, otherwise just cancel
        if carried_from and carried_from[0] == "table":
            meld_idx, slot_idx = carried_from[1], carried_from[2]
            if meld_idx < len(edited_table.melds):
                edited_table.melds[meld_idx].slots.insert(min(slot_idx, len(edited_table.melds[meld_idx].slots)), carried)
        carried = None
        carried_from = None

    running = True
    while running:
        screen.fill(bg_color)
        pygame.draw.rect(screen, panel_color, pygame.Rect(10, 10, 1260, 740), border_radius=8, width=2)

        table_title = font.render("Table (drag/drop to rearrange)", True, text_color)
        screen.blit(table_title, (20, 20))

        # Layout drawing pass also gives us hit boxes
        table_tiles = _layout_table_tiles(TABLE_START_X, TABLE_START_Y)
        hand_slots = _visible_hand_slots()
        hand_tiles = _layout_hand_tiles(hand_slots, 520)

        # Buttons (recomputed each frame for enabled state)
        buttons: List[Tuple[pygame.Rect, str, Callable[[], None], bool]] = []

        def add_button(x: int, y: int, label: str, handler: Callable[[], None], enabled: bool = True) -> None:
            rect = pygame.Rect(x, y, 150, 36)
            buttons.append((rect, label, handler, enabled))
            pygame.draw.rect(screen, panel_color if enabled else (32, 32, 40), rect, border_radius=6)
            pygame.draw.rect(screen, accent_color if enabled else disabled_color, rect, width=2, border_radius=6)
            lbl = font.render(label, True, text_color if enabled else disabled_color)
            screen.blit(lbl, lbl.get_rect(center=rect.center))

        can_play = timeline.at_end()
        add_button(520, 30, "Add RUN", lambda: edited_table.melds.append(Meld(MeldKind.RUN, [])), can_play)
        add_button(700, 30, "Add GROUP", lambda: edited_table.melds.append(Meld(MeldKind.GROUP, [])), can_play)

        def _reset_draft() -> None:
            nonlocal edited_table, carried, carried_from, selected_slot, message
            edited_table = timeline.current.table.canonicalize()
            carried = None
            carried_from = None
            selected_slot = None
            message = "Draft reset to current state"

        add_button(880, 30, "Reset draft", _reset_draft, can_play)

        def _on_validate() -> None:
            nonlocal message
            move, reason = build_play_move(timeline.current, edited_table)
            if move is None:
                message = f"Invalid move: {reason}"
                return
            _apply_move_and_refresh(move, "Move applied")

        add_button(1060, 30, "Validate PLAY", _on_validate, can_play)

        def _on_draw() -> None:
            nonlocal message
            move = Move.draw()
            legal, reason = is_legal_move(timeline.current, move)
            if not legal:
                message = reason
                return
            _apply_move_and_refresh(move, "Drew a tile")

        add_button(1060, 80, "Draw", _on_draw, can_play)

        def _on_pass() -> None:
            nonlocal message
            move = Move.skip()
            legal, reason = is_legal_move(timeline.current, move)
            if not legal:
                message = reason
                return
            _apply_move_and_refresh(move, "Passed turn")

        add_button(1060, 130, "Pass", _on_pass, can_play)

        timeline_rect = pygame.Rect(20, 470, 1240, 36)
        segments = _render_timeline_bar(timeline_rect)

        info_text = small_font.render(
            f"Player {timeline.current.current_player + 1} | Turn {timeline.current.turn_number} | Deck remaining: {len(timeline.current.deck_order) - timeline.current.deck_index}",
            True,
            text_color,
        )
        screen.blit(info_text, (20, 450))

        msg_surface = small_font.render(message, True, text_color)
        screen.blit(msg_surface, (20, 730))

        toggle_hint = small_font.render("Press 'G' to toggle godmode (show all hands & deck)", True, text_color)
        screen.blit(toggle_hint, (820, 450))

        if show_godmode:
            god_panel = pygame.Rect(520, 190, 740, 280)
            pygame.draw.rect(screen, panel_color, god_panel, border_radius=8)
            pygame.draw.rect(screen, accent_color, god_panel, width=2, border_radius=8)

            title = font.render("Godmode — full visibility", True, text_color)
            screen.blit(title, (god_panel.x + 12, god_panel.y + 8))

            y_cursor = god_panel.y + 40
            for idx, hand in enumerate(timeline.current.hands):
                label = small_font.render(
                    f"P{idx + 1} hand ({hand.total()} tiles)" + ("  ← current" if idx == timeline.current.current_player else ""),
                    True,
                    text_color,
                )
                screen.blit(label, (god_panel.x + 12, y_cursor))
                slots = _slots_from_counts(hand.counts)
                _layout_hand_tiles(slots, y_cursor + 16, per_row=15, tile_w=32, tile_h=44, start_x=god_panel.x + 12)
                rows = (len(slots) + 14) // 15
                y_cursor += 16 + rows * (44 + 8) + 6

            deck_remaining = timeline.current.deck_order[timeline.current.deck_index :]
            deck_label = small_font.render(f"Deck remaining: {len(deck_remaining)} (top shown)", True, text_color)
            screen.blit(deck_label, (god_panel.x + 12, y_cursor))
            deck_slots = _slots_from_tile_ids(deck_remaining[:45])
            _layout_hand_tiles(deck_slots, y_cursor + 16, per_row=15, tile_w=30, tile_h=42, start_x=god_panel.x + 12)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_g:
                    show_godmode = not show_godmode
                    message = "Godmode enabled" if show_godmode else "Godmode hidden"
                elif event.key == pygame.K_LEFT:
                    timeline.jump(timeline.index - 1)
                    edited_table = timeline.current.table.canonicalize()
                    selected_slot = None
                    message = ""
                elif event.key == pygame.K_RIGHT:
                    timeline.jump(timeline.index + 1)
                    edited_table = timeline.current.table.canonicalize()
                    selected_slot = None
                    message = ""
                elif event.key in (pygame.K_c, pygame.K_v):
                    if selected_slot and selected_slot[0] < len(edited_table.melds):
                        meld_idx, slot_idx = selected_slot
                        meld = edited_table.melds[meld_idx]
                        if 0 <= slot_idx < len(meld.slots):
                            slot = meld.slots[slot_idx]
                            if slot.tile_id == JOKER_ID:
                                new_color = slot.assigned_color or 0
                                new_value = slot.assigned_value or 1
                                if event.key == pygame.K_c:
                                    new_color = (new_color + 1) % timeline.current.ruleset.colors
                                else:
                                    new_value = (new_value % timeline.current.ruleset.values) + 1
                                meld.slots[slot_idx] = TileSlot(JOKER_ID, new_color, new_value)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Buttons
                    for rect, label, handler, enabled in buttons:
                        if enabled and rect.collidepoint(event.pos):
                            handler()
                    # Timeline jump
                    for seg_rect, idx in segments:
                        if seg_rect.collidepoint(event.pos):
                            timeline.jump(idx)
                            edited_table = timeline.current.table.canonicalize()
                            message = ""
                            break
                    # Drag start from table
                    if timeline.at_end():
                        for rect, slot, meld_idx, slot_idx in table_tiles:
                            if rect.collidepoint(event.pos):
                                carried = slot
                                carried_from = ("table", meld_idx, slot_idx)
                                if meld_idx < len(edited_table.melds):
                                    if slot_idx < len(edited_table.melds[meld_idx].slots):
                                        edited_table.melds[meld_idx].slots.pop(slot_idx)
                                        _remove_empty_melds(edited_table)
                                break
                        # Drag start from hand
                        if carried is None:
                            for rect, slot in hand_tiles:
                                if rect.collidepoint(event.pos):
                                    carried = slot
                                    carried_from = ("hand", -1, -1)
                                    break
                elif event.button == 3:
                    # Right click selects tile for assignment tweaks
                    for rect, slot, meld_idx, slot_idx in table_tiles:
                        if rect.collidepoint(event.pos):
                            selected_slot = (meld_idx, slot_idx)
                            break
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and carried is not None:
                    _handle_drop(event.pos, table_tiles, hand_tiles)
                    _remove_empty_melds(edited_table)

        if carried is not None:
            mouse_rect = pygame.Rect(pygame.mouse.get_pos()[0] - TILE_W // 2, pygame.mouse.get_pos()[1] - TILE_H // 2, TILE_W, TILE_H)
            _draw_tile(screen, mouse_rect, carried, highlight=True)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":  # pragma: no cover
    launch_gui()
