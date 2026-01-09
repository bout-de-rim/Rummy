from __future__ import annotations

"""Pygame GUI for Rummikub (Sprint 2).

This module contains two parts:
- pure helpers (`TimelineController`, `build_play_move`, ...), testable without
  pygame installed;
- the actual pygame UI (`launch_gui`), imported lazily to avoid making pygame a
  hard dependency for non-UI workflows.

Notes (UI):
- The GUI is intended to be used by humans *and* as a visualization/debugging
  surface during RL experimentation. As such, the UI includes an optional
  debug overlay ("Debug") and prioritizes deterministic rendering and
  input-to-state traceability.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import os

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

    non_empty = Table([meld for meld in edited_table.melds if meld.slots])
    canonical_table, error = _safe_canonical_table(non_empty)
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

    non_empty = Table([meld for meld in edited_table.melds if meld.slots])
    canonical_table, error = _safe_canonical_table(non_empty)
    if canonical_table is None:
        return state.hands[state.current_player].copy()

    delta, error = compute_delta_from_tables(state.table.canonicalize(), canonical_table)
    if delta is None:
        return state.hands[state.current_player].copy()

    try:
        return state.hands[state.current_player].sub(delta)
    except ValueError:
        return state.hands[state.current_player].copy()


def resolve_screenshot_path(env: dict | None = None) -> Optional[Path]:
    """Return a screenshot path if configured via environment variables."""

    env = env or os.environ
    raw = env.get("RUMMY_GUI_SCREENSHOT_PATH")
    if not raw:
        return None
    return Path(raw)


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
        raise ImportError("pygame is required for the GUI. Install it with `pip install pygame`.")

    ruleset = ruleset or Ruleset()
    timeline = TimelineController.from_state(new_game(ruleset=ruleset, rng_seed=seed))
    edited_table = timeline.current.table.canonicalize()

    pygame.init()
    pygame.display.set_caption("Rummikub — Sprint 2 GUI")
    screen = pygame.display.set_mode((1280, 768), pygame.RESIZABLE)
    clock = pygame.time.Clock()

    # Fonts
    font = pygame.font.SysFont("arial", 18)
    small_font = pygame.font.SysFont("arial", 14)
    title_font = pygame.font.SysFont("arial", 20, bold=True)

    # ---- UI state ------------------------------------------------------------
    carried: Optional[TileSlot] = None
    carried_from: Optional[Tuple[str, int, int]] = None  # ("table"/"hand", meld_idx, slot_idx)
    selected_slot: Optional[Tuple[int, int]] = None
    selected_empty_slot: Optional[Tuple[int, str]] = None  # (meld_idx, "before"/"after"/"only")

    show_debug = False
    debug_scroll = 0
    table_scroll = 0
    hand_scroll = 0

    modal_open = False
    modal_page = 0
    help_open = False

    hand_sort_mode = "color"  # "color" | "value"

    screenshot_path = resolve_screenshot_path()
    screenshot_taken = False

    # Toast/status
    toast_text = "Left click to drag tiles. Click a + slot then click a hand tile to add quickly. Press H for help."
    toast_kind = "info"  # info | success | warning | error
    toast_until_ms = 0

    def set_toast(text: str, kind: str = "info", ttl_ms: int = 4500) -> None:
        nonlocal toast_text, toast_kind, toast_until_ms
        toast_text = text
        toast_kind = kind
        toast_until_ms = pygame.time.get_ticks() + max(0, ttl_ms)

    # ---- Theme ---------------------------------------------------------------
    color_palette = [
        (214, 72, 72),
        (75, 139, 190),
        (40, 168, 112),
        (214, 156, 76),
    ]
    bg_color = (26, 29, 38)
    panel_color = (40, 46, 60)
    panel_color_2 = (33, 38, 50)
    text_color = (230, 232, 240)
    disabled_color = (110, 110, 120)
    accent_color = (109, 176, 255)
    warn_color = (255, 196, 72)
    error_color = (240, 90, 90)
    success_color = (120, 220, 140)

    # ---- Helpers -------------------------------------------------------------
    def _slots_from_counts(counts: List[int]) -> List[TileSlot]:
        slots: List[TileSlot] = []
        for tile_id, count in enumerate(counts):
            for _ in range(count):
                if tile_id == JOKER_ID:
                    slots.append(TileSlot(JOKER_ID, assigned_color=0, assigned_value=1))
                else:
                    slots.append(TileSlot(tile_id))
        # default ordering for readability; can be toggled
        if hand_sort_mode == "value":
            slots.sort(key=lambda s: (s.effective_value(), s.effective_color() if s.tile_id != JOKER_ID else s.assigned_color or 0))
        else:
            slots.sort(key=lambda s: (s.effective_color() if s.tile_id != JOKER_ID else s.assigned_color or 0, s.effective_value()))
        return slots

    def _slots_from_tile_ids(tile_ids: List[int], sort_slots: bool = False) -> List[TileSlot]:
        slots: List[TileSlot] = []
        for tile_id in tile_ids:
            if tile_id == JOKER_ID:
                slots.append(TileSlot(JOKER_ID, assigned_color=0, assigned_value=1))
            else:
                slots.append(TileSlot(tile_id))
        if sort_slots:
            slots.sort(key=lambda s: (s.effective_color() if s.tile_id != JOKER_ID else s.assigned_color or 0, s.effective_value()))
        return slots

    def _tile_label(slot: TileSlot) -> str:
        if slot.tile_id == JOKER_ID:
            suffix = "?" if slot.assigned_value is None else str(slot.assigned_value)
            return f"J{suffix}"
        return str((slot.tile_id % 13) + 1)

    def _tile_color(slot: TileSlot) -> Tuple[int, int, int]:
        if slot.tile_id == JOKER_ID:
            return (180, 92, 200)
        return color_palette[slot.tile_id // 13 % len(color_palette)]

    # Tile surface cache: (tile_id, assigned_color, assigned_value, w, h, highlight)
    tile_cache: dict[tuple, pygame.Surface] = {}

    def _draw_tile(surface: pygame.Surface, rect: pygame.Rect, slot: TileSlot, highlight: bool = False) -> None:
        key = (slot.tile_id, slot.assigned_color, slot.assigned_value, rect.width, rect.height, highlight)
        cached = tile_cache.get(key)
        if cached is None:
            s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            pygame.draw.rect(s, (245, 245, 245), pygame.Rect(0, 0, rect.width, rect.height), border_radius=8)
            pygame.draw.rect(s, accent_color if highlight else (60, 60, 70), pygame.Rect(0, 0, rect.width, rect.height), width=2, border_radius=8)
            inner = pygame.Rect(0, 0, rect.width, rect.height).inflate(-10, -10)
            pygame.draw.rect(s, _tile_color(slot), inner, border_radius=6)
            label = font.render(_tile_label(slot), True, (0, 0, 0))
            s.blit(label, label.get_rect(center=inner.center))
            cached = s
            tile_cache[key] = cached
        surface.blit(cached, rect.topleft)

    def _remove_empty_melds(table: Table) -> None:
        table.melds = [m for m in table.melds if m.slots]

    def _ensure_empty_meld(table: Table) -> None:
        empty_indices = [idx for idx, meld in enumerate(table.melds) if not meld.slots]
        if not empty_indices:
            table.melds.append(Meld(MeldKind.RUN, []))
        elif len(empty_indices) > 1:
            for idx in reversed(empty_indices[1:]):
                table.melds.pop(idx)

    def _meld_empty_slots(meld: Meld) -> List[str]:
        """Which ends can be extended (syntactic constraints only)."""
        if not meld.slots:
            return ["only"]
        if meld.kind == MeldKind.RUN:
            values = [slot.effective_value() for slot in meld.slots]
            min_val = min(values)
            max_val = max(values)
            slots: List[str] = []
            if min_val > 1:
                slots.append("before")
            if max_val < timeline.current.ruleset.values:
                slots.append("after")
            return slots
        if meld.kind == MeldKind.GROUP:
            return ["after"] if len(meld.slots) < 4 else []
        return []

    def _can_be_valid(kind: MeldKind, slots: List[TileSlot]) -> bool:
        if not slots:
            return True
        try:
            colors = [s.effective_color() for s in slots]
            values = [s.effective_value() for s in slots]
        except ValueError:
            return False
        if kind == MeldKind.GROUP:
            if len(slots) > 4:
                return False
            if len(set(values)) != 1:
                return False
            return len(set(colors)) == len(colors)
        if kind == MeldKind.RUN:
            if len(slots) > timeline.current.ruleset.values:
                return False
            if len(set(colors)) != 1:
                return False
            if len(set(values)) != len(values):
                return False
            min_val = min(values)
            max_val = max(values)
            if (max_val - min_val + 1) != len(values):
                return False
            return max_val <= timeline.current.ruleset.values and min_val >= 1
        return False

    def _choose_kind_for_slots(slots: List[TileSlot], preferred: Optional[MeldKind]) -> Optional[MeldKind]:
        possible = [kind for kind in (MeldKind.RUN, MeldKind.GROUP) if _can_be_valid(kind, slots)]
        if not possible:
            return None
        if preferred in possible:
            return preferred
        if len(possible) == 1:
            return possible[0]
        try:
            colors = [s.effective_color() for s in slots]
            values = [s.effective_value() for s in slots]
        except ValueError:
            return possible[0]
        if len(set(values)) == 1 and len(set(colors)) > 1:
            return MeldKind.GROUP
        if len(set(colors)) == 1:
            return MeldKind.RUN
        return possible[0]

    def _assign_joker_for_meld(slot: TileSlot, meld: Meld, to_left: bool, kind: MeldKind) -> Optional[TileSlot]:
        if slot.tile_id != JOKER_ID:
            return slot
        if kind == MeldKind.RUN:
            if meld.slots:
                color = meld.slots[0].effective_color()
                values = [s.effective_value() for s in meld.slots]
                new_value = min(values) - 1 if to_left else max(values) + 1
            else:
                color = 0
                new_value = 1
            return TileSlot(JOKER_ID, color, new_value)
        if kind == MeldKind.GROUP:
            if meld.slots:
                value = meld.slots[0].effective_value()
                used_colors = {s.effective_color() for s in meld.slots}
            else:
                value = 1
                used_colors = set()
            for color in range(timeline.current.ruleset.colors):
                if color not in used_colors:
                    return TileSlot(JOKER_ID, color, value)
            return None
        return None

    def _try_insert_into_meld(meld_idx: int, slot: TileSlot, position: str) -> bool:
        nonlocal selected_slot, selected_empty_slot
        if meld_idx < 0 or meld_idx >= len(edited_table.melds):
            return False
        meld = edited_table.melds[meld_idx]
        to_left = position == "before"
        preferred_kind = meld.kind if meld.slots else None
        for kind in (preferred_kind, MeldKind.RUN, MeldKind.GROUP):
            if kind is None:
                continue
            slot_to_use = _assign_joker_for_meld(slot, meld, to_left, kind)
            if slot_to_use is None:
                continue
            new_slots = list(meld.slots)
            if position == "only":
                to_left = False
            if to_left:
                new_slots.insert(0, slot_to_use)
                new_slot_idx = 0
            else:
                new_slots.append(slot_to_use)
                new_slot_idx = len(new_slots) - 1
            chosen_kind = _choose_kind_for_slots(new_slots, preferred=kind)
            if chosen_kind is None:
                continue
            if not _can_be_valid(chosen_kind, new_slots):
                continue
            edited_table.melds[meld_idx] = Meld(chosen_kind, new_slots)
            selected_slot = (meld_idx, new_slot_idx)
            available = _meld_empty_slots(edited_table.melds[meld_idx])
            if available:
                preferred_end = "before" if position in ("before", "only") else "after"
                selected_empty_slot = (meld_idx, preferred_end if preferred_end in available else available[0])
            return True
        return False

    def _visible_hand_slots() -> List[TileSlot]:
        remaining = remaining_hand_after_edit(timeline.current, edited_table)
        return _slots_from_counts(remaining.counts)

    def _compute_draft_delta() -> Tuple[Optional[TileMultiset], str, Optional[Table]]:
        """Compute delta_from_hand for the edited table. Returns (delta, error, canonical_edited)."""
        base = timeline.current.table.canonicalize()
        non_empty = Table([meld for meld in edited_table.melds if meld.slots])
        canonical_edited, err = _safe_canonical_table(non_empty)
        if canonical_edited is None:
            return None, err, None
        delta, err = compute_delta_from_tables(base, canonical_edited)
        return delta, err, canonical_edited

    def _build_added_tile_counts() -> List[int]:
        delta, _, _ = _compute_draft_delta()
        if delta is None:
            return [0] * len(timeline.current.hands[0].counts)
        return list(delta.counts)

    def _is_draft_active() -> bool:
        if not timeline.at_end():
            return False
        return any(_build_added_tile_counts()) or any(_mark_modified_melds())

    def _can_remove_from_hand(slot: TileSlot) -> bool:
        counts = _build_added_tile_counts()
        return counts[slot.tile_id] > 0

    def _mark_modified_melds() -> List[bool]:
        from collections import Counter

        base_melds = timeline.current.table.canonicalize().melds
        base_counter = Counter(m.effective_signature() for m in base_melds)
        modified_flags: List[bool] = []
        for meld in edited_table.melds:
            if not meld.slots:
                modified_flags.append(True)
                continue
            sig = meld.effective_signature()
            if base_counter[sig] > 0:
                base_counter[sig] -= 1
                modified_flags.append(False)
            else:
                modified_flags.append(True)
        return modified_flags

    # ---- Layout primitives ---------------------------------------------------
    @dataclass
    class Button:
        rect: pygame.Rect
        label: str
        handler: Callable[[], None]
        enabled: bool = True
        shortcut: str = ""
        tooltip: str = ""

    def _draw_button(btn: Button, mouse_pos: Tuple[int, int]) -> None:
        hovered = btn.rect.collidepoint(mouse_pos)
        base = panel_color if btn.enabled else (32, 32, 40)
        border = accent_color if btn.enabled else disabled_color
        if hovered and btn.enabled:
            base = (base[0] + 10, base[1] + 10, base[2] + 10)
        pygame.draw.rect(screen, base, btn.rect, border_radius=8)
        pygame.draw.rect(screen, border, btn.rect, width=2, border_radius=8)
        label = btn.label + (f" ({btn.shortcut})" if btn.shortcut else "")
        lbl = font.render(label, True, text_color if btn.enabled else disabled_color)
        screen.blit(lbl, lbl.get_rect(center=btn.rect.center))

    def _draw_tooltip(text: str, anchor: Tuple[int, int]) -> None:
        if not text:
            return
        pad = 8
        lines = text.split("\n")
        surfaces = [small_font.render(line, True, text_color) for line in lines]
        w = max(s.get_width() for s in surfaces) + pad * 2
        h = sum(s.get_height() for s in surfaces) + pad * 2
        x, y = anchor
        x = min(x + 14, screen.get_width() - w - 10)
        y = min(y + 14, screen.get_height() - h - 10)
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(screen, panel_color, rect, border_radius=8)
        pygame.draw.rect(screen, accent_color, rect, width=2, border_radius=8)
        y_cursor = rect.y + pad
        for s in surfaces:
            screen.blit(s, (rect.x + pad, y_cursor))
            y_cursor += s.get_height()

    def _draw_badge(text: str, rect: pygame.Rect, color: Tuple[int, int, int]) -> None:
        pygame.draw.rect(screen, color, rect, border_radius=10)
        lbl = small_font.render(text, True, (20, 20, 20))
        screen.blit(lbl, lbl.get_rect(center=rect.center))

    def _layout_hand_tiles(
        slots: List[TileSlot],
        origin: pygame.Rect,
        tile_w: int,
        tile_h: int,
        gap_x: int,
        gap_y: int,
        y_scroll: int,
    ) -> Tuple[List[Tuple[pygame.Rect, TileSlot]], int]:
        """Layout & render the hand area with clipping and vertical scrolling.

        Returns (hand_hitboxes, content_height_px).
        """
        hand_layout: List[Tuple[pygame.Rect, TileSlot]] = []

        # Content viewport (below the "Hand" header line)
        content_rect = pygame.Rect(origin.x + 10, origin.y + 34, origin.width - 20, origin.height - 44)
        content_rect.height = max(0, content_rect.height)

        per_row = max(1, (content_rect.width + gap_x) // (tile_w + gap_x))
        x0, y0 = content_rect.x, content_rect.y - y_scroll

        rows = (len(slots) + per_row - 1) // per_row if slots else 1
        content_h = rows * tile_h + max(0, rows - 1) * gap_y

        # Clip rendering to avoid drawing outside the panel (important on small windows)
        screen.set_clip(content_rect)

        for idx, slot in enumerate(slots):
            row = idx // per_row
            col = idx % per_row
            rect = pygame.Rect(x0 + col * (tile_w + gap_x), y0 + row * (tile_h + gap_y), tile_w, tile_h)
            _draw_tile(screen, rect, slot)
            hand_layout.append((rect, slot))

        screen.set_clip(None)
        return hand_layout, content_h

    def _layout_table_tiles(
        area: pygame.Rect,
        tile_w: int,
        tile_h: int,
        gap_x: int,
        gap_y: int,
        added_counts: List[int],
        modified_flags: List[bool],
        y_scroll: int,
    ) -> Tuple[
        List[Tuple[pygame.Rect, TileSlot, int, int]],
        List[Tuple[pygame.Rect, int, str]],
        List[pygame.Rect],
        int,
    ]:
        """Layout & render melds.

        Returns (tile_hitboxes, empty_slot_hitboxes, meld_header_hitboxes, content_height).
        """
        tile_hit: List[Tuple[pygame.Rect, TileSlot, int, int]] = []
        empty_hit: List[Tuple[pygame.Rect, int, str]] = []
        meld_headers: List[pygame.Rect] = []

        # table background
        pygame.draw.rect(screen, panel_color_2, area, border_radius=10)
        pygame.draw.rect(screen, panel_color, area, width=2, border_radius=10)

        x0 = area.x + 12
        y = area.y + 12 - y_scroll
        max_cols = max(1, (area.width - 24) // (tile_w + gap_x))

        # clip table scroll region
        screen.set_clip(area)

        for meld_idx, meld in enumerate(edited_table.melds):
            is_modified = modified_flags[meld_idx] if meld_idx < len(modified_flags) else True
            is_valid = _can_be_valid(meld.kind, meld.slots)

            header_h = 22
            header_rect = pygame.Rect(x0 - 6, y, area.width - 24 + 12, header_h)
            meld_headers.append(header_rect)

            header_bg = (63, 78, 112) if is_modified else panel_color
            if meld.slots and not is_valid:
                header_bg = (92, 52, 60)

            pygame.draw.rect(screen, header_bg, header_rect, border_radius=6)
            kind_label = meld.kind.value if meld.slots else "MELD"
            suffix = "" if is_valid else "  !"
            hdr = small_font.render(f"{kind_label} #{meld_idx + 1}{suffix}", True, text_color)
            screen.blit(hdr, (header_rect.x + 8, header_rect.y + 3))

            y_tiles = y + header_h + 8
            available = _meld_empty_slots(meld)

            # tiles with wrapping
            for slot_idx, slot in enumerate(meld.slots):
                row = slot_idx // max_cols
                col = slot_idx % max_cols
                rect = pygame.Rect(x0 + col * (tile_w + gap_x), y_tiles + row * (tile_h + gap_y), tile_w, tile_h)

                from_hand = False
                if added_counts[slot.tile_id] > 0:
                    added_counts[slot.tile_id] -= 1
                    from_hand = True

                _draw_tile(screen, rect, slot, highlight=selected_slot == (meld_idx, slot_idx))
                if from_hand:
                    pygame.draw.rect(screen, warn_color, rect, width=3, border_radius=8)

                tile_hit.append((rect, slot, meld_idx, slot_idx))

            # empty extension slots (visual affordance)
            if meld.slots:
                last_idx = max(0, len(meld.slots) - 1)
                last_row = last_idx // max_cols
                last_col = last_idx % max_cols
            else:
                last_row = 0
                last_col = -1

            if "before" in available:
                slot_rect = pygame.Rect(x0 - (tile_w + gap_x), y_tiles, tile_w, tile_h)
                empty_hit.append((slot_rect, meld_idx, "before"))
            if "only" in available:
                slot_rect = pygame.Rect(x0, y_tiles, tile_w, tile_h)
                empty_hit.append((slot_rect, meld_idx, "only"))
            if "after" in available:
                slot_rect = pygame.Rect(x0 + (last_col + 1) * (tile_w + gap_x), y_tiles + last_row * (tile_h + gap_y), tile_w, tile_h)
                empty_hit.append((slot_rect, meld_idx, "after"))

            # compute meld block height
            rows = max(1, (len(meld.slots) + max_cols - 1) // max_cols)
            block_h = header_h + 8 + rows * tile_h + (rows - 1) * gap_y + 14

            # subtle divider line between melds
            divider_y = y + block_h - 4
            pygame.draw.line(screen, (52, 58, 74), (area.x + 12, divider_y), (area.right - 12, divider_y), 1)

            y += block_h

        # render empty slots on top
        for rect, meld_idx, position in empty_hit:
            is_selected = selected_empty_slot == (meld_idx, position)
            border = accent_color if is_selected else (120, 126, 138)
            pygame.draw.rect(screen, border, rect, width=2, border_radius=8)
            ghost = rect.inflate(-10, -10)
            pygame.draw.rect(screen, (46, 50, 60), ghost, border_radius=6)
            plus = title_font.render("+", True, text_color)
            screen.blit(plus, plus.get_rect(center=ghost.center))

        screen.set_clip(None)

        # content height for scrolling
        content_h = (y + y_scroll) - (area.y + 12)
        return tile_hit, empty_hit, meld_headers, content_h

    def _render_timeline(rect: pygame.Rect) -> Tuple[List[Tuple[pygame.Rect, int]], pygame.Rect, pygame.Rect, pygame.Rect]:
        """Render a compact timeline with prev/next/history buttons."""
        pygame.draw.rect(screen, panel_color_2, rect, border_radius=10)
        pygame.draw.rect(screen, panel_color, rect, width=2, border_radius=10)

        prev_rect = pygame.Rect(rect.x + 10, rect.y + 6, 44, rect.height - 12)
        next_rect = pygame.Rect(prev_rect.right + 6, rect.y + 6, 44, rect.height - 12)
        hist_rect = pygame.Rect(next_rect.right + 10, rect.y + 6, 90, rect.height - 12)

        # segment bar
        bar_rect = pygame.Rect(hist_rect.right + 10, rect.y + 6, rect.width - (hist_rect.right - rect.x) - 20, rect.height - 12)

        pygame.draw.rect(screen, panel_color, prev_rect, border_radius=8)
        pygame.draw.rect(screen, accent_color, prev_rect, width=2, border_radius=8)
        pygame.draw.rect(screen, panel_color, next_rect, border_radius=8)
        pygame.draw.rect(screen, accent_color, next_rect, width=2, border_radius=8)
        pygame.draw.rect(screen, panel_color, hist_rect, border_radius=8)
        pygame.draw.rect(screen, accent_color, hist_rect, width=2, border_radius=8)

        screen.blit(font.render("<", True, text_color), font.render("<", True, text_color).get_rect(center=prev_rect.center))
        screen.blit(font.render(">", True, text_color), font.render(">", True, text_color).get_rect(center=next_rect.center))
        screen.blit(small_font.render("History", True, text_color), small_font.render("History", True, text_color).get_rect(center=hist_rect.center))

        segments: List[Tuple[pygame.Rect, int]] = []
        total = len(timeline.history)
        visible_count = 14
        gap = 4
        if total <= 1:
            # single pill
            pill = bar_rect.inflate(-4, -4)
            pygame.draw.rect(screen, (90, 94, 110), pill, border_radius=8)
            pygame.draw.rect(screen, accent_color, pill, width=2, border_radius=8)
            segments.append((pill, 0))
            label = small_font.render("0", True, text_color)
            screen.blit(label, label.get_rect(center=pill.center))
            return segments, prev_rect, next_rect, hist_rect

        # show last N steps but keep current in view
        start = max(0, min(timeline.index - visible_count // 2, total - visible_count))
        end = min(total, start + visible_count)
        count = end - start
        seg_w = max(8, (bar_rect.width - gap * (count - 1)) // max(1, count))
        x = bar_rect.x
        for idx in range(start, end):
            r = pygame.Rect(x, bar_rect.y, seg_w, bar_rect.height)
            color = accent_color if idx == timeline.index else (90, 94, 110)
            pygame.draw.rect(screen, color, r, border_radius=6)
            segments.append((r, idx))
            x += seg_w + gap

        # compact index label
        label = small_font.render(f"{timeline.index}/{total - 1}", True, text_color)
        screen.blit(label, (bar_rect.right - label.get_width(), rect.y - 16))
        return segments, prev_rect, next_rect, hist_rect

    def _apply_move_and_refresh(move: Move, success_message: str) -> None:
        nonlocal edited_table, selected_empty_slot, selected_slot, table_scroll, hand_scroll
        next_state = apply_move(timeline.current, move)
        timeline.append(next_state)
        edited_table = next_state.table.canonicalize()
        selected_empty_slot = None
        selected_slot = None
        table_scroll = 0
        hand_scroll = 0
        set_toast(success_message, kind="success", ttl_ms=2500)

    def _reset_draft() -> None:
        nonlocal edited_table, carried, carried_from, selected_slot, selected_empty_slot, table_scroll, hand_scroll
        edited_table = timeline.current.table.canonicalize()
        carried = None
        carried_from = None
        selected_slot = None
        selected_empty_slot = None
        table_scroll = 0
        hand_scroll = 0
        set_toast("Draft reset to current state", kind="info")

    def _add_meld() -> None:
        edited_table.melds.append(Meld(MeldKind.RUN, []))
        _ensure_empty_meld(edited_table)
        set_toast("Added an empty meld", kind="info", ttl_ms=2000)

    def _on_play() -> None:
        move, reason = build_play_move(timeline.current, edited_table)
        if move is None:
            set_toast(f"Invalid move: {reason}", kind="error")
            return
        _apply_move_and_refresh(move, "Move applied")

    def _on_draw() -> None:
        move = Move.draw()
        legal, reason = is_legal_move(timeline.current, move)
        if not legal:
            set_toast(reason, kind="error")
            return
        _apply_move_and_refresh(move, "Drew a tile")

    def _on_pass() -> None:
        move = Move.skip()
        legal, reason = is_legal_move(timeline.current, move)
        if not legal:
            set_toast(reason, kind="error")
            return
        _apply_move_and_refresh(move, "Passed turn")

    def _on_toggle_debug() -> None:
        nonlocal show_debug, debug_scroll
        show_debug = not show_debug
        debug_scroll = 0
        set_toast("Debug overlay enabled" if show_debug else "Debug overlay hidden", kind="info", ttl_ms=2000)

    def _on_toggle_help() -> None:
        nonlocal help_open
        help_open = not help_open

    def _on_toggle_sort() -> None:
        nonlocal hand_sort_mode
        hand_sort_mode = "value" if hand_sort_mode == "color" else "color"
        set_toast(f"Hand sort: {hand_sort_mode}", kind="info", ttl_ms=2000)

    def _open_history() -> None:
        nonlocal modal_open, modal_page
        modal_open = True
        modal_page = 0

    def _handle_drop(pos: Tuple[int, int], hand_area: pygame.Rect, tile_hitboxes: List[Tuple[pygame.Rect, TileSlot, int, int]]) -> None:
        nonlocal carried, carried_from, selected_slot
        if carried is None:
            return

        # Drop onto table: if dropped over a specific tile in a meld, insert before it.
        inserted = False
        if timeline.at_end():
            for rect, _, meld_idx, slot_idx in tile_hitboxes:
                if rect.collidepoint(pos):
                    if meld_idx < len(edited_table.melds):
                        edited_table.melds[meld_idx].slots.insert(slot_idx, carried)
                        selected_slot = (meld_idx, slot_idx)
                        inserted = True
                    break

        # Drop onto hand area => keep removed tile in hand (do not reinsert)
        if hand_area.collidepoint(pos):
            carried = None
            carried_from = None
            return

        if inserted:
            carried = None
            carried_from = None
            return

        # Drop on empty space -> revert if from table, otherwise cancel
        if carried_from and carried_from[0] == "table":
            meld_idx, slot_idx = carried_from[1], carried_from[2]
            if meld_idx < len(edited_table.melds):
                edited_table.melds[meld_idx].slots.insert(min(slot_idx, len(edited_table.melds[meld_idx].slots)), carried)
        carried = None
        carried_from = None

    # ---- Main loop -----------------------------------------------------------
    running = True
    while running:
        screen.fill(bg_color)
        screen_w, screen_h = screen.get_size()

        # Frame border
        pygame.draw.rect(screen, panel_color, pygame.Rect(10, 10, screen_w - 20, screen_h - 20), border_radius=12, width=2)

        # Responsive layout
        PAD = 16

        # Toolbar becomes multi-row on narrow windows to avoid button overlap.
        TITLE_ROW_H = 44
        BTN_ROW_H = 44
        if screen_w >= 1050:
            toolbar_rows = 1
        elif screen_w >= 760:
            toolbar_rows = 2
        else:
            toolbar_rows = 3
        TOOLBAR_H = TITLE_ROW_H + (toolbar_rows - 1) * BTN_ROW_H

        STATUS_H = 30
        TIMELINE_H = 44
        HAND_H = max(180, int(screen_h * 0.32))

        status_rect = pygame.Rect(PAD, screen_h - STATUS_H - PAD, screen_w - 2 * PAD, STATUS_H)
        hand_rect = pygame.Rect(PAD, status_rect.y - HAND_H - PAD, screen_w - 2 * PAD, HAND_H)
        timeline_rect = pygame.Rect(PAD, hand_rect.y - TIMELINE_H - PAD, screen_w - 2 * PAD, TIMELINE_H)
        toolbar_rect = pygame.Rect(PAD, PAD, screen_w - 2 * PAD, TOOLBAR_H)
        table_rect = pygame.Rect(PAD, toolbar_rect.bottom + PAD, screen_w - 2 * PAD, max(120, timeline_rect.y - (toolbar_rect.bottom + 2 * PAD)))

        # Tile sizing tuned for human readability
        tile_h = max(44, min(78, int(hand_rect.height / 2.55)))
        tile_w = max(30, int(tile_h * 0.72))
        gap_x = max(6, tile_w // 8)
        gap_y = max(8, tile_h // 6)

        # Invalidate tile cache if sizes changed
        if any(k[3] != tile_w or k[4] != tile_h for k in list(tile_cache.keys())[:1]):
            tile_cache.clear()

        # Ensure at least one empty meld exists for quick starting
        _ensure_empty_meld(edited_table)

        # Draft flags
        added_counts = _build_added_tile_counts()
        modified_flags = _mark_modified_melds()
        # Toolbar
        pygame.draw.rect(screen, panel_color_2, toolbar_rect, border_radius=12)
        pygame.draw.rect(screen, panel_color, toolbar_rect, width=2, border_radius=12)

        can_play = timeline.at_end()

        # --- Title row -----------------------------------------------------
        title = title_font.render("Rummikub", True, text_color)
        screen.blit(title, (toolbar_rect.x + 14, toolbar_rect.y + 10))

        # Turn / current player indicator (prominent)
        player_idx = timeline.current.current_player
        player_colors = [accent_color, warn_color, success_color, error_color]
        pcol = player_colors[player_idx % len(player_colors)]
        pill_text = small_font.render(f"Tour: Joueur {player_idx + 1}", True, pcol)
        pill_w = pill_text.get_width() + 20
        pill_rect = pygame.Rect(toolbar_rect.x + 14 + title.get_width() + 12, toolbar_rect.y + 12, pill_w, 22)
        pygame.draw.rect(screen, panel_color, pill_rect, border_radius=10)
        pygame.draw.rect(screen, pcol, pill_rect, width=2, border_radius=10)
        screen.blit(pill_text, pill_text.get_rect(center=pill_rect.center))

        deck_remaining = len(timeline.current.deck_order) - timeline.current.deck_index
        hdr_info = small_font.render(
            f"Tour {timeline.current.turn_number}   Deck {deck_remaining}",
            True,
            text_color,
        )
        screen.blit(hdr_info, (toolbar_rect.right - hdr_info.get_width() - 14, toolbar_rect.y + 14))

        # --- Buttons -------------------------------------------------------
        buttons: List[Button] = []
        mouse_pos = pygame.mouse.get_pos()
        hover_tooltip = ""

        btn_h = 36
        btn_min_w = 120 if toolbar_rows == 1 else (92 if toolbar_rows == 2 else 84)

        def add_btn_flow(x: int, y: int, label: str, handler: Callable[[], None], enabled: bool = True, shortcut: str = "", tooltip: str = "") -> int:
            w = max(btn_min_w, font.size(label)[0] + 26 + (22 if shortcut else 0))
            rect = pygame.Rect(x, y, w, btn_h)
            buttons.append(Button(rect, label, handler, enabled, shortcut=shortcut, tooltip=tooltip))
            return rect.right + 10

        def add_btn_right(x_right: int, y: int, label: str, handler: Callable[[], None], enabled: bool = True, shortcut: str = "", tooltip: str = "") -> int:
            w = max(btn_min_w, font.size(label)[0] + 26 + (22 if shortcut else 0))
            rect = pygame.Rect(x_right - w, y, w, btn_h)
            buttons.append(Button(rect, label, handler, enabled, shortcut=shortcut, tooltip=tooltip))
            return rect.x - 10

        # Labels become shorter as width decreases
        sort_label = f"Tri: {hand_sort_mode}" if toolbar_rows == 1 else "Tri"
        reset_label = "Reset draft" if toolbar_rows == 1 else "Reset"
        meld_label = "+ Meld" if toolbar_rows >= 2 else "+ Meld"

        if toolbar_rows == 1:
            by = toolbar_rect.y + 10
            bx = toolbar_rect.x + 240
            rx = toolbar_rect.right - 14

            # Right cluster
            rx = add_btn_right(rx, by, "Help", _on_toggle_help, enabled=True, shortcut="H", tooltip="Show key bindings and interaction tips.")
            rx = add_btn_right(rx, by, "Debug", _on_toggle_debug, enabled=True, shortcut="G", tooltip="Toggle full-visibility overlay (hands + deck).")
            rx = add_btn_right(rx, by, "Pass", _on_pass, enabled=can_play, shortcut="P", tooltip="Skip turn.")
            rx = add_btn_right(rx, by, "Draw", _on_draw, enabled=can_play, shortcut="D", tooltip="Draw one tile.")

            # Left cluster
            bx = add_btn_flow(bx, by, meld_label, _add_meld, enabled=can_play, shortcut="M", tooltip="Add an empty meld row.")
            bx = add_btn_flow(bx, by, reset_label, _reset_draft, enabled=can_play and _is_draft_active(), shortcut="R", tooltip="Restore table to the current timeline state.")
            bx = add_btn_flow(bx, by, sort_label, _on_toggle_sort, enabled=True, shortcut="T", tooltip="Toggle hand ordering (by color or by value).")

            # Play (center)
            play_rect = pygame.Rect(toolbar_rect.centerx - 80, by, 160, btn_h)
            buttons.append(Button(play_rect, "Play", _on_play, enabled=can_play, shortcut="Enter", tooltip="Validate the draft and commit as a PLAY move."))

        elif toolbar_rows == 2:
            by = toolbar_rect.y + TITLE_ROW_H + 6
            bx = toolbar_rect.x + 14
            rx = toolbar_rect.right - 14

            # Right cluster
            rx = add_btn_right(rx, by, "Help", _on_toggle_help, enabled=True, shortcut="H", tooltip="Show key bindings and interaction tips.")
            rx = add_btn_right(rx, by, "Debug", _on_toggle_debug, enabled=True, shortcut="G", tooltip="Toggle full-visibility overlay (hands + deck).")
            rx = add_btn_right(rx, by, "Pass", _on_pass, enabled=can_play, shortcut="P", tooltip="Skip turn.")
            rx = add_btn_right(rx, by, "Draw", _on_draw, enabled=can_play, shortcut="D", tooltip="Draw one tile.")

            # Left cluster
            bx = add_btn_flow(bx, by, meld_label, _add_meld, enabled=can_play, shortcut="M", tooltip="Add an empty meld row.")
            bx = add_btn_flow(bx, by, reset_label, _reset_draft, enabled=can_play and _is_draft_active(), shortcut="R", tooltip="Restore table to the current timeline state.")
            bx = add_btn_flow(bx, by, sort_label, _on_toggle_sort, enabled=True, shortcut="T", tooltip="Toggle hand ordering (by color or by value).")

            # Play between clusters (auto-shrinks)
            available = max(0, rx - bx - 10)
            play_w = min(240, max(140, available))
            play_x = bx + max(0, (available - play_w) // 2)
            play_rect = pygame.Rect(play_x, by, play_w, btn_h)
            buttons.append(Button(play_rect, "Play", _on_play, enabled=can_play, shortcut="Enter", tooltip="Validate the draft and commit as a PLAY move."))

        else:  # toolbar_rows == 3
            by_primary = toolbar_rect.y + TITLE_ROW_H + 6
            by_secondary = by_primary + BTN_ROW_H

            # Primary row: Play + Draw/Pass
            bx = toolbar_rect.x + 14
            rx = toolbar_rect.right - 14
            rx = add_btn_right(rx, by_primary, "Pass", _on_pass, enabled=can_play, shortcut="P", tooltip="Skip turn.")
            rx = add_btn_right(rx, by_primary, "Draw", _on_draw, enabled=can_play, shortcut="D", tooltip="Draw one tile.")

            available = max(0, rx - bx - 10)
            play_w = min(320, max(160, available))
            play_x = bx + max(0, (available - play_w) // 2)
            play_rect = pygame.Rect(play_x, by_primary, play_w, btn_h)
            buttons.append(Button(play_rect, "Play", _on_play, enabled=can_play, shortcut="Enter", tooltip="Validate the draft and commit as a PLAY move."))

            # Secondary row: draft utilities + overlays
            bx = toolbar_rect.x + 14
            rx = toolbar_rect.right - 14
            rx = add_btn_right(rx, by_secondary, "Help", _on_toggle_help, enabled=True, shortcut="H", tooltip="Show key bindings and interaction tips.")
            rx = add_btn_right(rx, by_secondary, "Debug", _on_toggle_debug, enabled=True, shortcut="G", tooltip="Toggle full-visibility overlay (hands + deck).")

            bx = add_btn_flow(bx, by_secondary, meld_label, _add_meld, enabled=can_play, shortcut="M", tooltip="Add an empty meld row.")
            bx = add_btn_flow(bx, by_secondary, reset_label, _reset_draft, enabled=can_play and _is_draft_active(), shortcut="R", tooltip="Restore table to the current timeline state.")
            bx = add_btn_flow(bx, by_secondary, sort_label, _on_toggle_sort, enabled=True, shortcut="T", tooltip="Toggle hand ordering (by color or by value).")

        for b in buttons:
            _draw_button(b, mouse_pos)
            if b.enabled and b.rect.collidepoint(mouse_pos):
                hover_tooltip = b.tooltip
        # Draft badge & history indicator (placed after the player pill)
        badge_x = min(toolbar_rect.right - 220, pill_rect.right + 10)
        badge_y = pill_rect.y
        if _is_draft_active():
            _draw_badge("Draft", pygame.Rect(badge_x, badge_y, 64, 22), warn_color)
            badge_x += 74
        if not timeline.at_end():
            _draw_badge("History", pygame.Rect(badge_x, badge_y, 82, 22), (180, 180, 190))

        # Timeline
        segments, prev_rect, next_rect, hist_rect = _render_timeline(timeline_rect)

        # Table
        table_tiles, empty_slots, meld_headers, content_h = _layout_table_tiles(
            table_rect,
            tile_w=tile_w,
            tile_h=tile_h,
            gap_x=gap_x,
            gap_y=gap_y,
            added_counts=added_counts,
            modified_flags=modified_flags,
            y_scroll=table_scroll,
        )

        # Empty-table affordance
        if not any(m.slots for m in edited_table.melds):
            hint = small_font.render("Table vide : cliquez sur un + puis sélectionnez une tuile (ou glissez-déposez).", True, disabled_color)
            screen.blit(hint, hint.get_rect(center=table_rect.center))

        # Table scroll bar (if needed)
        max_scroll = max(0, content_h - (table_rect.height - 24))
        table_scroll = max(0, min(table_scroll, max_scroll))
        if max_scroll > 0:
            track = pygame.Rect(table_rect.right - 10, table_rect.y + 10, 6, table_rect.height - 20)
            pygame.draw.rect(screen, (60, 65, 80), track, border_radius=4)
            thumb_h = max(24, int(track.height * (table_rect.height / max(content_h, 1))))
            thumb_y = int(track.y + (track.height - thumb_h) * (table_scroll / max_scroll))
            thumb = pygame.Rect(track.x, thumb_y, track.width, thumb_h)
            pygame.draw.rect(screen, accent_color, thumb, border_radius=4)

        # Hand panel
        pygame.draw.rect(screen, panel_color_2, hand_rect, border_radius=12)
        pygame.draw.rect(screen, panel_color, hand_rect, width=2, border_radius=12)
        hand_title = title_font.render("Hand", True, text_color)
        screen.blit(hand_title, (hand_rect.x + 12, hand_rect.y + 8))

        # Guidance when an empty slot is selected
        if selected_empty_slot and can_play:
            hint = small_font.render("Selected + slot: click a hand tile to add", True, warn_color)
            screen.blit(hint, (hand_rect.x + 100, hand_rect.y + 14))

        hand_slots = _visible_hand_slots()
        hand_count = sum(1 for _ in hand_slots)
        hand_meta = small_font.render(f"{hand_count} tiles", True, text_color)
        screen.blit(hand_meta, (hand_rect.right - hand_meta.get_width() - 12, hand_rect.y + 14))

        hand_tiles, hand_content_h = _layout_hand_tiles(
            hand_slots,
            hand_rect,
            tile_w=tile_w,
            tile_h=tile_h,
            gap_x=gap_x,
            gap_y=gap_y,
            y_scroll=hand_scroll,
        )

        
        # Hand scroll management
        hand_view = pygame.Rect(hand_rect.x + 10, hand_rect.y + 34, hand_rect.width - 20, hand_rect.height - 44)
        hand_view.height = max(0, hand_view.height)
        max_hand_scroll = max(0, hand_content_h - hand_view.height)
        hand_scroll = max(0, min(hand_scroll, max_hand_scroll))
        if max_hand_scroll > 0 and hand_view.height > 0:
            track = pygame.Rect(hand_rect.right - 10, hand_view.y, 6, hand_view.height)
            pygame.draw.rect(screen, (60, 65, 80), track, border_radius=4)
            thumb_h = max(24, int(track.height * (hand_view.height / max(hand_content_h, 1))))
            thumb_y = int(track.y + (track.height - thumb_h) * (hand_scroll / max_hand_scroll))
            thumb = pygame.Rect(track.x, thumb_y, track.width, thumb_h)
            pygame.draw.rect(screen, accent_color, thumb, border_radius=4)

# If dragging, highlight drop targets
        if carried is not None:
            pygame.draw.rect(screen, accent_color, hand_rect, width=2, border_radius=12)
            pygame.draw.rect(screen, accent_color, table_rect, width=2, border_radius=12)

        # Debug overlay
        if show_debug:
            debug_rect = pygame.Rect(int(screen_w * 0.42), int(screen_h * 0.18), int(screen_w * 0.56), int(screen_h * 0.56))
            pygame.draw.rect(screen, panel_color, debug_rect, border_radius=12)
            pygame.draw.rect(screen, accent_color, debug_rect, width=2, border_radius=12)

            title = title_font.render("Debug overlay (full visibility)", True, text_color)
            screen.blit(title, (debug_rect.x + 14, debug_rect.y + 10))
            hint = small_font.render("Scroll: wheel / PgUp PgDn", True, text_color)
            screen.blit(hint, (debug_rect.x + 14, debug_rect.y + 34))

            clip = debug_rect.inflate(-16, -16)
            clip.y += 54
            clip.height -= 54
            screen.set_clip(clip)

            y_cursor = debug_rect.y + 62 - debug_scroll
            for idx, hand in enumerate(timeline.current.hands):
                lbl = small_font.render(
                    f"P{idx + 1} hand ({hand.total()} tiles)" + ("  ← current" if idx == timeline.current.current_player else ""),
                    True,
                    text_color,
                )
                screen.blit(lbl, (debug_rect.x + 14, y_cursor))
                slots = _slots_from_counts(hand.counts)
                # compact tiles
                temp_tile_h = 44
                temp_tile_w = 30
                x0, y0 = debug_rect.x + 14, y_cursor + 16
                per_row = 15
                for j, slot in enumerate(slots[:120]):
                    row = j // per_row
                    col = j % per_row
                    rect = pygame.Rect(x0 + col * (temp_tile_w + 6), y0 + row * (temp_tile_h + 8), temp_tile_w, temp_tile_h)
                    _draw_tile(screen, rect, slot)
                rows = (min(len(slots), 120) + per_row - 1) // per_row
                y_cursor += 16 + rows * (temp_tile_h + 8) + 8

            deck_remaining_ids = timeline.current.deck_order[timeline.current.deck_index :]
            deck_lbl = small_font.render(f"Deck remaining: {len(deck_remaining_ids)} (top 20 shown)", True, text_color)
            screen.blit(deck_lbl, (debug_rect.x + 14, y_cursor))
            preview = _slots_from_tile_ids(deck_remaining_ids[:20])
            x0, y0 = debug_rect.x + 14, y_cursor + 16
            for j, slot in enumerate(preview):
                rect = pygame.Rect(x0 + j * (28 + 6), y0, 28, 40)
                _draw_tile(screen, rect, slot)
            y_cursor += 16 + 40 + 12

            content_h = y_cursor - (debug_rect.y + 62)
            screen.set_clip(None)
            max_dbg = max(0, content_h - clip.height)
            debug_scroll = max(0, min(debug_scroll, max_dbg))

        # Help overlay
        if help_open:
            overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0, 0))
            help_rect = pygame.Rect(int(screen_w * 0.18), int(screen_h * 0.14), int(screen_w * 0.64), int(screen_h * 0.68))
            pygame.draw.rect(screen, panel_color, help_rect, border_radius=12)
            pygame.draw.rect(screen, accent_color, help_rect, width=2, border_radius=12)
            screen.blit(title_font.render("Help", True, text_color), (help_rect.x + 16, help_rect.y + 14))

            lines = [
                "Core interactions:",
                "• Drag tiles from Hand ↔ Table (only at the end of the timeline)",
                "• Click a + slot (before/after/only), then click a hand tile to add quickly",
                "• Right click a table tile: select it; if it came from hand, right click returns it to hand",
                "Joker:",
                "• Select a joker tile, then press C to cycle color, V to cycle value",
                "Timeline:",
                "• Left/Right arrows: step through history",
                "• History button: jump to any step",
                "Shortcuts:",
                "• Enter: Play   • D: Draw   • P: Pass   • M: + Meld   • R: Reset draft   • T: Toggle hand sort",
                "• G: Debug overlay   • H: Help   • Esc: close overlay / quit",
            ]
            y = help_rect.y + 54
            for line in lines:
                s = small_font.render(line, True, text_color)
                screen.blit(s, (help_rect.x + 16, y))
                y += 22

        # Modal history picker
        buttons_modal: List[Tuple[pygame.Rect, int]] = []
        modal_rect = pygame.Rect(0, 0, 0, 0)
        prev_m = next_m = close_m = pygame.Rect(0, 0, 0, 0)
        if modal_open:
            overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            screen.blit(overlay, (0, 0))
            modal_rect = pygame.Rect(int(screen_w * 0.18), int(screen_h * 0.18), int(screen_w * 0.64), int(screen_h * 0.64))
            pygame.draw.rect(screen, panel_color, modal_rect, border_radius=12)
            pygame.draw.rect(screen, accent_color, modal_rect, width=2, border_radius=12)
            title = title_font.render("Select a previous step", True, text_color)
            screen.blit(title, (modal_rect.x + 16, modal_rect.y + 14))

            page_size = max(12, (modal_rect.height - 120) // 28)
            total_pages = max(1, (len(timeline.history) + page_size - 1) // page_size)
            modal_page = max(0, min(modal_page, total_pages - 1))
            start = modal_page * page_size
            end = min(len(timeline.history), start + page_size)

            y_cursor = modal_rect.y + 54
            for idx in range(start, end):
                rect = pygame.Rect(modal_rect.x + 20, y_cursor, modal_rect.width - 40, 24)
                color = accent_color if idx == timeline.index else (90, 94, 110)
                pygame.draw.rect(screen, color, rect, border_radius=6)
                label = small_font.render(f"Step {idx}  (Turn {timeline.history[idx].turn_number}, P{timeline.history[idx].current_player + 1})", True, text_color)
                screen.blit(label, (rect.x + 10, rect.y + 4))
                buttons_modal.append((rect, idx))
                y_cursor += 28

            prev_m = pygame.Rect(modal_rect.x + 20, modal_rect.bottom - 44, 90, 28)
            next_m = pygame.Rect(modal_rect.right - 110, modal_rect.bottom - 44, 90, 28)
            close_m = pygame.Rect(modal_rect.right - 110, modal_rect.y + 14, 90, 28)

            for r, t in [(prev_m, "Prev"), (next_m, "Next"), (close_m, "Close")]:
                pygame.draw.rect(screen, panel_color_2, r, border_radius=8)
                pygame.draw.rect(screen, accent_color, r, width=2, border_radius=8)
                screen.blit(small_font.render(t, True, text_color), small_font.render(t, True, text_color).get_rect(center=r.center))

        # Status bar
        pygame.draw.rect(screen, panel_color_2, status_rect, border_radius=10)
        pygame.draw.rect(screen, panel_color, status_rect, width=2, border_radius=10)
        now = pygame.time.get_ticks()
        if toast_until_ms and now > toast_until_ms:
            toast_kind = "info"
            toast_until_ms = 0

        kind_color = {
            "info": text_color,
            "success": success_color,
            "warning": warn_color,
            "error": error_color,
        }.get(toast_kind, text_color)
        msg = small_font.render(toast_text, True, kind_color)
        screen.blit(msg, (status_rect.x + 12, status_rect.y + 7))

        # Persistent turn indicator (useful for multi-player debugging)
        turn_lbl = small_font.render(f"À jouer : Joueur {player_idx + 1}", True, pcol)
        screen.blit(turn_lbl, (status_rect.right - turn_lbl.get_width() - 12, status_rect.y + 7))

        # Tooltips (draw last)
        if hover_tooltip and not (help_open or modal_open):
            _draw_tooltip(hover_tooltip, mouse_pos)

        # One-time screenshot support
        if screenshot_path and not screenshot_taken:
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(screen, str(screenshot_path))
            screenshot_taken = True

        # Events ----------------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                tile_cache.clear()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if modal_open:
                        modal_open = False
                    elif help_open:
                        help_open = False
                    else:
                        running = False
                elif event.key == pygame.K_h:
                    _on_toggle_help()
                elif event.key == pygame.K_g:
                    _on_toggle_debug()
                elif event.key == pygame.K_PAGEUP:
                    if show_debug:
                        debug_scroll = max(0, debug_scroll - 120)
                    elif table_rect.collidepoint(mouse_pos):
                        table_scroll = max(0, table_scroll - 120)
                elif event.key == pygame.K_PAGEDOWN:
                    if show_debug:
                        debug_scroll += 120
                    elif table_rect.collidepoint(mouse_pos):
                        table_scroll += 120
                elif event.key == pygame.K_LEFT:
                    timeline.jump(timeline.index - 1)
                    edited_table = timeline.current.table.canonicalize()
                    selected_slot = None
                    selected_empty_slot = None
                    carried = None
                    carried_from = None
                    hand_scroll = 0
                    set_toast("", kind="info", ttl_ms=0)
                elif event.key == pygame.K_RIGHT:
                    timeline.jump(timeline.index + 1)
                    edited_table = timeline.current.table.canonicalize()
                    selected_slot = None
                    selected_empty_slot = None
                    carried = None
                    carried_from = None
                    hand_scroll = 0
                    set_toast("", kind="info", ttl_ms=0)
                elif event.key == pygame.K_RETURN:
                    if can_play:
                        _on_play()
                elif event.key == pygame.K_d:
                    if can_play:
                        _on_draw()
                elif event.key == pygame.K_p:
                    if can_play:
                        _on_pass()
                elif event.key == pygame.K_m:
                    if can_play:
                        _add_meld()
                elif event.key == pygame.K_r:
                    if can_play:
                        _reset_draft()
                elif event.key == pygame.K_t:
                    _on_toggle_sort()
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
                                tile_cache.clear()
                                set_toast(f"Joker set to color={new_color}, value={new_value}", kind="info", ttl_ms=2000)
                elif event.key == pygame.K_s:
                    if screenshot_path:
                        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                        pygame.image.save(screen, str(screenshot_path))
                        set_toast(f"Screenshot saved to {screenshot_path}", kind="success")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if help_open:
                        # click outside to close
                        help_open = False
                        continue

                    if modal_open:
                        if close_m.collidepoint(event.pos):
                            modal_open = False
                            continue
                        if prev_m.collidepoint(event.pos):
                            modal_page = max(0, modal_page - 1)
                            continue
                        if next_m.collidepoint(event.pos):
                            modal_page += 1
                            continue
                        for rect, idx in buttons_modal:
                            if rect.collidepoint(event.pos):
                                timeline.jump(idx)
                                edited_table = timeline.current.table.canonicalize()
                                selected_slot = None
                                selected_empty_slot = None
                                carried = None
                                carried_from = None
                                modal_open = False
                                break
                        continue

                    # toolbar buttons
                    for b in buttons:
                        if b.enabled and b.rect.collidepoint(event.pos):
                            b.handler()
                            break

                    # timeline controls
                    if prev_rect.collidepoint(event.pos):
                        timeline.jump(timeline.index - 1)
                        edited_table = timeline.current.table.canonicalize()
                        selected_slot = None
                        selected_empty_slot = None
                        carried = None
                        carried_from = None
                        hand_scroll = 0
                        continue
                    if next_rect.collidepoint(event.pos):
                        timeline.jump(timeline.index + 1)
                        edited_table = timeline.current.table.canonicalize()
                        selected_slot = None
                        selected_empty_slot = None
                        carried = None
                        carried_from = None
                        hand_scroll = 0
                        continue
                    if hist_rect.collidepoint(event.pos):
                        _open_history()
                        continue
                    for seg_rect, idx in segments:
                        if seg_rect.collidepoint(event.pos):
                            timeline.jump(idx)
                            edited_table = timeline.current.table.canonicalize()
                            selected_slot = None
                            selected_empty_slot = None
                            carried = None
                            carried_from = None
                            hand_scroll = 0
                            break

                    # select meld extension slot
                    for slot_rect, meld_idx, position in empty_slots:
                        if slot_rect.collidepoint(event.pos):
                            selected_empty_slot = None if selected_empty_slot == (meld_idx, position) else (meld_idx, position)
                            break

                    # drag start from table
                    if can_play:
                        for rect, slot, meld_idx, slot_idx in table_tiles:
                            if rect.collidepoint(event.pos):
                                carried = slot
                                carried_from = ("table", meld_idx, slot_idx)
                                if meld_idx < len(edited_table.melds) and slot_idx < len(edited_table.melds[meld_idx].slots):
                                    edited_table.melds[meld_idx].slots.pop(slot_idx)
                                    _remove_empty_melds(edited_table)
                                    selected_slot = None
                                break

                        # drag/click from hand
                        if carried is None:
                            for rect, slot in hand_tiles:
                                if rect.collidepoint(event.pos):
                                    if selected_empty_slot and can_play:
                                        meld_idx, position = selected_empty_slot
                                        placed = _try_insert_into_meld(meld_idx, slot, position)
                                        if placed:
                                            set_toast("Tile added", kind="info", ttl_ms=1500)
                                        else:
                                            set_toast("Cannot extend meld with this tile", kind="warning")
                                        break
                                    carried = slot
                                    carried_from = ("hand", -1, -1)
                                    break
                elif event.button == 3:
                    if modal_open or help_open:
                        continue
                    # right click selects tile or returns it to hand if it was added from hand
                    for rect, slot, meld_idx, slot_idx in table_tiles:
                        if rect.collidepoint(event.pos):
                            if can_play and _can_remove_from_hand(slot):
                                edited_table.melds[meld_idx].slots.pop(slot_idx)
                                _remove_empty_melds(edited_table)
                                set_toast("Returned tile to hand", kind="info", ttl_ms=2000)
                            else:
                                selected_slot = (meld_idx, slot_idx)
                                if slot.tile_id == JOKER_ID:
                                    set_toast("Joker selected. Use C/V to change color/value.", kind="info", ttl_ms=3000)
                            break
                    for slot_rect, meld_idx, position in empty_slots:
                        if slot_rect.collidepoint(event.pos):
                            if position == "only" and not edited_table.melds[meld_idx].slots:
                                edited_table.melds.pop(meld_idx)
                                _ensure_empty_meld(edited_table)
                                set_toast("Empty meld removed", kind="info", ttl_ms=2000)
                            break
                elif event.button in (4, 5):
                    # mouse wheel
                    if show_debug:
                        delta = -60 if event.button == 4 else 60
                        debug_scroll = max(0, debug_scroll + delta)
                    elif hand_rect.collidepoint(event.pos):
                        delta = -90 if event.button == 4 else 90
                        hand_scroll = max(0, hand_scroll + delta)
                    elif table_rect.collidepoint(event.pos):
                        delta = -90 if event.button == 4 else 90
                        table_scroll = max(0, table_scroll + delta)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and carried is not None:
                    _handle_drop(event.pos, hand_rect, table_tiles)
                    _remove_empty_melds(edited_table)

        # Drag ghost
        if carried is not None:
            mx, my = pygame.mouse.get_pos()
            mouse_rect = pygame.Rect(mx - tile_w // 2, my - tile_h // 2, tile_w, tile_h)
            _draw_tile(screen, mouse_rect, carried, highlight=True)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":  # pragma: no cover
    launch_gui()
