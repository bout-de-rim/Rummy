
from __future__ import annotations

"""
Rummikub — Compact GUI (v5)

New in v5 (requested):
- Drag & drop between HAND <-> TABLE and within TABLE (subject to authorization).
  * Hand -> Table: drop on a RUN row or a GROUP column.
  * Table -> Hand: only allowed for tiles that were placed from the player's hand in the current draft.
  * Table -> Table: drag any tile; drop onto a RUN row / GROUP column (joker may be reassigned to match target).
- Startup/opening period indicator:
  * Shows whether the current player is still "opening" (needs >=30 points) and the current draft points toward 30.
  * Uses best-effort introspection to detect whether the engine tracks "opened" per player; otherwise shows points anyway.
- Provenance / ownership visual code:
  * Tiles placed from the current player's hand during the current draft are highlighted on the table.
  * The tile drawn by the last DRAW action is highlighted in the hand (if detectable).
- Debug (godmode) restored:
  * Shows all hands and remaining deck (top preview + counts) with scrolling.
  * Uses state.deck_order / state.deck_index when available.

Hard constraint:
- This GUI only supports exactly 4 colors and 13 values. If not, it refuses to start.
"""

from dataclasses import dataclass
import os
import pathlib
from typing import Dict, List, Optional, Tuple, Any
import traceback

# --- Imports: support package and script execution -----------------------------
try:
    from .engine import apply_move, is_legal_move
    from .meld import Meld, MeldKind
    from .move import Move, MoveKind
    from .multiset import TileMultiset
    from .rules import Ruleset
    from .state import GameState, new_game
    from .table import Table
    from .tiles import JOKER_ID, TileSlot
except Exception:  # fallback when executed as a script
    from engine import apply_move, is_legal_move
    from meld import Meld, MeldKind
    from move import Move, MoveKind
    from multiset import TileMultiset
    from rules import Ruleset
    from state import GameState, new_game
    from table import Table
    from tiles import JOKER_ID, TileSlot

try:
    import pygame  # type: ignore
except Exception:  # pragma: no cover
    pygame = None  # type: ignore


# --- Core helpers (adapted from original gui.py) --------------------------------

def _safe_canonical_table(table: Table) -> Tuple[Optional[Table], str]:
    try:
        return table.canonicalize(), ""
    except ValueError as exc:
        return None, str(exc)


def compute_delta_from_tables(base_table: Table, edited_table: Table) -> Tuple[Optional[TileMultiset], str]:
    base_ms = base_table.multiset().counts
    new_ms = edited_table.multiset().counts
    diff = [n - b for n, b in zip(new_ms, base_ms)]
    if any(d < 0 for d in diff):
        return None, "table is missing tiles from previous state"
    return TileMultiset(diff), ""


def resolve_screenshot_path() -> pathlib.Path:
    env_path = os.getenv("RUMMY_GUI_SCREENSHOT_PATH")
    if env_path:
        return pathlib.Path(env_path)
    return pathlib.Path("rummy_gui_screenshot.png")


def build_play_move(state: GameState, edited_table: Table) -> Tuple[Optional[Move], str]:
    base_table = state.table.canonicalize()
    draft_points = _draft_points_toward_opening(base_table, edited_table)
    if not state.initial_meld_done[state.current_player]:
        if draft_points >= state.ruleset.initial_meld_min_points:
            edited_table = _merge_new_run_melds_for_opening(state, edited_table)
    non_empty = Table([meld for meld in edited_table.melds if meld.slots])
    canonical_table, error = _safe_canonical_table(non_empty)
    if canonical_table is None:
        return None, error

    delta, error = compute_delta_from_tables(base_table, canonical_table)
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
    non_empty = Table([meld for meld in edited_table.melds if meld.slots])
    canonical_table, _ = _safe_canonical_table(non_empty)
    if canonical_table is None:
        return state.hands[state.current_player].copy()

    delta, _ = compute_delta_from_tables(state.table.canonicalize(), canonical_table)
    if delta is None:
        return state.hands[state.current_player].copy()

    try:
        return state.hands[state.current_player].sub(delta)
    except ValueError:
        return state.hands[state.current_player].copy()


@dataclass
class TimelineController:
    timeline: List[GameState]
    index: int = 0

    @classmethod
    def from_state(cls, state: GameState) -> "TimelineController":
        return cls([state], 0)

    @property
    def current(self) -> GameState:
        return self.timeline[self.index]

    def at_end(self) -> bool:
        return self.index == len(self.timeline) - 1

    def append(self, state: GameState) -> None:
        self.timeline.append(state)
        self.index = len(self.timeline) - 1

    def jump(self, index: int) -> None:
        if 0 <= index < len(self.timeline):
            self.index = index


# --- Theme ---------------------------------------------------------------------

BG = (22, 27, 34)
PANEL = (30, 36, 46)
PANEL_LINE = (54, 63, 77)
TEXT = (220, 226, 235)
SUB = (164, 174, 187)
ACCENT = (255, 105, 180)
HILITE_NEW = (64, 255, 255)  # last drawn / emphasis
ERR = (235, 87, 87)
WARN = (255, 170, 40)

COLOR_PALETTE = [
    (220, 80, 80),    # red
    (86, 148, 227),   # blue
    (66, 171, 119),   # green
    (221, 164, 66),   # yellow/orange
]
JOKER_COLOR = (170, 110, 210)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _mix(c: Tuple[int, int, int], f: float, toward=(0, 0, 0)) -> Tuple[int, int, int]:
    return (int(c[0] * (1 - f) + toward[0] * f),
            int(c[1] * (1 - f) + toward[1] * f),
            int(c[2] * (1 - f) + toward[2] * f))


def _tile_value(slot: TileSlot) -> Optional[int]:
    if slot.tile_id == JOKER_ID:
        return slot.assigned_value
    return (slot.tile_id % 13) + 1


def _tile_color_index(slot: TileSlot) -> Optional[int]:
    if slot.tile_id == JOKER_ID:
        return slot.assigned_color
    return (slot.tile_id // 13) % 4


def _tile_bg(slot: TileSlot) -> Tuple[int, int, int]:
    ci = _tile_color_index(slot)
    if ci is None:
        return JOKER_COLOR
    return COLOR_PALETTE[ci]


def _tile_text_color(bg: Tuple[int, int, int]) -> Tuple[int, int, int]:
    lum = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
    return (20, 22, 26) if lum > 140 else (245, 246, 250)


def _hand_grid_counts(current_hand: TileMultiset) -> Dict[Tuple[int, int], int]:
    grid: Dict[Tuple[int, int], int] = {}
    for tid, cnt in enumerate(current_hand.counts):
        if cnt <= 0:
            continue
        if tid == JOKER_ID:
            grid[(0, 13)] = grid.get((0, 13), 0) + cnt
        else:
            color = (tid // 13) % 4
            value_idx = (tid % 13)
            grid[(color, value_idx)] = grid.get((color, value_idx), 0) + cnt
    return grid


def _total_tiles(ms: TileMultiset) -> int:
    return sum(ms.counts)


def _points_for_slot(slot: TileSlot) -> int:
    v = _tile_value(slot)
    return int(v or 0)


def _draft_points_toward_opening(base_table: Table, edited_table: Table) -> int:
    """Sum points of tiles newly added to the table compared to base_table, using joker assignments on the edited table."""
    base_counts = base_table.multiset().counts
    # count occurrences in edited_table by tile_id, but keep per-slot to assign joker values
    # We mark a slot as "new" if its occurrence index exceeds base_count for that tile_id.
    occ_seen = [0] * len(base_counts)
    pts = 0
    for meld in edited_table.melds:
        for s in meld.slots:
            tid = s.tile_id
            if tid < 0 or tid >= len(base_counts):
                continue
            occ_seen[tid] += 1
            if occ_seen[tid] > base_counts[tid]:
                pts += _points_for_slot(s)
    return pts


def _new_only_meld_indices(base_table: Table, edited_table: Table) -> set[int]:
    base_counts = base_table.multiset().counts
    seen = [0] * len(base_counts)
    new_only: set[int] = set()
    for idx, meld in enumerate(edited_table.melds):
        meld_new_flags: List[bool] = []
        for s in meld.slots:
            tid = s.tile_id
            if 0 <= tid < len(base_counts):
                seen[tid] += 1
                is_new = seen[tid] > base_counts[tid]
            else:
                is_new = False
            meld_new_flags.append(is_new)
        if meld_new_flags and all(meld_new_flags):
            new_only.add(idx)
    return new_only


def _merge_new_run_melds_for_opening(state: GameState, edited_table: Table) -> Table:
    base_table = state.table.canonicalize()
    base_counts = base_table.multiset().counts
    seen = [0] * len(base_counts)
    new_run_slots_by_color: Dict[int, List[TileSlot]] = {0: [], 1: [], 2: [], 3: []}
    kept_melds: List[Meld] = []

    for meld in edited_table.melds:
        meld_new_flags: List[bool] = []
        for s in meld.slots:
            tid = s.tile_id
            if 0 <= tid < len(base_counts):
                seen[tid] += 1
                is_new = seen[tid] > base_counts[tid]
            else:
                is_new = False
            meld_new_flags.append(is_new)
        if meld.kind == MeldKind.RUN and meld.slots and all(meld_new_flags):
            color = _run_color_for_meld(meld)
            new_run_slots_by_color[color].extend(meld.slots)
        else:
            kept_melds.append(meld)

    for color, slots in new_run_slots_by_color.items():
        if not slots:
            continue
        slots_sorted = sorted(slots, key=lambda s: _tile_value(s) or 0)
        current: List[TileSlot] = []
        prev_val: Optional[int] = None
        for s in slots_sorted:
            v = _tile_value(s)
            if v is None:
                if current:
                    kept_melds.append(Meld(kind=MeldKind.RUN, slots=current))
                current = [s]
                prev_val = v
                continue
            if prev_val is None or v == prev_val + 1:
                current.append(s)
            else:
                kept_melds.append(Meld(kind=MeldKind.RUN, slots=current))
                current = [s]
            prev_val = v
        if current:
            kept_melds.append(Meld(kind=MeldKind.RUN, slots=current))

    return Table(kept_melds)


def _last_drawn_tile_for_player(state: GameState, player: int) -> Optional[int]:
    for event in reversed(state.event_log):
        if event.player != player:
            continue
        if event.move_kind == MoveKind.DRAW.value:
            return event.payload.get("tile")
    return None


# --- Hit testing structures ----------------------------------------------------

@dataclass
class HandCell:
    rect: pygame.Rect
    row: int
    col: int
    tile_id: Optional[int]
    count: int


@dataclass
class RunsRowHit:
    rect: pygame.Rect
    row: int


@dataclass
class GroupsColHit:
    rect: pygame.Rect
    block: int
    value_idx: int


@dataclass
class TileHit:
    rect: pygame.Rect
    meld_idx: int
    slot_idx: int


# --- Board mapping -------------------------------------------------------------

def _run_color_for_meld(m: Meld) -> int:
    for s in m.slots:
        if s.tile_id != JOKER_ID:
            return (s.tile_id // 13) % 4
    for s in m.slots:
        if s.tile_id == JOKER_ID and s.assigned_color is not None:
            return s.assigned_color
    return 0


def _run_value_range(meld: Meld) -> Tuple[int, int]:
    values = [v for v in (_tile_value(s) for s in meld.slots) if v is not None]
    if not values:
        return (1, 1)
    return (min(values), max(values))


def map_runs_rows_to_meld_indices(table: Table) -> Dict[int, List[int]]:
    rows: Dict[int, List[int]] = {i: [] for i in range(8)}
    per_color: Dict[int, List[Tuple[int, Tuple[int, int]]]] = {0: [], 1: [], 2: [], 3: []}
    for idx, m in enumerate(table.melds):
        if m.kind != MeldKind.RUN or not m.slots:
            continue
        c = _run_color_for_meld(m)
        per_color.setdefault(c, []).append((idx, _run_value_range(m)))

    for color in range(4):
        idxs = per_color.get(color, [])
        idxs.sort(key=lambda item: (item[1][0], item[1][1]))
        row_slots = {color * 2: [], color * 2 + 1: []}
        for meld_idx, (vmin, vmax) in idxs:
            placed = False
            for row in (color * 2, color * 2 + 1):
                overlaps = any(not (vmax < a or vmin > b) for a, b in row_slots[row])
                if not overlaps:
                    row_slots[row].append((vmin, vmax))
                    rows[row].append(meld_idx)
                    placed = True
                    break
            if not placed:
                row = min((color * 2, color * 2 + 1), key=lambda r: len(row_slots[r]))
                row_slots[row].append((vmin, vmax))
                rows[row].append(meld_idx)
    return rows


def groups_by_value(table: Table) -> Dict[int, List[int]]:
    d: Dict[int, List[int]] = {i: [] for i in range(13)}
    for idx, m in enumerate(table.melds):
        if m.kind != MeldKind.GROUP or not m.slots:
            continue
        vals: List[int] = []
        for s in m.slots:
            v = _tile_value(s)
            if v is None:
                vals = []
                break
            vals.append(v)
        if not vals:
            continue
        unique_val = vals[0]
        if any(v != unique_val for v in vals):
            continue
        d[unique_val - 1].append(idx)
    return d


def group_block_count(table: Table) -> int:
    by_val = groups_by_value(table)
    max_per_value = max((len(v) for v in by_val.values()), default=0)
    return max(2, max_per_value)


def map_groups_cells_to_meld_indices(table: Table) -> Dict[Tuple[int, int], int]:
    mapping: Dict[Tuple[int, int], int] = {}
    by_val = groups_by_value(table)
    for val_idx, melds in by_val.items():
        for block, meld_idx in enumerate(melds):
            mapping[(block, val_idx)] = meld_idx
    return mapping


# --- Local "authorized" checks -------------------------------------------------

def can_insert_into_run(run_slots: List[TileSlot], new_slot: TileSlot, row_color: int) -> Tuple[bool, str]:
    if new_slot.tile_id == JOKER_ID:
        if new_slot.assigned_color != row_color:
            return False, "joker color mismatch"
    else:
        if ((new_slot.tile_id // 13) % 4) != row_color:
            return False, "tile color mismatch"

    candidate = list(run_slots) + [new_slot]

    vals: List[int] = []
    for s in candidate:
        v = _tile_value(s)
        if v is None:
            return False, "joker has no value"
        vals.append(v)

    if len(set(vals)) != len(vals):
        return False, "duplicate value"

    if len(vals) <= 2:
        return True, ""

    vs = sorted(vals)
    if vs[-1] - vs[0] != len(vs) - 1:
        return False, "run not consecutive"
    return True, ""


def can_insert_into_group(group_slots: List[TileSlot], new_slot: TileSlot, value_idx: int) -> Tuple[bool, str]:
    target_val = value_idx + 1
    candidate = list(group_slots) + [new_slot]
    if len(candidate) > 4:
        return False, "group too large"

    colors: List[int] = []
    for s in candidate:
        v = _tile_value(s)
        if v is None:
            return False, "joker has no value"
        if v != target_val:
            return False, "value mismatch"
        c = _tile_color_index(s)
        if c is None:
            return False, "joker has no color"
        colors.append(c)

    if len(set(colors)) != len(colors):
        return False, "duplicate color"
    return True, ""


# --- Provenance helpers (highlight tiles from hand during current draft) -------

def _table_tile_occurrence_counts(table: Table, n_tile_ids: int) -> List[int]:
    counts = [0] * n_tile_ids
    for m in table.melds:
        for s in m.slots:
            tid = s.tile_id
            if 0 <= tid < n_tile_ids:
                counts[tid] += 1
    return counts


def _compute_new_occurrence_cutoff(base_table: Table, edited_table: Table) -> Tuple[List[int], List[int]]:
    """Return (base_counts, new_counts) as flat counts per tile_id."""
    base_counts = base_table.multiset().counts
    edited_counts = edited_table.multiset().counts
    return base_counts, edited_counts


def _is_slot_new_by_occurrence(
    slot: TileSlot,
    base_counts: List[int],
    seen_counts: List[int],
) -> bool:
    tid = slot.tile_id
    if tid < 0 or tid >= len(base_counts):
        return False
    seen_counts[tid] += 1
    return seen_counts[tid] > base_counts[tid]


# --- Drawing primitives --------------------------------------------------------

def draw_panel(surface, rect: pygame.Rect):
    pygame.draw.rect(surface, PANEL, rect, border_radius=10)
    pygame.draw.rect(surface, PANEL_LINE, rect, width=2, border_radius=10)


def draw_button(surface, rect: pygame.Rect, label: str, font):
    pygame.draw.rect(surface, (46, 56, 69), rect, border_radius=10)
    pygame.draw.rect(surface, ACCENT, rect, width=2, border_radius=10)
    txt = font.render(label, True, TEXT)
    surface.blit(txt, txt.get_rect(center=rect.center))


def draw_tile(
    surface,
    rect: pygame.Rect,
    slot: TileSlot,
    small_font,
    highlight_border: Optional[Tuple[int, int, int]] = None,
    count_badge: Optional[int] = None,
    stacked_back: bool = False,
    ghost: bool = False,
):
    radius = 6
    bg = _tile_bg(slot)
    if stacked_back:
        bg = _mix(bg, 0.22, toward=(0, 0, 0))
    if ghost:
        bg = _mix(bg, 0.45, toward=BG)

    border = highlight_border if highlight_border is not None else (60, 70, 85)
    pygame.draw.rect(surface, bg, rect, border_radius=radius)
    if highlight_border == HILITE_NEW:
        outer = rect.inflate(8, 8)
        pygame.draw.rect(surface, highlight_border, outer, width=3, border_radius=radius + 2)
    pygame.draw.rect(surface, border, rect, width=3 if highlight_border else 2, border_radius=radius)

    inner = rect.inflate(-8, -8)
    v = _tile_value(slot)
    label = (f"J{v}" if slot.tile_id == JOKER_ID else str(v if v is not None else "?"))

    font_size = max(12, int(inner.height * 0.58))
    font = pygame.font.SysFont("arial", font_size, bold=True)
    txt = font.render(label, True, _tile_text_color(bg))
    surface.blit(txt, txt.get_rect(center=inner.center))

    if slot.tile_id == JOKER_ID:
        pygame.draw.line(surface, _tile_text_color(bg), (rect.left + 6, rect.bottom - 6), (rect.right - 6, rect.top + 6), 2)

    if count_badge and count_badge > 1:
        badge = small_font.render(f"×{count_badge}", True, (245, 246, 250))
        badge_bg = pygame.Rect(rect.right - 22, rect.top + 4, 18, 18)
        pygame.draw.rect(surface, (55, 65, 85), badge_bg, border_radius=9)
        surface.blit(badge, badge.get_rect(center=badge_bg.center))


def status_bar(surface, rect: pygame.Rect, left: str, right: str, small_font):
    pygame.draw.rect(surface, PANEL, rect, border_radius=10)
    pygame.draw.rect(surface, PANEL_LINE, rect, width=2, border_radius=10)
    l = small_font.render(left, True, SUB)
    r = small_font.render(right, True, SUB)
    surface.blit(l, (rect.x + 10, rect.y + (rect.height - l.get_height()) // 2))
    surface.blit(r, (rect.right - 10 - r.get_width(), rect.y + (rect.height - r.get_height()) // 2))


# --- Drag state ----------------------------------------------------------------

@dataclass
class DragPayload:
    source: str  # "hand" or "table"
    tile_id: int
    slot: TileSlot
    # for table source
    meld_idx: Optional[int] = None
    slot_idx: Optional[int] = None
    # for hand source
    hand_cell: Optional[Tuple[int, int]] = None


# --- GUI entry -----------------------------------------------------------------


@dataclass
class DraftMoveManager:
    state: GameState
    edited_table: Table
    hand_joker_value: int = 1

    def refresh(
        self,
        *,
        state: Optional[GameState] = None,
        edited_table: Optional[Table] = None,
        hand_joker_value: Optional[int] = None,
    ) -> None:
        if state is not None:
            self.state = state
        if edited_table is not None:
            self.edited_table = edited_table
        if hand_joker_value is not None:
            self.hand_joker_value = hand_joker_value

    def remove_slot_from_table(self, meld_idx: int, slot_idx: int) -> TileSlot:
        m = self.edited_table.melds[meld_idx]
        s = m.slots.pop(slot_idx)
        if not m.slots:
            # keep structure minimal
            self.edited_table.melds.pop(meld_idx)
        return s

    def adapt_slot_for_target(self, existing: TileSlot, target: Tuple[str, int, int]) -> TileSlot:
        """Adjust joker assignment to match drop target when needed; non-jokers unchanged."""
        kind, a, b = target
        if existing.tile_id != JOKER_ID:
            return existing

        if kind == "group":
            block, value_idx = a, b
            target_val = value_idx + 1
            mapping = map_groups_cells_to_meld_indices(self.edited_table)
            meld_idx = mapping.get((block, value_idx))
            used = set()
            if meld_idx is not None:
                for s in self.edited_table.melds[meld_idx].slots:
                    c = _tile_color_index(s)
                    if c is not None:
                        used.add(c)
            # For a joker dragged from table, if it already has a color that is unused, keep it; else choose first missing.
            color = existing.assigned_color
            if color is None or color in used:
                color = next((c for c in range(4) if c not in used), 0)
            return TileSlot(JOKER_ID, assigned_color=color, assigned_value=target_val)

        # run
        row = a
        row_color = row // 2
        v = existing.assigned_value if existing.assigned_value is not None else self.hand_joker_value
        return TileSlot(JOKER_ID, assigned_color=row_color, assigned_value=v)

    def assign_joker_for_run(
        self,
        slot: TileSlot,
        row_color: int,
        touching_ranges: List[Tuple[int, int]],
    ) -> TileSlot:
        if slot.tile_id != JOKER_ID:
            return slot
        candidates: List[int] = []
        for vmin, vmax in touching_ranges:
            candidates.extend([vmin - 1, vmax + 1])
        candidates = [v for v in candidates if 1 <= v <= 13]
        chosen: Optional[int] = None
        if slot.assigned_value in candidates:
            chosen = slot.assigned_value
        elif candidates:
            chosen = candidates[0]
        else:
            chosen = slot.assigned_value or self.hand_joker_value
        return TileSlot(JOKER_ID, assigned_color=row_color, assigned_value=chosen)

    def new_only_melds_for_player(self) -> Optional[set[int]]:
        if self.state.initial_meld_done[self.state.current_player]:
            return None
        base_table = self.state.table.canonicalize()
        return _new_only_meld_indices(base_table, self.edited_table)

    def run_candidate_ranges(self, row: int, row_color: int, new_only_melds: Optional[set[int]]) -> List[Tuple[int, int, int]]:
        row_map = map_runs_rows_to_meld_indices(self.edited_table)
        other_row = row_color * 2 + (1 if row == row_color * 2 else 0)
        meld_candidates: List[int] = []
        for candidate_idx in row_map.get(row, []):
            if candidate_idx not in meld_candidates:
                meld_candidates.append(candidate_idx)
        for candidate_idx in row_map.get(other_row, []):
            if candidate_idx not in meld_candidates:
                meld_candidates.append(candidate_idx)

        candidate_ranges: List[Tuple[int, int, int]] = []
        for candidate_idx in meld_candidates:
            if new_only_melds is not None and candidate_idx not in new_only_melds:
                continue
            vmin, vmax = _run_value_range(self.edited_table.melds[candidate_idx])
            candidate_ranges.append((candidate_idx, vmin, vmax))
        return candidate_ranges

    @staticmethod
    def touching_run_indices(slot_value: int, candidate_ranges: List[Tuple[int, int, int]]) -> List[int]:
        touching: List[int] = []
        for candidate_idx, vmin, vmax in candidate_ranges:
            if slot_value in (vmin - 1, vmax + 1):
                touching.append(candidate_idx)
        return touching

    def merge_touching_runs(
        self,
        touching: List[int],
        slot: TileSlot,
        row_color: int,
        slot_value: int,
    ) -> Tuple[bool, str]:
        if len(touching) < 2:
            return False, "not mergeable"
        touching_sorted = sorted(touching, key=lambda idx: _run_value_range(self.edited_table.melds[idx])[0])
        for i in range(len(touching_sorted) - 1):
            first_idx = touching_sorted[i]
            second_idx = touching_sorted[i + 1]
            first_range = _run_value_range(self.edited_table.melds[first_idx])
            second_range = _run_value_range(self.edited_table.melds[second_idx])
            if slot_value == first_range[1] + 1 and slot_value == second_range[0] - 1:
                combined_slots = (
                    self.edited_table.melds[first_idx].slots
                    + self.edited_table.melds[second_idx].slots
                )
                ok, why = can_insert_into_run(combined_slots, slot, row_color)
                if not ok:
                    return False, why
                for idx in sorted([first_idx, second_idx], reverse=True):
                    self.edited_table.melds.pop(idx)
                self.edited_table.melds.append(
                    Meld(kind=MeldKind.RUN, slots=combined_slots + [slot])
                )
                return True, ""
        return False, "run not consecutive"

    def insert_into_existing_run(self, touching: List[int], slot: TileSlot, row_color: int, slot_value: int) -> Tuple[bool, str]:
        for candidate_idx in touching:
            vmin, vmax = _run_value_range(self.edited_table.melds[candidate_idx])
            if slot_value not in (vmin - 1, vmax + 1):
                continue
            ok, _ = can_insert_into_run(self.edited_table.melds[candidate_idx].slots, slot, row_color)
            if ok:
                self.edited_table.melds[candidate_idx].slots.append(slot)
                return True, ""
        return False, "run not consecutive"

    def find_group_meld(self, block: int, value_idx: int, new_only_melds: Optional[set[int]]) -> Tuple[Optional[int], str]:
        mapping = map_groups_cells_to_meld_indices(self.edited_table)
        meld_idx = mapping.get((block, value_idx))
        if meld_idx is None:
            return None, ""
        if new_only_melds is not None and meld_idx not in new_only_melds:
            return None, "meld non autorisé avant ouverture"
        return meld_idx, ""

    def insert_slot_into_target(self, slot: TileSlot, target: Tuple[str, int, int]) -> Tuple[bool, str]:
        """Try to insert slot into target meld; create meld if needed."""
        kind, a, b = target

        if kind == "run":
            row = a
            row_color = row // 2
            new_only_melds = self.new_only_melds_for_player()
            candidate_ranges = self.run_candidate_ranges(row, row_color, new_only_melds)
            touching_ranges = [(vmin, vmax) for _, vmin, vmax in candidate_ranges]
            slot = self.assign_joker_for_run(slot, row_color, touching_ranges)
            slot_value = _tile_value(slot)
            if slot_value is None:
                return False, "joker has no value"

            touching = self.touching_run_indices(slot_value, candidate_ranges)

            if not touching:
                ok, why = can_insert_into_run([], slot, row_color)
                if not ok:
                    return False, why
                self.edited_table.melds.append(Meld(kind=MeldKind.RUN, slots=[slot]))
                return True, ""

            merged, merge_reason = self.merge_touching_runs(touching, slot, row_color, slot_value)
            if merged:
                return True, ""
            inserted, insert_reason = self.insert_into_existing_run(touching, slot, row_color, slot_value)
            if inserted:
                return True, ""
            return False, merge_reason if merge_reason != "not mergeable" else insert_reason

        # group
        block, value_idx = a, b
        new_only_melds = self.new_only_melds_for_player()
        meld_idx, reason = self.find_group_meld(block, value_idx, new_only_melds)
        if reason:
            return False, reason
        if meld_idx is None:
            ok, why = can_insert_into_group([], slot, value_idx)
            if not ok:
                return False, why
            self.edited_table.melds.append(Meld(kind=MeldKind.GROUP, slots=[slot]))
            return True, ""
        ok, why = can_insert_into_group(self.edited_table.melds[meld_idx].slots, slot, value_idx)
        if not ok:
            return False, why
        self.edited_table.melds[meld_idx].slots.append(slot)
        return True, ""

    def slot_can_return_to_hand(self, meld_idx: int, slot_idx: int) -> bool:
        """Only allow returning to hand tiles that were added from hand in this draft (delta positive)."""
        base_counts = self.state.table.canonicalize().multiset().counts
        # Determine if this specific slot is beyond base occurrences.
        seen = [0] * len(base_counts)
        # Reproduce the same traversal order as rendering for determinism.
        for mi, meld in enumerate(self.edited_table.melds):
            for si, s in enumerate(meld.slots):
                tid = s.tile_id
                if 0 <= tid < len(base_counts):
                    seen[tid] += 1
                    is_new = seen[tid] > base_counts[tid]
                else:
                    is_new = False
                if mi == meld_idx and si == slot_idx:
                    return is_new
        return False

    @staticmethod
    def clone_table(table: Table) -> Table:
        return Table([Meld(kind=m.kind, slots=list(m.slots)) for m in table.melds])

    def can_move_table_tile(self, meld_idx: int, slot_idx: int) -> bool:
        if self.state.initial_meld_done[self.state.current_player]:
            return True
        return self.slot_can_return_to_hand(meld_idx, slot_idx)

    def move_table_slot(self, meld_idx: int, slot_idx: int, drop_target: Tuple[str, int, int]) -> Tuple[bool, str]:
        if not self.can_move_table_tile(meld_idx, slot_idx):
            return False, "Déplacement refusé (avant ouverture)."
        snapshot = self.clone_table(self.edited_table)
        try:
            slot = self.edited_table.melds[meld_idx].slots[slot_idx]
        except Exception:
            slot = None
        try:
            removed = self.remove_slot_from_table(meld_idx, slot_idx)
        except Exception:
            removed = slot
        if removed is None:
            return False, "Déplacement refusé: tuile introuvable."
        moved = self.adapt_slot_for_target(removed, drop_target)
        ok, why = self.insert_slot_into_target(moved, drop_target)
        if not ok:
            self.edited_table = snapshot
            return False, f"Déplacement refusé: {why}"
        return True, "Déplacement effectué."

def launch_gui(seed: Optional[int] = None, ruleset: Optional[Ruleset] = None) -> None:  # pragma: no cover
    if pygame is None:
        raise ImportError("pygame is required for the GUI. Install it with `pip install pygame`.")

    ruleset = ruleset or Ruleset()
    if getattr(ruleset, "colors", 4) != 4 or getattr(ruleset, "values", 13) != 13:
        raise RuntimeError("This compact GUI only supports exactly 4 colors and 13 values.")

    pygame.init()
    pygame.display.set_caption("Rummikub — Compact GUI")
    screen = pygame.display.set_mode((1280, 768), pygame.RESIZABLE)
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("arial", 18)
    title_font = pygame.font.SysFont("arial", 24, bold=True)
    small_font = pygame.font.SysFont("arial", 14)
    axis_font = pygame.font.SysFont("arial", 12)

    # Game state + timeline
    state = new_game(ruleset=ruleset, rng_seed=seed)
    timeline: List[GameState] = [state]
    time_idx = 0

    def current() -> GameState:
        return timeline[time_idx]

    edited_table = current().table.canonicalize()
    manager = DraftMoveManager(current(), edited_table, hand_joker_value)

    # UI state
    selected_target: Optional[Tuple[str, int, int]] = None  # ('run', row, -1) or ('group', block, value_idx)
    message = "Glisser/déposer: main ↔ plateau. Ou sélectionnez un RUN/GROUP puis cliquez une tuile."
    show_debug = False
    show_help = False
    pending_draw_confirm = False
    invalid_melds: set[int] = set()
    debug_scroll = 10**9
    debug_log: List[str] = []
    hand_joker_value = 1

    # Drag state
    drag: Optional[DragPayload] = None
    drag_offset = (0, 0)

    # Hit regions
    hand_cells: List[HandCell] = []
    runs_row_hits: List[RunsRowHit] = []
    groups_col_hits: List[GroupsColHit] = []
    tile_hits: List[TileHit] = []

    def reset_draft():
        nonlocal edited_table, selected_target, drag, pending_draw_confirm, invalid_melds, manager
        edited_table = current().table.canonicalize()
        manager.refresh(state=current(), edited_table=edited_table)
        selected_target = None
        drag = None
        pending_draw_confirm = False
        invalid_melds = set()

    def _draft_delta_from_hand(state: GameState, table: Table) -> Optional[TileMultiset]:
        non_empty = Table([meld for meld in table.melds if meld.slots])
        canonical_table, error = _safe_canonical_table(non_empty)
        if canonical_table is None:
            return None
        delta, error = compute_delta_from_tables(state.table.canonicalize(), canonical_table)
        if delta is None:
            return None
        return delta

    def _is_noop_draft(state: GameState, table: Table) -> bool:
        non_empty = Table([meld for meld in table.melds if meld.slots])
        canonical_table, error = _safe_canonical_table(non_empty)
        if canonical_table is None:
            return False
        base_table = state.table.canonicalize()
        return canonical_table == base_table

    def _perform_draw_action():
        nonlocal edited_table, message, pending_draw_confirm, time_idx, invalid_melds, manager
        before = current()
        if before.deck_index >= len(before.deck_order):
            move = Move.skip()
            message = "Pioche impossible: PASS."
        else:
            move = Move.draw()
        try:
            nxt = apply_move(before, move)
        except ValueError as exc:
            pending_draw_confirm = False
            message = f"Pioche impossible: {exc}"
            return

        if time_idx != len(timeline) - 1:
            timeline[:] = timeline[: time_idx + 1]
        timeline.append(nxt)
        time_idx = len(timeline) - 1
        edited_table = current().table.canonicalize()
        manager.refresh(state=current(), edited_table=edited_table)
        pending_draw_confirm = False
        invalid_melds = set()
        message = "Pioche effectuée."

    def crash_screen(tb: str):
        lines = tb.splitlines()[-40:]
        W, H = screen.get_size()
        while True:
            screen.fill((15, 18, 23))
            screen.blit(title_font.render("GUI crashed — traceback:", True, ERR), (20, 20))
            y = 60
            for ln in lines:
                s = small_font.render(ln[:180], True, (230, 230, 230))
                screen.blit(s, (20, y))
                y += small_font.get_height() + 2
                if y > H - 40:
                    break
            screen.blit(small_font.render("Press ESC or close the window.", True, WARN), (20, H - 28))
            pygame.display.flip()
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    return
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

    def try_place_from_hand(tile_id: int):
        nonlocal message, pending_draw_confirm, invalid_melds, edited_table
        if selected_target is None:
            message = "Aucun meld sélectionné."
            return
        if tile_id != JOKER_ID:
            slot = TileSlot(tile_id)
        else:
            # joker -> use target adaptation
            slot = TileSlot(JOKER_ID, assigned_color=None, assigned_value=hand_joker_value)
            slot = manager.adapt_slot_for_target(slot, selected_target)

        ok, why = manager.insert_slot_into_target(slot, selected_target)
        if ok:
            edited_table = manager.edited_table
            message = "Tuile placée."
            pending_draw_confirm = False
            invalid_melds = set()
        else:
            message = f"Placement refusé: {why}"

    def _drop_target_from_pos(pos: Tuple[int, int]) -> Tuple[Optional[Tuple[str, int, int]], bool]:
        drop_target: Optional[Tuple[str, int, int]] = None
        dropped_to_hand = hand_panel.collidepoint(pos)
        for rr in runs_row_hits:
            if rr.rect.collidepoint(pos):
                drop_target = ("run", rr.row, -1)
                break
        if drop_target is None:
            for gc in groups_col_hits:
                if gc.rect.collidepoint(pos):
                    drop_target = ("group", gc.block, gc.value_idx)
                    break
        return drop_target, dropped_to_hand

    def _start_drag_from_table(pos: Tuple[int, int]) -> bool:
        nonlocal drag, message, drag_offset
        for th in tile_hits:
            if th.rect.collidepoint(pos):
                if not manager.can_move_table_tile(th.meld_idx, th.slot_idx):
                    message = "Déplacement refusé (avant ouverture)."
                    return True
                s = edited_table.melds[th.meld_idx].slots[th.slot_idx]
                drag = DragPayload(
                    source="table",
                    tile_id=s.tile_id,
                    slot=s,
                    meld_idx=th.meld_idx,
                    slot_idx=th.slot_idx,
                )
                drag_offset = (th.rect.x - pos[0], th.rect.y - pos[1])
                return True
        return False

    def _start_drag_from_hand(pos: Tuple[int, int]) -> bool:
        nonlocal drag, drag_offset
        for hc in hand_cells:
            if hc.rect.collidepoint(pos) and hc.count > 0 and hc.tile_id is not None:
                tid = hc.tile_id
                if tid == JOKER_ID:
                    slot = TileSlot(JOKER_ID, assigned_color=None, assigned_value=hand_joker_value)
                else:
                    slot = TileSlot(tid)
                drag = DragPayload(source="hand", tile_id=tid, slot=slot, hand_cell=(hc.row, hc.col))
                drag_offset = (-int(tile_w * 0.4), -int(tile_h * 0.45))
                return True
        return False

    def _apply_drop(drop_target: Optional[Tuple[str, int, int]], dropped_to_hand: bool) -> None:
        nonlocal drag, message, pending_draw_confirm, edited_table, manager
        if drag is None:
            return
        if dropped_to_hand and drag.source == "table":
            if drag.meld_idx is not None and drag.slot_idx is not None and manager.slot_can_return_to_hand(drag.meld_idx, drag.slot_idx):
                manager.remove_slot_from_table(drag.meld_idx, drag.slot_idx)
                edited_table = manager.edited_table
                message = "Tuile rendue à la main."
            else:
                message = "Retour à la main refusé (tuile du plateau initial)."
            return

        if drop_target is None:
            if drag.source == "hand" and selected_target is not None:
                try_place_from_hand(drag.tile_id)
            return

        if drag.source == "hand":
            slot = drag.slot
            if slot.tile_id == JOKER_ID:
                slot = manager.adapt_slot_for_target(slot, drop_target)
            ok, why = manager.insert_slot_into_target(slot, drop_target)
            edited_table = manager.edited_table
            message = "Tuile placée." if ok else f"Placement refusé: {why}"
            if ok:
                pending_draw_confirm = False
            return

        assert drag.meld_idx is not None and drag.slot_idx is not None
        ok, move_message = manager.move_table_slot(drag.meld_idx, drag.slot_idx, drop_target)
        edited_table = manager.edited_table
        message = move_message

    def cycle_table_joker_value(meld_idx: int, slot_idx: int):
        nonlocal edited_table, message
        m = edited_table.melds[meld_idx]
        s = m.slots[slot_idx]
        if s.tile_id != JOKER_ID:
            return
        new_v = 1 if (s.assigned_value is None) else (s.assigned_value % 13) + 1
        new_c = 0 if s.assigned_color is None else s.assigned_color
        m.slots[slot_idx] = TileSlot(JOKER_ID, assigned_color=new_c, assigned_value=new_v)
        message = f"Joker => valeur {new_v}"

    # --- loop ------------------------------------------------------------------

    try:
        running = True
        while running:
            W, H = screen.get_size()
            screen.fill(BG)

            margin = 10
            header_h = 56
            status_h = 32
            hand_h = max(180, min(260, int(H * 0.28)))
            board_h = H - header_h - hand_h - status_h - 4 * margin
            if board_h < 160:
                board_h = 160
                hand_h = max(140, H - header_h - board_h - status_h - 4 * margin)

            header = pygame.Rect(margin, margin, W - 2 * margin, header_h)
            board = pygame.Rect(margin, header.bottom + margin, W - 2 * margin, board_h)
            hand_panel = pygame.Rect(margin, board.bottom + margin, W - 2 * margin, hand_h)
            status = pygame.Rect(margin, hand_panel.bottom + margin, W - 2 * margin, status_h)

            # Header
            pygame.draw.rect(screen, PANEL, header, border_radius=10)
            pygame.draw.rect(screen, PANEL_LINE, header, width=2, border_radius=10)
            screen.blit(title_font.render("Rummikub", True, TEXT), (header.x + 12, header.y + 12))

            # Buttons
            buttons = [("Play (Enter)", "play"), ("Draw (D)", "draw"), ("Reset (R)", "reset"), ("Debug (G)", "debug"), ("Help (H)", "help")]
            bx = header.x + 180
            by = header.y + 10
            bh = 36
            gap = 10
            bw_map = {"play": 150, "draw": 120, "reset": 130, "debug": 130, "help": 120}
            btn_rects: Dict[str, pygame.Rect] = {}
            line_right = header.right - 10
            for label, key in buttons:
                bw = bw_map[key]
                if bx + bw > line_right:
                    bx = header.x + 180
                    by += bh + 8
                r = pygame.Rect(bx, by, bw, bh)
                btn_rects[key] = r
                draw_button(screen, r, label, font)
                bx += bw + gap

            # Board split
            left_w = int(board.width * 0.56)
            right_w = board.width - left_w - margin
            runs_panel = pygame.Rect(board.x, board.y, left_w, board.height)
            groups_panel = pygame.Rect(runs_panel.right + margin, board.y, right_w, board.height)
            draw_panel(screen, runs_panel)
            draw_panel(screen, groups_panel)

            base_table = current().table.canonicalize()
            base_counts, _ = _compute_new_occurrence_cutoff(base_table, edited_table)
            seen_for_new = [0] * len(base_counts)

            row_map = map_runs_rows_to_meld_indices(edited_table)
            group_map = map_groups_cells_to_meld_indices(edited_table)
            nblocks = group_block_count(edited_table)
            tile_hits = []
            meld_bounds: Dict[int, pygame.Rect] = {}
            meld_tile_bounds: Dict[int, pygame.Rect] = {}

            # RUNS grid
            runs_row_hits = []
            grid_x = runs_panel.x + 12
            grid_y = runs_panel.y + 20
            grid_w = runs_panel.width - 24
            grid_h = runs_panel.height - 28
            rows, cols = 8, 13
            cell_w = grid_w / cols
            cell_h = grid_h / rows
            tile_w = max(22, int(cell_w * 0.90))
            tile_h = max(30, int(cell_h * 0.90))

            if selected_target and selected_target[0] == "run":
                sel_row = selected_target[1]
                hl = pygame.Rect(int(grid_x), int(grid_y + sel_row * cell_h), int(grid_w), int(cell_h))
                pygame.draw.rect(screen, (40, 50, 66), hl)

            for r in range(rows + 1):
                y = grid_y + r * cell_h
                pygame.draw.line(screen, PANEL_LINE, (grid_x, y), (grid_x + grid_w, y), 1)
            for c in range(cols + 1):
                x = grid_x + c * cell_w
                pygame.draw.line(screen, PANEL_LINE, (x, grid_y), (x, grid_y + grid_h), 1)

            for c in range(cols):
                screen.blit(axis_font.render(str(c + 1), True, SUB), (int(grid_x + c * cell_w + 3), runs_panel.y + 4))

            for r in range(rows):
                rr = pygame.Rect(int(grid_x), int(grid_y + r * cell_h), int(grid_w), int(cell_h))
                runs_row_hits.append(RunsRowHit(rect=rr, row=r))
                sw = pygame.Rect(runs_panel.x + 4, int(grid_y + r * cell_h + 4), 6, int(cell_h - 8))
                pygame.draw.rect(screen, COLOR_PALETTE[r // 2], sw, border_radius=3)

            # Draw run tiles by value column
            for row, meld_idxs in row_map.items():
                for meld_idx in meld_idxs:
                    meld = edited_table.melds[meld_idx]
                    for slot_idx, s in enumerate(meld.slots):
                        v = _tile_value(s)
                        if not v:
                            continue
                        col = _clamp(v - 1, 0, 12)
                        cx = grid_x + col * cell_w + (cell_w - tile_w) / 2
                        cy = grid_y + row * cell_h + (cell_h - tile_h) / 2
                        rect = pygame.Rect(int(cx), int(cy), tile_w, tile_h)

                        is_new = _is_slot_new_by_occurrence(s, base_counts, seen_for_new)
                        border = ACCENT if is_new else None
                        draw_tile(screen, rect, s, small_font, highlight_border=border)
                        tile_hits.append(TileHit(rect=rect, meld_idx=meld_idx, slot_idx=slot_idx))
                        if meld_idx in meld_bounds:
                            meld_bounds[meld_idx].union_ip(rect)
                        else:
                            meld_bounds[meld_idx] = rect.copy()
                        if meld_idx in meld_tile_bounds:
                            meld_tile_bounds[meld_idx].union_ip(rect)
                        else:
                            meld_tile_bounds[meld_idx] = rect.copy()

            # GROUPS blocks
            groups_col_hits = []
            g_x = groups_panel.x + 12
            g_y = groups_panel.y + 20
            g_w = groups_panel.width - 24
            g_h = groups_panel.height - 28
            block_h = g_h / nblocks if nblocks > 0 else g_h
            rh = block_h / 4.0
            col_w = g_w / 13.0

            if selected_target and selected_target[0] == "group":
                sel_block, sel_val = selected_target[1], selected_target[2]
                hl = pygame.Rect(int(g_x + sel_val * col_w), int(g_y + sel_block * block_h), int(col_w), int(block_h))
                pygame.draw.rect(screen, (40, 50, 66), hl)

            for c in range(13):
                screen.blit(axis_font.render(str(c + 1), True, SUB), (int(g_x + c * col_w + 3), groups_panel.y + 4))

            for b in range(nblocks):
                top = g_y + b * block_h
                for c in range(13):
                    cr = pygame.Rect(int(g_x + c * col_w), int(top), int(col_w), int(block_h))
                    groups_col_hits.append(GroupsColHit(rect=cr, block=b, value_idx=c))
                for r in range(4 + 1):
                    y = top + r * rh
                    pygame.draw.line(screen, PANEL_LINE, (g_x, y), (g_x + g_w, y), 1)
                for c in range(13 + 1):
                    x = g_x + c * col_w
                    pygame.draw.line(screen, PANEL_LINE, (x, top), (x, top + block_h), 1)
                for r in range(4):
                    sw = pygame.Rect(groups_panel.x + 4, int(top + r * rh + 4), 6, int(rh - 8))
                    pygame.draw.rect(screen, COLOR_PALETTE[r], sw, border_radius=3)

            for (block, val_idx), meld_idx in group_map.items():
                top = g_y + block * block_h
                meld = edited_table.melds[meld_idx]
                for slot_idx, s in enumerate(meld.slots):
                    v = _tile_value(s)
                    if v is None:
                        continue
                    display_val_idx = _clamp(v - 1, 0, 12)
                    c = _tile_color_index(s)
                    if c is None:
                        c = 0
                    cx = g_x + display_val_idx * col_w + (col_w - tile_w) / 2
                    cy = top + c * rh + (rh - tile_h) / 2
                    rect = pygame.Rect(int(cx), int(cy), tile_w, tile_h)

                    is_new = _is_slot_new_by_occurrence(s, base_counts, seen_for_new)
                    border = ACCENT if is_new else None
                    draw_tile(screen, rect, s, small_font, highlight_border=border)
                    tile_hits.append(TileHit(rect=rect, meld_idx=meld_idx, slot_idx=slot_idx))
                    if meld_idx in meld_bounds:
                        meld_bounds[meld_idx].union_ip(rect)
                    else:
                        meld_bounds[meld_idx] = rect.copy()
                    if meld_idx in meld_tile_bounds:
                        meld_tile_bounds[meld_idx].union_ip(rect)
                    else:
                        meld_tile_bounds[meld_idx] = rect.copy()

            # HAND grid
            draw_panel(screen, hand_panel)
            screen.blit(font.render("Hand", True, SUB), (hand_panel.x + 10, hand_panel.y + 6))

            hx = hand_panel.x + 12
            hy = hand_panel.y + 26
            hw = hand_panel.width - 24
            hh = hand_panel.height - 36
            hrows, hcols = 4, 14
            cw = hw / hcols
            ch = hh / hrows
            tw = max(22, int(cw * 0.90))
            th = max(30, int(ch * 0.90))

            for r in range(hrows + 1):
                y = hy + r * ch
                pygame.draw.line(screen, PANEL_LINE, (hx, y), (hx + hw, y), 1)
            for c in range(hcols + 1):
                x = hx + c * cw
                pygame.draw.line(screen, PANEL_LINE, (x, hy), (x, hy + hh), 1)

            for r in range(4):
                sw = pygame.Rect(hand_panel.x + 4, int(hy + r * ch + 4), 6, int(ch - 8))
                pygame.draw.rect(screen, COLOR_PALETTE[r], sw, border_radius=3)

            for c in range(13):
                screen.blit(axis_font.render(str(c + 1), True, SUB), (int(hx + c * cw + 3), hand_panel.y + 6))
            screen.blit(axis_font.render("J", True, SUB), (int(hx + 13 * cw + 3), hand_panel.y + 6))
            j_val_lbl = small_font.render(f"J={hand_joker_value}", True, SUB)
            screen.blit(j_val_lbl, (hand_panel.right - 10 - j_val_lbl.get_width(), hand_panel.y + 6))

            remaining = remaining_hand_after_edit(current(), edited_table)
            grid = _hand_grid_counts(remaining)
            hand_cells = []

            # last drawn highlight (if applicable)
            last_drawn_tile = _last_drawn_tile_for_player(current(), current().current_player)

            for r in range(hrows):
                for c in range(hcols):
                    count = grid.get((r, c), 0)
                    rep: Optional[int] = None
                    if count > 0:
                        rep = (JOKER_ID if c == 13 else (r * 13 + c))
                    cell_rect = pygame.Rect(int(hx + c * cw), int(hy + r * ch), int(cw), int(ch))
                    hand_cells.append(HandCell(rect=cell_rect, row=r, col=c, tile_id=rep, count=count))

                    if count > 0 and rep is not None:
                        main = cell_rect.inflate(int(-cw * 0.10), int(-ch * 0.10))
                        if count > 1:
                            back = main.move(6, 5)
                            back_slot = TileSlot(JOKER_ID, assigned_color=None, assigned_value=hand_joker_value) if rep == JOKER_ID else TileSlot(rep)
                            draw_tile(screen, back, back_slot, small_font, stacked_back=True)

                        slot = TileSlot(JOKER_ID, assigned_color=None, assigned_value=hand_joker_value) if rep == JOKER_ID else TileSlot(rep)

                        border = None
                        if last_drawn_tile == rep:
                            border = HILITE_NEW

                        # If dragging from this cell, hide the tile (visual)
                        if drag and drag.source == "hand" and drag.tile_id == rep and drag.hand_cell == (r, c):
                            # draw nothing here (ghost will be drawn)
                            pass
                        else:
                            draw_tile(screen, main, slot, small_font, highlight_border=border, count_badge=count)

            # Opening indicator (status left)
            p = current().current_player
            is_opened = current().initial_meld_done[p]
            draft_pts = _draft_points_toward_opening(base_table, edited_table)
            opening_txt = ""
            if not is_opened:
                opening_txt = f"Ouverture: {draft_pts}/30"
            else:
                opening_txt = "Ouvert"

            # Status bar
            right = f"À jouer : Joueur {p + 1}  |  {opening_txt}"
            status_bar(screen, status, message, right, small_font)

            if meld_bounds:
                for meld_idx, rect in meld_bounds.items():
                    outline = rect.inflate(6, 6)
                    if meld_idx in invalid_melds:
                        pygame.draw.rect(screen, ERR, outline, width=3, border_radius=8)
                    elif show_debug:
                        pygame.draw.rect(screen, (255, 255, 255), outline, width=2, border_radius=8)

            # Debug/Godmode overlay (restored)
            if show_debug:
                dbg = pygame.Rect(margin, header.bottom + margin, int(W * 0.45), int(H * 0.55))
                pygame.draw.rect(screen, (20, 26, 34), dbg, border_radius=10)
                pygame.draw.rect(screen, (90, 98, 118), dbg, width=2, border_radius=10)
                screen.blit(font.render("Godmode — full visibility", True, TEXT), (dbg.x + 12, dbg.y + 8))
                screen.blit(small_font.render("Scroll: mouse wheel", True, SUB), (dbg.x + 12, dbg.y + 28))

                pad = 12
                clip = pygame.Rect(dbg.x + pad, dbg.y + 52, dbg.width - 2 * pad, dbg.height - 62)

                # gather deck info
                deck_order = getattr(current(), "deck_order", None)
                deck_index = getattr(current(), "deck_index", None)
                if isinstance(deck_order, list) and isinstance(deck_index, int):
                    deck_remaining = deck_order[deck_index:]
                    deck_head = deck_remaining[:20]
                    deck_txt = f"Deck remaining: {len(deck_remaining)} (top {len(deck_head)} shown)"
                else:
                    # fallback: if there's a 'deck' list
                    deck_list = getattr(current(), "deck", None)
                    if isinstance(deck_list, list):
                        deck_remaining = deck_list
                        deck_head = deck_remaining[:20]
                        deck_txt = f"Deck remaining: {len(deck_remaining)} (top {len(deck_head)} shown)"
                    else:
                        deck_remaining = []
                        deck_head = []
                        deck_txt = "Deck: (unknown structure)"

                # build lines with mini tiles
                content: List[Tuple[str, Optional[List[int]]]] = []
                content.append((f"timeline={time_idx+1}/{len(timeline)}", None))
                content.append((f"current_player={p+1}", None))
                content.append((f"draft_points={draft_pts}", None))
                content.append(("", None))

                hands = getattr(current(), "hands", [])
                for i, h in enumerate(hands):
                    total = h.total() if hasattr(h, "total") else _total_tiles(h)
                    marker = "  ← current" if i == p else ""
                    content.append((f"P{i+1} hand ({total} tiles){marker}", None))
                    # show as tile ids expanded (limited)
                    # If engine provides counts, expand into tile ids for preview
                    ids: List[int] = []
                    if hasattr(h, "counts"):
                        counts = h.counts
                        for tid, cnt in enumerate(counts):
                            ids.extend([tid] * min(cnt, 2))  # cap per tile to keep compact
                        ids = ids[:60]
                    content.append(("", ids))

                content.append((deck_txt, deck_head))

                # measure and render with scrolling
                lh = small_font.get_height() + 2
                # dynamic height: lines + tile rows
                # We'll render, measuring on the fly.
                max_scroll = 10**9  # computed during render

                prev = screen.get_clip()
                screen.set_clip(clip)

                y = clip.y - debug_scroll
                tile_size = 22
                tile_gap = 6
                max_width = clip.width
                for header_line, tile_list in content:
                    if header_line:
                        screen.blit(small_font.render(header_line, True, (210, 216, 225)), (clip.x, y))
                        y += lh
                    if tile_list is not None:
                        x = clip.x
                        row_h = tile_size + 6
                        for tid in tile_list:
                            # wrap
                            if x + tile_size > clip.x + max_width:
                                x = clip.x
                                y += row_h
                            if tid == JOKER_ID:
                                slot = TileSlot(JOKER_ID, assigned_color=None, assigned_value=1)
                            else:
                                slot = TileSlot(tid)
                            rect = pygame.Rect(x, y, tile_size, int(tile_size * 1.3))
                            draw_tile(screen, rect, slot, small_font, ghost=True)
                            x += tile_size + tile_gap
                        y += row_h + 6

                screen.set_clip(prev)

                # compute max_scroll based on final y
                content_h = (y - clip.y) + debug_scroll
                max_scroll = max(0, int(content_h - clip.height))
                debug_scroll = _clamp(debug_scroll, 0, max_scroll)

            help_close = None
            # Help panel
            if show_help:
                overlay = pygame.Rect(int(W * 0.12), int(H * 0.18), int(W * 0.76), int(H * 0.64))
                draw_panel(screen, overlay)
                screen.blit(title_font.render("Aide", True, TEXT), (overlay.x + 16, overlay.y + 12))
                lines = [
                    "• Glisser/déposer : main ↔ plateau (RUN/GROUP).",
                    "• Clic gauche : sélectionner RUN/GROUP si besoin.",
                    "• Clic droit : cycler valeur du joker (main/plateau).",
                    "• Enter / Play : valider un PLAY.",
                    "• D / Draw : piocher.",
                    "• R / Reset : annuler le draft.",
                    "• G / Debug : godmode.",
                    "• H / Help : fermer ce panneau.",
                ]
                y = overlay.y + 52
                for ln in lines:
                    screen.blit(small_font.render(ln, True, SUB), (overlay.x + 18, y))
                    y += small_font.get_height() + 6
                help_close = pygame.Rect(overlay.right - 140, overlay.bottom - 50, 120, 34)
                draw_button(screen, help_close, "Fermer", font)

            # Draw confirmation panel
            confirm_rect = None
            confirm_yes = None
            confirm_no = None
            if pending_draw_confirm:
                overlay = pygame.Rect(int(W * 0.30), int(H * 0.35), int(W * 0.40), int(H * 0.24))
                confirm_rect = overlay
                draw_panel(screen, overlay)
                screen.blit(title_font.render("Confirmer la pioche ?", True, TEXT), (overlay.x + 16, overlay.y + 16))
                screen.blit(
                    small_font.render("Aucun meld placé : un PLAY déclenchera une pioche.", True, SUB),
                    (overlay.x + 16, overlay.y + 56),
                )
                confirm_yes = pygame.Rect(overlay.x + 20, overlay.bottom - 50, 140, 34)
                confirm_no = pygame.Rect(overlay.right - 160, overlay.bottom - 50, 140, 34)
                draw_button(screen, confirm_yes, "Confirmer", font)
                draw_button(screen, confirm_no, "Annuler", font)

            # Drag ghost tile
            if drag is not None:
                mx, my = pygame.mouse.get_pos()
                ghost_rect = pygame.Rect(mx + drag_offset[0], my + drag_offset[1], int(tile_w * 0.95), int(tile_h * 0.95))
                draw_tile(screen, ghost_rect, drag.slot, small_font, highlight_border=ACCENT, ghost=True)

            # --- Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)

                elif event.type == pygame.MOUSEWHEEL:
                    if show_debug:
                        debug_scroll -= event.y * 24

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_RETURN:
                        if pending_draw_confirm:
                            _perform_draw_action()
                            continue

                        delta = _draft_delta_from_hand(current(), edited_table)
                        if delta is not None and delta.total() == 0 and _is_noop_draft(current(), edited_table):
                            pending_draw_confirm = True
                            message = "Confirmer la pioche."
                            continue

                        move, err = build_play_move(current(), edited_table)
                        if move:
                            nxt = apply_move(current(), move)
                            if time_idx != len(timeline) - 1:
                                timeline[:] = timeline[: time_idx + 1]
                            timeline.append(nxt)
                            time_idx = len(timeline) - 1
                            edited_table = current().table.canonicalize()
                            manager.refresh(state=current(), edited_table=edited_table)
                            pending_draw_confirm = False
                            invalid_melds = set()
                            message = "PLAY effectué."
                        else:
                            message = f"PLAY invalide: {err}"
                            debug_log.append(f"PLAY invalid: {err}")
                            invalid_melds = set()
                            for idx, meld in enumerate(edited_table.melds):
                                ok, _ = meld.is_valid()
                                if not ok:
                                    invalid_melds.add(idx)
                    elif event.key == pygame.K_d:
                        _perform_draw_action()
                    elif event.key == pygame.K_r:
                        reset_draft()
                        message = "Draft réinitialisé."
                    elif event.key == pygame.K_g:
                        show_debug = not show_debug
                        if show_debug:
                            debug_scroll = 10**9
                    elif event.key == pygame.K_h:
                        show_help = not show_help
                        if show_help:
                            message = "Aide affichée."
                        else:
                            message = "Aide fermée."
                    elif event.key == pygame.K_LEFT:
                        if time_idx > 0:
                            time_idx -= 1
                            reset_draft()
                            message = "Historique: précédent."
                    elif event.key == pygame.K_RIGHT:
                        if time_idx < len(timeline) - 1:
                            time_idx += 1
                            reset_draft()
                            message = "Historique: suivant."

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if pending_draw_confirm and confirm_rect:
                            if confirm_yes and confirm_yes.collidepoint(event.pos):
                                _perform_draw_action()
                            elif confirm_no and confirm_no.collidepoint(event.pos):
                                pending_draw_confirm = False
                                message = "Pioche annulée."
                            elif btn_rects["play"].collidepoint(event.pos):
                                _perform_draw_action()
                            continue
                        if show_help:
                            if help_close and help_close.collidepoint(event.pos):
                                show_help = False
                                message = "Aide fermée."
                            continue
                        # Buttons
                        if btn_rects["play"].collidepoint(event.pos):
                            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
                            continue
                        if btn_rects["draw"].collidepoint(event.pos):
                            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d))
                            continue
                        if btn_rects["reset"].collidepoint(event.pos):
                            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
                            continue
                        if btn_rects["debug"].collidepoint(event.pos):
                            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g))
                            continue
                        if btn_rects["help"].collidepoint(event.pos):
                            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_h))
                            continue

                        # Start drag from table tile (prefer table over hand if overlap)
                        if _start_drag_from_table(event.pos):
                            continue

                        # Start drag from hand cell (pick representative tile id)
                        if _start_drag_from_hand(event.pos):
                            continue

                        # Click-select target (fallback interaction)
                        started = False
                        for rr in runs_row_hits:
                            if rr.rect.collidepoint(event.pos):
                                selected_target = ("run", rr.row, -1)
                                message = f"RUN sélectionné (couleur {rr.row//2 + 1}, slot {rr.row%2 + 1})"
                                started = True
                                break
                        if started:
                            continue
                        for gc in groups_col_hits:
                            if gc.rect.collidepoint(event.pos):
                                selected_target = ("group", gc.block, gc.value_idx)
                                message = f"GROUP sélectionné (valeur {gc.value_idx+1}, bloc {gc.block+1})"
                                break

                    elif event.button == 3:
                        # Right click: hand joker cycles its chosen value (used for runs)
                        for hc in hand_cells:
                            if hc.rect.collidepoint(event.pos) and hc.tile_id == JOKER_ID and hc.count > 0:
                                hand_joker_value = (hand_joker_value % 13) + 1
                                manager.refresh(hand_joker_value=hand_joker_value)
                                message = f"Joker main => valeur {hand_joker_value}"
                                break
                        else:
                            # Right click: table joker cycles its assigned value
                            for th in tile_hits:
                                if th.rect.collidepoint(event.pos):
                                    if edited_table.melds[th.meld_idx].slots[th.slot_idx].tile_id == JOKER_ID:
                                        cycle_table_joker_value(th.meld_idx, th.slot_idx)
                                    break

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and drag is not None:
                        drop_target, dropped_to_hand = _drop_target_from_pos(event.pos)
                        _apply_drop(drop_target, dropped_to_hand)
                        drag = None

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

    except Exception:
        crash_screen(traceback.format_exc())


if __name__ == "__main__":
    launch_gui()
