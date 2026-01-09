
from __future__ import annotations

"""
Compact Rummikub GUI — runs left, groups right, hand as fixed 4x14 grid.

Design constraints enforced at startup:
- colors == 4
- values == 13

Interaction model:
- Select a target meld on the table:
    * RUN target: click a RUN row (left panel)
    * GROUP target: click a GROUP column within a block (right panel)
- Then click a tile in the HAND grid to send one instance of that tile to the selected target,
  only if the local constraints of that meld would still be satisfied.
- Commit with Enter (PLAY), Draw with D, Reset draft with R.
- Debug overlay: toggle with G, scroll with mouse wheel (always reaches bottom).
- Help hint: H.

Jokers:
- In hand: always displayed in the last column (J) on row 0.
- When placing:
    * into a GROUP: joker value is forced to the selected column value; joker color chosen as first missing.
    * into a RUN: joker color forced to row color; joker value uses the current "hand joker value" (right‑click joker in hand to cycle 1..13).
- Right‑click a joker on the TABLE cycles its assigned value 1..13.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import os
import traceback

# --- Imports: support package and script execution -----------------------------
try:
    from .engine import apply_move, is_legal_move
    from .meld import Meld, MeldKind
    from .move import Move
    from .multiset import TileMultiset
    from .rules import Ruleset
    from .state import GameState, new_game
    from .table import Table
    from .tiles import JOKER_ID, TileSlot
except Exception:  # fallback when executed as a script
    from engine import apply_move, is_legal_move
    from meld import Meld, MeldKind
    from move import Move
    from multiset import TileMultiset
    from rules import Ruleset
    from state import GameState, new_game
    from table import Table
    from tiles import JOKER_ID, TileSlot

try:
    import pygame  # type: ignore
except Exception:  # pragma: no cover
    pygame = None  # type: ignore


# --- Pure helpers (adapted from the original gui.py) ---------------------------

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


def build_play_move(state: GameState, edited_table: Table) -> Tuple[Optional[Move], str]:
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


# --- Theme --------------------------------------------------------------------

BG = (22, 27, 34)
PANEL = (30, 36, 46)
PANEL_LINE = (54, 63, 77)
TEXT = (220, 226, 235)
SUB = (164, 174, 187)
ACCENT = (88, 138, 255)
ERR = (235, 87, 87)
WARN = (255, 170, 40)

COLOR_PALETTE = [
    (220, 80, 80),   # red
    (86, 148, 227),  # blue
    (66, 171, 119),  # green
    (221, 164, 66),  # yellow/orange
]


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _tile_value(slot: TileSlot) -> Optional[int]:
    if slot.tile_id == JOKER_ID:
        return slot.assigned_value
    return (slot.tile_id % 13) + 1


def _tile_color_index(slot: TileSlot) -> Optional[int]:
    if slot.tile_id == JOKER_ID:
        return slot.assigned_color
    return (slot.tile_id // 13) % 4


def _ms_total(ms: TileMultiset) -> int:
    return sum(ms.counts)


def _hand_grid(current_hand: TileMultiset) -> Dict[Tuple[int, int], int]:
    """Return mapping (row,col)->count for a fixed 4x14 grid.

    Rows 0..3 correspond to colors (R,B,G,Y). Cols 0..12 correspond to values 1..13. Col 13 is Jokers.
    """
    grid: Dict[Tuple[int, int], int] = {}
    for tid, cnt in enumerate(current_hand.counts):
        if cnt <= 0:
            continue
        if tid == JOKER_ID:
            grid[(0, 13)] = grid.get((0, 13), 0) + cnt
        else:
            color = (tid // 13) % 4
            value_idx = (tid % 13)  # 0..12
            grid[(color, value_idx)] = grid.get((color, value_idx), 0) + cnt
    return grid


# --- Hit testing ---------------------------------------------------------------

@dataclass
class HandCell:
    rect: pygame.Rect
    row: int
    col: int
    tile_id: Optional[int]  # representative id (if count>0), else None
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
    # Determine run color from first non-joker tile; fallback to assigned_color of a joker; else 0
    for s in m.slots:
        if s.tile_id != JOKER_ID:
            return (s.tile_id // 13) % 4
    for s in m.slots:
        if s.tile_id == JOKER_ID and s.assigned_color is not None:
            return s.assigned_color
    return 0


def map_runs_rows_to_meld_indices(table: Table) -> Dict[int, int]:
    """Map run melds to the 8 run rows (2 per color)."""
    rows: Dict[int, int] = {}
    per_color: Dict[int, List[int]] = {0: [], 1: [], 2: [], 3: []}
    for idx, m in enumerate(table.melds):
        if m.kind != MeldKind.RUN or not m.slots:
            continue
        c = _run_color_for_meld(m)
        per_color.setdefault(c, []).append(idx)
    for color in range(4):
        idxs = per_color.get(color, [])
        for k, meld_idx in enumerate(idxs[:2]):
            row = color * 2 + k
            rows[row] = meld_idx
    return rows


def groups_by_value(table: Table) -> Dict[int, List[int]]:
    """Map value_idx 0..12 -> group meld indices for that value."""
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
    """Number of 4x13 blocks required so each value can have enough group slots.

    Default minimum is 2 blocks, but if any value has >2 groups, grow to max_per_value.
    """
    by_val = groups_by_value(table)
    max_per_value = max((len(v) for v in by_val.values()), default=0)
    return max(2, max_per_value)


def map_groups_cells_to_meld_indices(table: Table) -> Dict[Tuple[int, int], int]:
    """Map (block,value_idx)->meld_idx for group melds, block being the occurrence rank for that value."""
    mapping: Dict[Tuple[int, int], int] = {}
    by_val = groups_by_value(table)
    for val_idx, melds in by_val.items():
        for block, meld_idx in enumerate(melds):
            mapping[(block, val_idx)] = meld_idx
    return mapping


# --- Local placement constraints (for "si c'est autorisé") ---------------------

def _run_values_and_colors(slots: List[TileSlot]) -> Tuple[List[int], List[int]]:
    vals: List[int] = []
    cols: List[int] = []
    for s in slots:
        v = _tile_value(s)
        c = _tile_color_index(s)
        if v is None or c is None:
            continue
        vals.append(v)
        cols.append(c)
    return vals, cols


def can_insert_into_run(run_slots: List[TileSlot], new_slot: TileSlot, row_color: int) -> Tuple[bool, str]:
    # enforce color consistency (non-jokers) and assigned_color
    if new_slot.tile_id == JOKER_ID:
        if new_slot.assigned_color != row_color:
            return False, "joker color mismatch"
    else:
        if ((new_slot.tile_id // 13) % 4) != row_color:
            return False, "tile color mismatch with run row"

    candidate = list(run_slots) + [new_slot]
    # all concrete colors must be the row color
    for s in candidate:
        if s.tile_id != JOKER_ID:
            if ((s.tile_id // 13) % 4) != row_color:
                return False, "run mixed colors"
        else:
            if s.assigned_color != row_color:
                return False, "run joker color mismatch"

    # values: allow len<2 freely; else require "consecutive set" using assigned joker values
    vals: List[int] = []
    for s in candidate:
        v = _tile_value(s)
        if v is None:
            return False, "joker has no value"
        vals.append(v)
    if len(vals) <= 1:
        return True, ""
    if len(set(vals)) != len(vals):
        return False, "duplicate value in run"
    vals_sorted = sorted(vals)
    if vals_sorted[-1] - vals_sorted[0] != len(vals_sorted) - 1:
        return False, "run not consecutive"
    return True, ""


def can_insert_into_group(group_slots: List[TileSlot], new_slot: TileSlot, value_idx: int) -> Tuple[bool, str]:
    target_val = value_idx + 1
    candidate = list(group_slots) + [new_slot]
    if len(candidate) > 4:
        return False, "group too large"

    # enforce values all equal to target_val
    for s in candidate:
        v = _tile_value(s)
        if v is None:
            return False, "joker has no value"
        if v != target_val:
            return False, "group value mismatch"

    # enforce distinct colors for non-jokers and assigned colors for jokers
    colors: List[int] = []
    for s in candidate:
        c = _tile_color_index(s)
        if c is None:
            return False, "joker has no color"
        colors.append(c)
    if len(set(colors)) != len(colors):
        return False, "duplicate color in group"
    return True, ""


# --- Drawing ------------------------------------------------------------------

def draw_panel(surface, rect: pygame.Rect, title: Optional[str], font):
    pygame.draw.rect(surface, PANEL, rect, border_radius=10)
    pygame.draw.rect(surface, PANEL_LINE, rect, width=2, border_radius=10)
    if title:
        label = font.render(title, True, SUB)
        surface.blit(label, (rect.x + 10, rect.y + 6))


def draw_button(surface, rect: pygame.Rect, label: str, font, enabled: bool = True):
    bg = (46, 56, 69) if enabled else (35, 42, 52)
    border = ACCENT if enabled else (60, 70, 82)
    pygame.draw.rect(surface, bg, rect, border_radius=10)
    pygame.draw.rect(surface, border, rect, width=2, border_radius=10)
    txt = font.render(label, True, TEXT if enabled else (120, 130, 140))
    surface.blit(txt, txt.get_rect(center=rect.center))


def draw_tile(surface, rect: pygame.Rect, slot: TileSlot, small_font, highlight=False, count_badge: Optional[int] = None):
    radius = 6
    border = ACCENT if highlight else (60, 70, 85)
    pygame.draw.rect(surface, (245, 245, 245), rect, border_radius=radius)
    pygame.draw.rect(surface, border, rect, width=2, border_radius=radius)

    inner = rect.inflate(-8, -8)
    v = _tile_value(slot)
    label = f"J{v}" if slot.tile_id == JOKER_ID else str(v if v is not None else "?")

    font_size = max(12, int(inner.height * 0.52))
    font = pygame.font.SysFont("arial", font_size, bold=True)
    txt = font.render(label, True, (40, 40, 50))
    surface.blit(txt, txt.get_rect(center=inner.center))

    if count_badge and count_badge > 1:
        badge = small_font.render(f"×{count_badge}", True, (245, 245, 245))
        badge_bg = pygame.Rect(rect.right - 20, rect.top + 4, 16, 16)
        pygame.draw.rect(surface, (55, 65, 85), badge_bg, border_radius=8)
        surface.blit(badge, badge.get_rect(center=badge_bg.center))


def status_bar(surface, rect: pygame.Rect, left: str, right: str, small_font, color=SUB):
    pygame.draw.rect(surface, PANEL, rect, border_radius=10)
    pygame.draw.rect(surface, PANEL_LINE, rect, width=2, border_radius=10)
    l = small_font.render(left, True, color)
    r = small_font.render(right, True, color)
    surface.blit(l, (rect.x + 10, rect.y + (rect.height - l.get_height()) // 2))
    surface.blit(r, (rect.right - 10 - r.get_width(), rect.y + (rect.height - r.get_height()) // 2))


# --- GUI entry ----------------------------------------------------------------

def launch_gui(seed: Optional[int] = None, ruleset: Optional[Ruleset] = None) -> None:  # pragma: no cover
    if pygame is None:
        raise ImportError("pygame is required for the GUI. Install it with `pip install pygame`.")

    ruleset = ruleset or Ruleset()
    if getattr(ruleset, "colors", 4) != 4 or getattr(ruleset, "values", 13) != 13:
        raise RuntimeError("This compact GUI only supports exactly 4 colors and 13 values (Rummikub standard).")

    pygame.init()
    pygame.display.set_caption("Rummikub — Compact GUI")
    screen = pygame.display.set_mode((1280, 768), pygame.RESIZABLE)
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("arial", 18)
    title_font = pygame.font.SysFont("arial", 24, bold=True)
    small_font = pygame.font.SysFont("arial", 14)

    # Game state
    state = new_game(ruleset=ruleset, rng_seed=seed)
    timeline: List[GameState] = [state]
    time_idx = 0

    def current() -> GameState:
        return timeline[time_idx]

    edited_table = current().table.canonicalize()

    # UI state
    selected_target: Optional[Tuple[str, int, int]] = None  # ('run', row, -1) or ('group', block, value_idx)
    message = "Sélectionnez un RUN (gauche) ou un GROUP (droite), puis cliquez une tuile de la main pour l'envoyer si autorisé."
    show_debug = False
    debug_scroll = 10**9
    debug_log: List[str] = []
    hand_joker_value = 1  # used when placing joker into a run

    # Hit regions
    hand_cells: List[HandCell] = []
    runs_row_hits: List[RunsRowHit] = []
    groups_col_hits: List[GroupsColHit] = []
    tile_hits: List[TileHit] = []

    # --- Helpers for mutation --------------------------------------------------

    def reset_draft():
        nonlocal edited_table, selected_target
        edited_table = current().table.canonicalize()
        selected_target = None

    def append_debug(s: str):
        debug_log.append(s)

    def make_slot_for_target(tile_id: int, target: Tuple[str, int, int], table: Table) -> TileSlot:
        """Create a TileSlot instance appropriate for the placement target."""
        kind, a, b = target
        if tile_id != JOKER_ID:
            return TileSlot(tile_id)

        if kind == "group":
            block, value_idx = a, b
            target_val = value_idx + 1
            # choose first missing color in that group meld (if any)
            mapping = map_groups_cells_to_meld_indices(table)
            meld_idx = mapping.get((block, value_idx))
            used = set()
            if meld_idx is not None:
                for s in table.melds[meld_idx].slots:
                    c = _tile_color_index(s)
                    if c is not None:
                        used.add(c)
            color = next((c for c in range(4) if c not in used), 0)
            return TileSlot(JOKER_ID, assigned_color=color, assigned_value=target_val)

        # run
        row = a
        row_color = row // 2
        return TileSlot(JOKER_ID, assigned_color=row_color, assigned_value=hand_joker_value)

    def try_place_from_hand(tile_id: int):
        nonlocal edited_table, message
        if selected_target is None:
            message = "Aucun meld sélectionné."
            return

        slot = make_slot_for_target(tile_id, selected_target, edited_table)

        kind, a, b = selected_target
        if kind == "run":
            row = a
            row_color = row // 2
            row_map = map_runs_rows_to_meld_indices(edited_table)
            meld_idx = row_map.get(row)
            if meld_idx is None:
                # new meld with a single tile is allowed as draft
                ok, why = can_insert_into_run([], slot, row_color)
                if not ok:
                    message = f"Placement refusé: {why}"
                    return
                edited_table.melds.append(Meld(kind=MeldKind.RUN, slots=[slot]))
                message = "Tuile placée dans RUN."
                return
            ok, why = can_insert_into_run(edited_table.melds[meld_idx].slots, slot, row_color)
            if not ok:
                message = f"Placement refusé: {why}"
                return
            edited_table.melds[meld_idx].slots.append(slot)
            message = "Tuile placée dans RUN."
            return

        # group
        block, value_idx = a, b
        mapping = map_groups_cells_to_meld_indices(edited_table)
        meld_idx = mapping.get((block, value_idx))
        if meld_idx is None:
            ok, why = can_insert_into_group([], slot, value_idx)
            if not ok:
                message = f"Placement refusé: {why}"
                return
            edited_table.melds.append(Meld(kind=MeldKind.GROUP, slots=[slot]))
            message = "Tuile placée dans GROUP."
            return
        ok, why = can_insert_into_group(edited_table.melds[meld_idx].slots, slot, value_idx)
        if not ok:
            message = f"Placement refusé: {why}"
            return
        edited_table.melds[meld_idx].slots.append(slot)
        message = "Tuile placée dans GROUP."

    def cycle_table_joker_value(meld_idx: int, slot_idx: int):
        nonlocal edited_table, message
        m = edited_table.melds[meld_idx]
        s = m.slots[slot_idx]
        if s.tile_id != JOKER_ID:
            return
        new_v = 1 if (s.assigned_value is None) else (s.assigned_value % 13) + 1
        # keep assigned_color
        new_c = 0 if s.assigned_color is None else s.assigned_color
        m.slots[slot_idx] = TileSlot(JOKER_ID, assigned_color=new_c, assigned_value=new_v)
        message = f"Joker => valeur {new_v}"

    # --- Error screen wrapper: prevents "silent close" -------------------------

    def run_loop():
        nonlocal screen, time_idx, edited_table, selected_target, message
        nonlocal show_debug, debug_scroll, hand_joker_value
        nonlocal hand_cells, runs_row_hits, groups_col_hits, tile_hits

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

            # --- Header
            pygame.draw.rect(screen, PANEL, header, border_radius=10)
            pygame.draw.rect(screen, PANEL_LINE, header, width=2, border_radius=10)
            screen.blit(title_font.render("Rummikub", True, TEXT), (header.x + 12, header.y + 12))

            # Buttons (no Pass, no +Meld)
            # Robust layout: wrap buttons into available space
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
                draw_button(screen, r, label, font, enabled=True)
                bx += bw + gap

            # Update header height if it wrapped (visual only; panels already placed)
            # (We accept that wrapped header may overlap board for very small windows; user asked collisions must disappear)
            # So: if we wrapped, we simply draw over header area; the board starts below original header rect.

            # --- Board split: left RUNS, right GROUPS
            left_w = int(board.width * 0.56)
            right_w = board.width - left_w - margin
            runs_panel = pygame.Rect(board.x, board.y, left_w, board.height)
            groups_panel = pygame.Rect(runs_panel.right + margin, board.y, right_w, board.height)

            draw_panel(screen, runs_panel, "RUNS", font)
            draw_panel(screen, groups_panel, "GROUPS", font)

            # Build mapping
            row_map = map_runs_rows_to_meld_indices(edited_table)
            group_map = map_groups_cells_to_meld_indices(edited_table)
            nblocks = group_block_count(edited_table)

            # Tile hit list (for right-click joker)
            tile_hits = []

            # --- Runs grid (8x13)
            runs_row_hits = []
            grid_x = runs_panel.x + 12
            grid_y = runs_panel.y + 24
            grid_w = runs_panel.width - 24
            grid_h = runs_panel.height - 36
            rows, cols = 8, 13
            cell_w = grid_w / cols
            cell_h = grid_h / rows
            tile_w = max(22, int(cell_w * 0.78))
            tile_h = max(30, int(cell_h * 0.78))

            # highlight selected row background
            if selected_target and selected_target[0] == "run":
                sel_row = selected_target[1]
                hl = pygame.Rect(grid_x, grid_y + sel_row * cell_h, grid_w, cell_h)
                pygame.draw.rect(screen, (40, 50, 66), hl)

            # grid lines
            for r in range(rows + 1):
                y = grid_y + r * cell_h
                pygame.draw.line(screen, PANEL_LINE, (grid_x, y), (grid_x + grid_w, y), 1)
            for c in range(cols + 1):
                x = grid_x + c * cell_w
                pygame.draw.line(screen, PANEL_LINE, (x, grid_y), (x, grid_y + grid_h), 1)

            # Row hit rects
            for r in range(rows):
                rr = pygame.Rect(grid_x, int(grid_y + r * cell_h), int(grid_w), int(cell_h))
                runs_row_hits.append(RunsRowHit(rect=rr, row=r))

            # Draw run tiles placed by value (column)
            for row, meld_idx in row_map.items():
                meld = edited_table.melds[meld_idx]
                for slot_idx, s in enumerate(meld.slots):
                    v = _tile_value(s)
                    if not v:
                        continue
                    col = v - 1
                    cx = grid_x + col * cell_w + (cell_w - tile_w) / 2
                    cy = grid_y + row * cell_h + (cell_h - tile_h) / 2
                    rect = pygame.Rect(int(cx), int(cy), tile_w, tile_h)
                    draw_tile(screen, rect, s, small_font, highlight=False)
                    tile_hits.append(TileHit(rect=rect, meld_idx=meld_idx, slot_idx=slot_idx))

            # --- Groups grid: blocks stacked vertically, each block 4x13
            g_x = groups_panel.x + 12
            g_y = groups_panel.y + 24
            g_w = groups_panel.width - 24
            g_h = groups_panel.height - 36
            block_h = g_h / nblocks if nblocks > 0 else g_h
            rh = block_h / 4.0
            col_w = g_w / 13.0

            groups_col_hits = []
            # highlight selected group column area
            if selected_target and selected_target[0] == "group":
                sel_block, sel_val = selected_target[1], selected_target[2]
                hl = pygame.Rect(int(g_x + sel_val * col_w), int(g_y + sel_block * block_h), int(col_w), int(block_h))
                pygame.draw.rect(screen, (40, 50, 66), hl)

            # grid lines + selectable columns
            for b in range(nblocks):
                top = g_y + b * block_h
                # column hits (value selection per block)
                for c in range(13):
                    cr = pygame.Rect(int(g_x + c * col_w), int(top), int(col_w), int(block_h))
                    groups_col_hits.append(GroupsColHit(rect=cr, block=b, value_idx=c))
                # lines
                for r in range(4 + 1):
                    y = top + r * rh
                    pygame.draw.line(screen, PANEL_LINE, (g_x, y), (g_x + g_w, y), 1)
                for c in range(13 + 1):
                    x = g_x + c * col_w
                    pygame.draw.line(screen, PANEL_LINE, (x, top), (x, top + block_h), 1)

            # Draw group tiles: column = value, row = color
            for (block, val_idx), meld_idx in group_map.items():
                top = g_y + block * block_h
                meld = edited_table.melds[meld_idx]
                used = set()
                # determine used colors to place jokers in unused rows deterministically
                for s in meld.slots:
                    c = _tile_color_index(s)
                    if c is not None:
                        used.add(c)

                for slot_idx, s in enumerate(meld.slots):
                    # value must match val_idx + 1 for a good display; if not, still display at its own value
                    v = _tile_value(s)
                    if v is None:
                        continue
                    display_val_idx = val_idx
                    if v != val_idx + 1:
                        display_val_idx = _clamp(v - 1, 0, 12)

                    c = _tile_color_index(s)
                    if c is None:
                        # pick first unused
                        c = next((k for k in range(4) if k not in used), 0)
                        used.add(c)

                    cx = g_x + display_val_idx * col_w + (col_w - tile_w) / 2
                    cy = top + c * rh + (rh - tile_h) / 2
                    rect = pygame.Rect(int(cx), int(cy), tile_w, tile_h)
                    draw_tile(screen, rect, s, small_font, highlight=False)
                    tile_hits.append(TileHit(rect=rect, meld_idx=meld_idx, slot_idx=slot_idx))

            # --- Hand panel: fixed 4x14 grid with stacking
            draw_panel(screen, hand_panel, "Hand", font)
            hand_cells = []
            hx = hand_panel.x + 12
            hy = hand_panel.y + 30
            hw = hand_panel.width - 24
            hh = hand_panel.height - 42
            hrows, hcols = 4, 14
            cw = hw / hcols
            ch = hh / hrows
            tw = max(22, int(cw * 0.78))
            th = max(30, int(ch * 0.78))

            # grid lines
            for r in range(hrows + 1):
                y = hy + r * ch
                pygame.draw.line(screen, PANEL_LINE, (hx, y), (hx + hw, y), 1)
            for c in range(hcols + 1):
                x = hx + c * cw
                pygame.draw.line(screen, PANEL_LINE, (x, hy), (x, hy + hh), 1)

            remaining = remaining_hand_after_edit(current(), edited_table)
            grid = _hand_grid(remaining)

            # pre-compute representative tile_id per cell
            for r in range(hrows):
                for c in range(hcols):
                    count = grid.get((r, c), 0)
                    rep: Optional[int] = None
                    if count > 0:
                        if c == 13:
                            rep = JOKER_ID
                        else:
                            rep = r * 13 + c
                    cell_rect = pygame.Rect(int(hx + c * cw), int(hy + r * ch), int(cw), int(ch))
                    hand_cells.append(HandCell(rect=cell_rect, row=r, col=c, tile_id=rep, count=count))

                    if count > 0 and rep is not None:
                        # stacking effect
                        main_rect = cell_rect.inflate(int(-cw * 0.18), int(-ch * 0.18))
                        if count > 1:
                            under = main_rect.move(6, 5)
                            pygame.draw.rect(screen, (45, 55, 70), under, border_radius=6)

                        if rep == JOKER_ID:
                            slot = TileSlot(JOKER_ID, assigned_color=0, assigned_value=hand_joker_value)
                        else:
                            slot = TileSlot(rep)
                        draw_tile(screen, main_rect, slot, small_font, highlight=False, count_badge=count)

            # show current hand joker value near joker column header (minimal)
            j_label = small_font.render(f"J={hand_joker_value}", True, SUB)
            screen.blit(j_label, (int(hx + 13 * cw + 6), int(hy - 18)))

            # --- Status bar
            right = f"À jouer : Joueur {current().current_player + 1}"
            status_bar(screen, status, message, right, small_font, color=SUB)

            # --- Debug overlay
            if show_debug:
                dbg_w = int(W * 0.38)
                dbg_h = int(H * 0.42)
                dbg = pygame.Rect(margin, header.bottom + margin, dbg_w, dbg_h)
                pygame.draw.rect(screen, (20, 26, 34), dbg, border_radius=10)
                pygame.draw.rect(screen, (90, 98, 118), dbg, width=2, border_radius=10)
                pad = 8
                clip = pygame.Rect(dbg.x + pad, dbg.y + pad, dbg.width - 2 * pad, dbg.height - 2 * pad)
                # Build lines
                lines = [
                    f"timeline={time_idx+1}/{len(timeline)}",
                    f"turn={getattr(current(), 'turn', '?')}  deck={len(getattr(current(), 'deck', []))}",
                    f"hand={_ms_total(remaining)}",
                    f"table_melds={len(edited_table.melds)}  blocks={nblocks}",
                    f"selected={selected_target}",
                    "",
                    *debug_log[-300:],
                ]
                # Measure content height
                lh = small_font.get_height() + 2
                content_h = lh * len(lines)
                max_scroll = max(0, content_h - clip.height)
                debug_scroll = _clamp(debug_scroll, 0, max_scroll)

                # Draw with clipping
                prev_clip = screen.get_clip()
                screen.set_clip(clip)
                y = clip.y - debug_scroll
                for ln in lines:
                    s = small_font.render(ln, True, (210, 216, 225))
                    screen.blit(s, (clip.x, y))
                    y += lh
                screen.set_clip(prev_clip)

            # --- Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)

                elif event.type == pygame.MOUSEWHEEL:
                    if show_debug:
                        debug_scroll -= event.y * 24  # natural
                    # no other scrolling in this compact UI

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_RETURN:
                        move, err = build_play_move(current(), edited_table)
                        if move:
                            nxt = apply_move(current(), move)
                            # truncate history if not at end
                            nonlocal_time_idx = time_idx  # read
                            if time_idx != len(timeline) - 1:
                                timeline[:] = timeline[: time_idx + 1]
                            timeline.append(nxt)
                            time_idx = len(timeline) - 1
                            edited_table = current().table.canonicalize()
                            message = "PLAY effectué."
                            append_debug("PLAY committed")
                        else:
                            message = f"PLAY invalide: {err}"
                            append_debug(f"PLAY invalid: {err}")
                    elif event.key == pygame.K_d:
                        move = Move.draw()
                        nxt = apply_move(current(), move)
                        if time_idx != len(timeline) - 1:
                            timeline[:] = timeline[: time_idx + 1]
                        timeline.append(nxt)
                        time_idx = len(timeline) - 1
                        edited_table = current().table.canonicalize()
                        message = "Pioche effectuée."
                        append_debug("DRAW committed")
                    elif event.key == pygame.K_r:
                        reset_draft()
                        message = "Draft réinitialisé."
                    elif event.key == pygame.K_g:
                        show_debug = not show_debug
                        if show_debug:
                            debug_scroll = 10**9
                    elif event.key == pygame.K_h:
                        message = "Aide: clic RUN/GROUP pour sélectionner; clic tuile main pour placer; Enter PLAY; D pioche; R reset; clic droit joker: valeur."
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
                        # Buttons
                        if btn_rects["play"].collidepoint(event.pos):
                            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
                        elif btn_rects["draw"].collidepoint(event.pos):
                            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d))
                        elif btn_rects["reset"].collidepoint(event.pos):
                            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
                        elif btn_rects["debug"].collidepoint(event.pos):
                            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g))
                        elif btn_rects["help"].collidepoint(event.pos):
                            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_h))
                        else:
                            # Select run row
                            hit_any = False
                            for rr in runs_row_hits:
                                if rr.rect.collidepoint(event.pos):
                                    selected_target = ("run", rr.row, -1)
                                    message = f"RUN sélectionné (couleur {rr.row//2 + 1}, slot {rr.row%2 + 1})"
                                    hit_any = True
                                    break
                            if hit_any:
                                pass
                            else:
                                # Select group column
                                for gc in groups_col_hits:
                                    if gc.rect.collidepoint(event.pos):
                                        selected_target = ("group", gc.block, gc.value_idx)
                                        message = f"GROUP sélectionné (valeur {gc.value_idx+1}, bloc {gc.block+1})"
                                        hit_any = True
                                        break
                                if not hit_any:
                                    # Place from hand if clicked a non-empty cell
                                    for hc in hand_cells:
                                        if hc.rect.collidepoint(event.pos) and hc.count > 0 and hc.tile_id is not None:
                                            try_place_from_hand(hc.tile_id)
                                            break

                    elif event.button == 3:
                        # Right-click on hand joker -> cycle hand_joker_value
                        for hc in hand_cells:
                            if hc.rect.collidepoint(event.pos) and hc.tile_id == JOKER_ID and hc.count > 0:
                                hand_joker_value = (hand_joker_value % 13) + 1
                                message = f"Joker main => valeur {hand_joker_value}"
                                append_debug(f"hand joker value -> {hand_joker_value}")
                                break
                        else:
                            # Right-click on table joker tile -> cycle assigned_value
                            for th in tile_hits:
                                if th.rect.collidepoint(event.pos):
                                    if edited_table.melds[th.meld_idx].slots[th.slot_idx].tile_id == JOKER_ID:
                                        cycle_table_joker_value(th.meld_idx, th.slot_idx)
                                        append_debug(f"table joker cycled at meld={th.meld_idx}, slot={th.slot_idx}")
                                    break

            pygame.display.flip()
            clock.tick(60)

    try:
        run_loop()
    except Exception:
        # Keep window open and display the traceback to avoid "silent close"
        tb = traceback.format_exc()
        lines = tb.splitlines()[-40:]
        W, H = screen.get_size()
        while True:
            screen.fill((15, 18, 23))
            title = title_font.render("GUI crashed — traceback:", True, ERR)
            screen.blit(title, (20, 20))
            y = 60
            for ln in lines:
                s = small_font.render(ln[:180], True, (230, 230, 230))
                screen.blit(s, (20, y))
                y += small_font.get_height() + 2
                if y > H - 40:
                    break
            hint = small_font.render("Press ESC or close the window.", True, WARN)
            screen.blit(hint, (20, H - 28))
            pygame.display.flip()
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    return
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return


if __name__ == "__main__":  # allows standalone execution
    launch_gui()
