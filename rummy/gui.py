
from __future__ import annotations

"""
Rummikub — Compact GUI (grid-first)

Fixes in this version (v4):
- Colored tiles are back (tile fill color derived from tile color; jokers get purple or assigned color)
- Better duplicate stack effect (real "second tile" behind + ×N badge)
- Subtle axis labels (1..13 and J) so empty grids are still usable
- Local placement validation (only place if it keeps the target run/group locally consistent)
- Robust crash screen to avoid "silent close"

Hard constraint:
- This GUI only supports exactly 4 colors and 13 values. If not, it refuses to start.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
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


# --- Theme ---------------------------------------------------------------------

BG = (22, 27, 34)
PANEL = (30, 36, 46)
PANEL_LINE = (54, 63, 77)
TEXT = (220, 226, 235)
SUB = (164, 174, 187)
ACCENT = (88, 138, 255)
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


def map_runs_rows_to_meld_indices(table: Table) -> Dict[int, int]:
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


# --- Drawing primitives --------------------------------------------------------

def draw_panel(surface, rect: pygame.Rect):
    pygame.draw.rect(surface, PANEL, rect, border_radius=10)
    pygame.draw.rect(surface, PANEL_LINE, rect, width=2, border_radius=10)


def draw_button(surface, rect: pygame.Rect, label: str, font):
    pygame.draw.rect(surface, (46, 56, 69), rect, border_radius=10)
    pygame.draw.rect(surface, ACCENT, rect, width=2, border_radius=10)
    txt = font.render(label, True, TEXT)
    surface.blit(txt, txt.get_rect(center=rect.center))


def draw_tile(surface, rect: pygame.Rect, slot: TileSlot, small_font, highlight=False, count_badge: Optional[int] = None, stacked_back: bool = False):
    radius = 6
    bg = _tile_bg(slot)
    if stacked_back:
        bg = _mix(bg, 0.22, toward=(0, 0, 0))
    border = ACCENT if highlight else (60, 70, 85)

    pygame.draw.rect(surface, bg, rect, border_radius=radius)
    pygame.draw.rect(surface, border, rect, width=2, border_radius=radius)

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


# --- GUI entry -----------------------------------------------------------------

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

    # UI state
    selected_target: Optional[Tuple[str, int, int]] = None
    message = "Sélectionnez un RUN (gauche) ou un GROUP (droite), puis cliquez une tuile de la main pour l'envoyer si autorisé."
    show_debug = False
    debug_scroll = 10**9
    debug_log: List[str] = []
    hand_joker_value = 1

    # Hit regions
    hand_cells: List[HandCell] = []
    runs_row_hits: List[RunsRowHit] = []
    groups_col_hits: List[GroupsColHit] = []
    tile_hits: List[TileHit] = []

    def reset_draft():
        nonlocal edited_table, selected_target
        edited_table = current().table.canonicalize()
        selected_target = None

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

    def make_slot_for_target(tile_id: int, target: Tuple[str, int, int]) -> TileSlot:
        nonlocal hand_joker_value, edited_table
        kind, a, b = target
        if tile_id != JOKER_ID:
            return TileSlot(tile_id)

        if kind == "group":
            block, value_idx = a, b
            target_val = value_idx + 1
            mapping = map_groups_cells_to_meld_indices(edited_table)
            meld_idx = mapping.get((block, value_idx))
            used = set()
            if meld_idx is not None:
                for s in edited_table.melds[meld_idx].slots:
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
        slot = make_slot_for_target(tile_id, selected_target)
        kind, a, b = selected_target

        if kind == "run":
            row = a
            row_color = row // 2
            row_map = map_runs_rows_to_meld_indices(edited_table)
            meld_idx = row_map.get(row)
            if meld_idx is None:
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
        new_c = 0 if s.assigned_color is None else s.assigned_color
        m.slots[slot_idx] = TileSlot(JOKER_ID, assigned_color=new_c, assigned_value=new_v)
        message = f"Joker => valeur {new_v}"

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

            row_map = map_runs_rows_to_meld_indices(edited_table)
            group_map = map_groups_cells_to_meld_indices(edited_table)
            nblocks = group_block_count(edited_table)
            tile_hits = []

            # RUNS grid 8x13
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

            for row, meld_idx in row_map.items():
                meld = edited_table.melds[meld_idx]
                for slot_idx, s in enumerate(meld.slots):
                    v = _tile_value(s)
                    if not v:
                        continue
                    col = _clamp(v - 1, 0, 12)
                    cx = grid_x + col * cell_w + (cell_w - tile_w) / 2
                    cy = grid_y + row * cell_h + (cell_h - tile_h) / 2
                    rect = pygame.Rect(int(cx), int(cy), tile_w, tile_h)
                    draw_tile(screen, rect, s, small_font)
                    tile_hits.append(TileHit(rect=rect, meld_idx=meld_idx, slot_idx=slot_idx))

            # GROUPS grid blocks x (4x13)
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
                    draw_tile(screen, rect, s, small_font)
                    tile_hits.append(TileHit(rect=rect, meld_idx=meld_idx, slot_idx=slot_idx))

            # HAND grid 4x14
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
                        draw_tile(screen, main, slot, small_font, count_badge=count)

            # Status bar
            right = f"À jouer : Joueur {current().current_player + 1}"
            status_bar(screen, status, message, right, small_font)

            # Debug overlay (optional) — kept minimal here
            if show_debug:
                dbg = pygame.Rect(margin, header.bottom + margin, int(W * 0.38), int(H * 0.42))
                pygame.draw.rect(screen, (20, 26, 34), dbg, border_radius=10)
                pygame.draw.rect(screen, (90, 98, 118), dbg, width=2, border_radius=10)
                pad = 8
                clip = pygame.Rect(dbg.x + pad, dbg.y + pad, dbg.width - 2 * pad, dbg.height - 2 * pad)
                lines = [
                    f"timeline={time_idx+1}/{len(timeline)}",
                    f"turn={getattr(current(), 'turn', '?')} deck={len(getattr(current(), 'deck', []))}",
                    f"hand_remaining={sum(remaining.counts)}",
                    f"edited_melds={len(edited_table.melds)} group_blocks={nblocks}",
                    f"selected_target={selected_target}",
                    "",
                    *debug_log[-300:],
                ]
                lh = small_font.get_height() + 2
                content_h = lh * len(lines)
                max_scroll = max(0, content_h - clip.height)
                debug_scroll = _clamp(debug_scroll, 0, max_scroll)
                prev = screen.get_clip()
                screen.set_clip(clip)
                y = clip.y - debug_scroll
                for ln in lines:
                    screen.blit(small_font.render(ln, True, (210, 216, 225)), (clip.x, y))
                    y += lh
                screen.set_clip(prev)

            # Events
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
                        move, err = build_play_move(current(), edited_table)
                        if move:
                            nxt = apply_move(current(), move)
                            if time_idx != len(timeline) - 1:
                                timeline[:] = timeline[: time_idx + 1]
                            timeline.append(nxt)
                            time_idx = len(timeline) - 1
                            edited_table = current().table.canonicalize()
                            message = "PLAY effectué."
                        else:
                            message = f"PLAY invalide: {err}"
                            debug_log.append(f"PLAY invalid: {err}")
                    elif event.key == pygame.K_d:
                        move = Move.draw()
                        nxt = apply_move(current(), move)
                        if time_idx != len(timeline) - 1:
                            timeline[:] = timeline[: time_idx + 1]
                        timeline.append(nxt)
                        time_idx = len(timeline) - 1
                        edited_table = current().table.canonicalize()
                        message = "Pioche effectuée."
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
                            hit_any = False
                            for rr in runs_row_hits:
                                if rr.rect.collidepoint(event.pos):
                                    selected_target = ("run", rr.row, -1)
                                    message = f"RUN sélectionné (couleur {rr.row//2 + 1}, slot {rr.row%2 + 1})"
                                    hit_any = True
                                    break
                            if not hit_any:
                                for gc in groups_col_hits:
                                    if gc.rect.collidepoint(event.pos):
                                        selected_target = ("group", gc.block, gc.value_idx)
                                        message = f"GROUP sélectionné (valeur {gc.value_idx+1}, bloc {gc.block+1})"
                                        hit_any = True
                                        break
                            if not hit_any:
                                for hc in hand_cells:
                                    if hc.rect.collidepoint(event.pos) and hc.count > 0 and hc.tile_id is not None:
                                        try_place_from_hand(hc.tile_id)
                                        break
                    elif event.button == 3:
                        # hand joker cycles its chosen value (used for runs)
                        for hc in hand_cells:
                            if hc.rect.collidepoint(event.pos) and hc.tile_id == JOKER_ID and hc.count > 0:
                                hand_joker_value = (hand_joker_value % 13) + 1
                                message = f"Joker main => valeur {hand_joker_value}"
                                break
                        else:
                            # table joker cycles its assigned value
                            for th in tile_hits:
                                if th.rect.collidepoint(event.pos):
                                    if edited_table.melds[th.meld_idx].slots[th.slot_idx].tile_id == JOKER_ID:
                                        cycle_table_joker_value(th.meld_idx, th.slot_idx)
                                    break

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
    except Exception:
        crash_screen(traceback.format_exc())


if __name__ == "__main__":
    launch_gui()
