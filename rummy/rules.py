from dataclasses import dataclass


@dataclass(frozen=True)
class Ruleset:
    num_players: int = 4
    colors: int = 4
    values: int = 13
    copies_per_tiletype: int = 2
    num_jokers: int = 2
    initial_hand_size: int = 14
    draw_ends_turn: bool = True
    allow_pass_when_play_available: bool = False
    initial_meld_min_points: int = 30
    initial_meld_must_use_only_hand: bool = True
    initial_meld_allow_joker: bool = True
    joker_can_substitute_any_tile: bool = True
    joker_reclaim_to_hand_allowed: bool = False

    def deck_size(self) -> int:
        normal_tiles = self.colors * self.values * self.copies_per_tiletype
        return normal_tiles + self.num_jokers
