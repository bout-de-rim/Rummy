# Sprint 0 — Spécification & conventions (détaillé)

Ce document consolide la conception convenue : plateau en melds, mains en multi-sets, coup ReplaceTable, déterminisme, sérialisation, invariants et plan de tests.

---

## 1) Objectifs Sprint 0
- figer un `Ruleset` paramétrable (avec défauts “standard”)
- définir le modèle de données (TileType, Multiset, Meld, Table, GameState)
- définir le type de coup `Move` (ReplaceTable)
- définir canonisation + hashing
- définir sérialisation JSON
- établir les invariants et un plan de tests

---

## 2) Ruleset (paramètres)

### Paramètres de base (défauts proposés)
- `num_players`: 2–4 (défaut 4 ; option 5 pour simulation à décider)
- `colors`: 4
- `values`: 1..13
- `copies_per_tiletype`: 2
- `num_jokers`: 2
- `initial_hand_size`: 14

### Tour de jeu
- `draw_ends_turn`: True (défaut) — variante : piocher puis jouer
- `allow_pass_when_play_available`: False (défaut) ou True (variante)

### Première pose
- `initial_meld_min_points`: 30
- `initial_meld_must_use_only_hand`: True (défaut standard)
- `initial_meld_allow_joker`: à confirmer (dépend des variantes)

### Joker
- `joker_can_substitute_any_tile`: True
- `joker_reclaim_rule`: à confirmer (remplacer le joker par la tuile correspondante et réutiliser le joker dans la table finale du même tour)
- `joker_reclaim_to_hand_allowed`: False (défaut) — le joker reste sur table, éventuellement déplacé

### Fin de partie / score
- fin : première main vide
- scoring : optionnel (Sprint 1 peut stocker seulement le gagnant + tuiles restantes)

---

## 3) Modèle de données

### TileTypeID
- mapping stable : `id = color*13 + (value-1)`, `joker_id=52`

### Multiset
- vecteur length 53
- opérations : add/sub/leq/total

### MeldSlot
- normal : `{tile_id}`
- joker : `{tile_id: 52, assigned_color: 0..3, assigned_value: 1..13}`
- joker sans assignation => invalide

### Meld
- `{kind: RUN|GROUP, slots: [MeldSlot...]}`

### Table
- liste de Melds (canonisée)

### GameState (minimal)
- `ruleset`
- `current_player`
- `hands: list[Multiset]`
- `table: Table`
- `deck_order: list[TileTypeID]` + `deck_index`
- `initial_meld_done: list[bool]`
- `turn_number`
- `rng_seed` (audit)

---

## 4) Move (ReplaceTable)

### DRAW / PASS / PLAY
- PLAY :
  - `delta_from_hand: Multiset`
  - `new_table: Table`

### Légalité PLAY
1. `delta_from_hand ⊆ main[player]`
2. `new_table` : tous melds valides
3. conservation : `multiset(new_table) == multiset(old_table) + delta_from_hand`
4. première pose : règles du Ruleset

---

## 5) Canonisation

### RUN
- trier slots par valeur effective croissante

### GROUP
- trier slots par couleur effective croissante

### Table
- trier melds par clé stable (kind, couleur, min value, len, signature)

---

## 6) Sérialisation JSON

### Multiset
- format compact : `[[tile_id, count], ...]` (zéros omis)

### Meld
- slots explicites (joker assigné)

### EventLog
- événements : `player`, `move`, optionnel : `state_key_before/after`

---

## 7) Invariants (tests)

### Invariants globaux
- conservation stricte des tuiles
- mains jamais négatives
- deck_index cohérent
- alternance des tours

### Invariants melds
- melds valides
- jokers assignés

---

## 8) Plan de tests (Sprint 0)

1. Fixtures melds valides/invalides
   - GROUP avec doublon de couleur => invalide
   - RUN avec trou non couvert => invalide
   - RUN avec joker mauvaise couleur => invalide
   - joker sans assignation => invalide
2. Canonisation
   - permutations d’entrée => même forme canonique
3. Sérialisation round-trip
   - objet -> json -> objet => égalité canonique
4. Conservation (property-based)
   - générer états aléatoires valides + appliquer opérations multiset + vérifier invariants

---

## 9) Points à trancher (bloquants)
1. Après DRAW, le tour se termine-t-il ? (`draw_ends_turn`)
2. Jokers autorisés sur la première pose pour atteindre 30 points ?
3. Règle de reprise du joker (remplacement et réutilisation le même tour) ?
4. Nombre de joueurs : 2–4 strict ou autoriser 5 en simulation ?
5. Joker : exigence “toujours assigné” confirmée (recommandée) ?
