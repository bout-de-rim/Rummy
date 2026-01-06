# Core Design — Rummikub Engine

Ce document spécifie le **contrat fondamental** du moteur de jeu. Il fait foi pour toute implémentation, UI ou agent RL.

---

## 1) Représentation des tuiles

### TileTypeID
- 4 couleurs × 13 valeurs = 52 types
- + 1 type JOKER
- Identifiant stable :
  - `id = color_index * 13 + (value - 1)` avec `value ∈ [1..13]`
  - `joker_id = 52`

---

## 2) Multi-set (mains, comptages, etc.)

### Définition
Un multi-set est un vecteur de longueur 53 : `count[tile_id]`.

### Propriétés
- comptes entiers ≥ 0
- opérations fermées :
  - `add(a, b)`
  - `sub(a, b)` (précondition : `b ⊆ a`)
  - `leq(a, b)` (inclusion)
  - `total(a)`

---

## 3) Melds (source of truth du plateau)

### Types
- `RUN` : même couleur, valeurs consécutives, longueur ≥ 3
- `GROUP` : même valeur, couleurs distinctes, longueur 3 ou 4

### Joker
- Les jokers sont autorisés selon le `Ruleset`.
- Un joker posé sur table est **toujours assigné explicitement** :
  - `assigned_color` et `assigned_value`
- Un joker **non assigné** rend le meld invalide.

### Représentation recommandée
Un meld est une liste de `slots`, où chaque slot est :
- soit une tuile normale (tile_id 0..51)
- soit un joker (tile_id 52) avec assignation `(assigned_color, assigned_value)`

Le multiset “physique” d’un meld compte les jokers comme jokers (tile_id=52), indépendamment de leur assignation.

---

## 4) Table

- La table est une **liste de melds**
- Canonisée et triée selon une clé stable (cf. Canonisation)
- Un `table_multiset` dérivé peut être calculé/caché pour validation rapide (sans devenir la source of truth)

---

## 5) Coup (Move) — ReplaceTable

### Types
- `DRAW`
- `PASS`
- `PLAY`

### PLAY
- `delta_from_hand` : multi-set (tuiles prises de la main du joueur actif)
- `new_table_melds` : table complète après le coup (incluant tous réarrangements)

### Légalité (PLAY)
Un PLAY est légal si et seulement si :
1. `delta_from_hand ⊆ hand[player]`
2. tous les melds de `new_table_melds` sont valides (incl. assignations jokers)
3. **Conservation** :
   `multiset(new_table_melds) == multiset(old_table_melds) + delta_from_hand`
4. contraintes de première pose (si non faite)

---

## 6) Validité des melds (définition)

### GROUP
- longueur 3 ou 4
- toutes les **valeurs effectives** identiques
- toutes les **couleurs effectives** distinctes
- joker : `assigned_value` = valeur du groupe ; `assigned_color` set de couleurs distinctes des autres

### RUN
- longueur ≥ 3
- toutes les **couleurs effectives** identiques
- valeurs effectives strictement consécutives
- pas de doublon de valeur effective
- pas de “wrap” 13→1

---

## 7) Canonisation & hashing

### Canonisation d’un meld
- RUN : ordonner slots par valeur effective croissante
- GROUP : ordonner slots par couleur effective croissante

`effective_color/value` :
- tuile normale : dérivée du tile_id
- joker : `(assigned_color, assigned_value)`

### Canonisation de la table
Trier les melds par clé stable, par exemple :
- `(kind, effective_color, effective_min_value, length, effective_signature)`
où `effective_signature` capture la séquence des valeurs/couleurs effectives.

### Hash
- `table_key = hash(serialisation_canonique(table))`
- `state_key = hash(hands + table_key + deck_index + current_player + flags)`

---

## 8) Sérialisation

- JSON lisible, round-trip
- Multiset : format compact conseillé `[[tile_id, count], ...]` en omettant les zéros
- Meld : liste de slots (avec assignations jokers explicites)
- Event log : liste d’événements `DRAW/PASS/PLAY` avec états/hashes optionnels

---

## 9) Invariants globaux

- Conservation : `sum(hands) + multiset(table) + remaining_deck == deck_total`
- mains jamais négatives
- melds sur table toujours valides (après application d’un move)
- replays déterministes à seed donnée
