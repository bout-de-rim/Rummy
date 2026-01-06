# Roadmap – Rummikub RL Playground

Cette roadmap est structurée par sprints incrémentaux. Chaque sprint doit produire un système **jouable, testable et stable**.

> Exigence transversale : chaque sprint ajoute/renforce des **tests** (unitaires + invariants).

---

## Sprint 0 — Spécification & conventions (FAIT - documentation)

### Objectifs
- Verrouiller les règles et variantes (Ruleset)
- Définir les structures de données (tuile, multiset, meld, table, état)
- Définir le type de coup (ReplaceTable)
- Définir canonisation, hashing, sérialisation
- Définir les invariants et le plan de tests

### Livrables
- `SPRINT0_SPEC.md` (spécification détaillée)
- `DESIGN_CORE.md` (contrat core)

### Tests (spécifiés)
- Fixtures de melds valides/invalides
- Canonisation : équivalences -> même forme canonique
- Sérialisation round-trip
- Invariants de conservation (property-based)

---

## Sprint 1 — Core moteur (CLI / IPython)

### Objectifs
- Jouer une partie complète (2–4 joueurs) sans UI graphique
- Replays déterministes

### Livrables
- Moteur : `apply(move) -> new_state`
  - `DRAW`, `PASS`, `PLAY(delta_from_hand, new_table)`
- Validation : `is_legal_move(state, move) -> (bool, reason)`
- Gestion des tours, pioche, fin de partie
- Historique événementiel (event log) + replay
- Runner IPython / CLI

### Tests
- Conservation globale : somme(hands)+table+deck_restant == deck_total
- Aucun compte négatif
- `DRAW` : retire exactement 1 tuile du deck et l’ajoute à la main
- `PLAY` : conservation table + delta ; melds valides ; règles première pose
- Replays : seed identique => trajectoire identique

### Critères d’acceptation
- Parties se terminent sans crash
- Invariants jamais violés
- Un replay reproduit exactement une partie

---

## Sprint 2 — GUI (pygame)

### Objectifs
- Visualiser et manipuler explicitement la table en melds
- Inspection et debug via historique

### Livrables
- Rendu plateau en melds (lisible)
- Affichage mains (mode joueur / debug)
- Timeline (step/back/jump)
- Mode “dessiner un coup” : drag&drop, création/suppression/réarrangement de melds
- Validation interactive + message d’erreur

### Tests
- Tests UI minimaux (smoke tests) + tests d’intégration du pipeline `UI -> Move -> Engine`
- Snapshot tests (si pertinents) sur sérialisation d’états

---

## Sprint 3 — Canonicalisation & performance de validation

### Objectifs
- Stabiliser définitivement la représentation plateau (melds) et accélérer les checks
- Préparer caches (hash, multiset dérivé)

### Livrables
- Canonisation stricte melds/table (implémentée, pas seulement spécifiée)
- Hash stable d’état/table
- Cache du multiset dérivé de la table (optimisation, sans changer la “source of truth”)

### Tests
- Canonisation : idempotence, invariance au tri d’entrée
- Hash : équivalences -> même hash (ou même clé canonique)
- Bench micro : validation sous seuil (contrat de perf à définir)

---

## Sprint 4 — Génération de coups légaux (candidats)

### Objectifs
- Produire un ensemble de coups légaux candidats en temps borné

### Livrables
- `generate_candidate_moves(state, K, budget) -> [Move]`
  - d’abord sans réarrangement
  - puis réarrangement limité (budget 1 meld cassé, etc.)
  - puis beam search heuristique
- Mesures : temps moyen, #candidats, taux de légalité

### Tests
- Tous les coups générés sont légaux (`is_legal_move == True`)
- Tests de non-régression sur corpus d’états
- Bench : génération <= seuil (à définir)

---

## Sprint 5 — Bots baselines & harness d’évaluation

### Livrables
- Bot random légal (uniforme sur candidats)
- Bot heuristique (réduit la main, conserve jokers, etc.)
- Harness : tournois + métriques (winrate, durée, tuiles restantes, temps/turn)
- Sauvegarde des replays

### Tests
- Stabilité : milliers de parties sans crash
- Invariants respectés en simulation massive
- Replays valides

---

## Sprint 6 — Interface RL

### Livrables
- API type Gym : `reset/step`
- Observation : multi-sets + encodage plateau + scalaires
- Actions : index sur coups candidats + masque
- Logging : (obs, legal_moves, action, reward, next_obs, done)

### Tests
- Déterminisme `reset(seed)`
- Contrat shapes/dtypes observation
- Cohérence logging/replay

---

## Sprint 7 — HRL & self-play (optionnel)

### Livrables
- Macro-actions (DRAW / PLAY_SIMPLE / PLAY_REARRANGE_SMALL / PLAY_REARRANGE_LARGE)
- Micro-choix parmi candidats
- Self-play contrôlé
- Premières expériences RL

### Tests
- Non-régression sur bots baselines
- Contrôle de distribution macro-actions (sanity checks)
