# Roadmap – Rummikub RL Playground

## Sprint 0 — Spécification & conventions
- Ruleset paramétrable
- Modèle de données (tuile, multiset, meld, table)
- Type de coup (ReplaceTable)
- Canonisation + hashing
- Sérialisation JSON
- Invariants et plan de tests

## Sprint 1 — Core moteur (CLI / IPython)
- draw / pass / play
- validation forte
- historique événementiel
- replays déterministes

## Sprint 2 — GUI (pygame)
- affichage des melds
- navigation historique
- construction graphique des coups

## Sprint 3 — Validation & canonicalisation
- validateur melds
- hash stable des états

## Sprint 4 — Génération de coups légaux
- générateur de candidats
- heuristiques & beam search

## Sprint 5 — Bots & évaluation
- bot random
- bot heuristique
- harness de tournois

## Sprint 6 — Interface RL
- environnement type Gym
- logging de trajectoires

## Sprint 7 — HRL & self-play
