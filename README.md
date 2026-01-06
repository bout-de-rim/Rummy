# Rummikub RL Playground

Ce projet implémente un moteur de jeu **Rummikub** rigoureux, déterministe et orienté **apprentissage par renforcement**,
avec une représentation explicite du plateau en *melds* et des mains en *multi-sets*.

Objectifs :
1. Fournir un **core de jeu fiable**, jouable par des humains (CLI / GUI),
2. Servir de **socle expérimental** pour des agents RL (self-play, HRL, MCTS guidé, imitation learning).

---

## Principes de conception (décisions actées)

- **Plateau = liste canonique de melds** (`RUN` / `GROUP`) : *source of truth*.
- **Main = multi-set** : comptage par type de tuile (52 types + joker).
- **Coup = ReplaceTable** :
  - `delta_from_hand` (multi-set) + `new_table_melds` (table complète après réarrangement).
- **Historique événementiel** (event sourcing) : replays déterministes, undo/redo par replay.
- **Jokers explicitement assignés dans les melds** (couleur/valeur effectives) pour rendre la validité déterministe.
- **Canonisation + hashing** des melds et de la table pour déduplication, caches, debug, et reproductibilité.
- Architecture orientée :
  - validation forte des règles,
  - génération de coups candidats (top-K) avec budget,
  - intégration RL via « candidats + masque d’actions ».

Voir [`DESIGN_CORE.md`](DESIGN_CORE.md) pour le contrat technique, et [`SPRINT0_SPEC.md`](SPRINT0_SPEC.md) pour la spécification du Sprint 0.

---

## Statut

- Sprint 0 : spécification et conventions (documentées)
- Sprint 1+ : implémentation moteur / UI / génération / RL (voir roadmap)

---

## Structure de dépôt (prévisionnelle)

```
.
├── README.md
├── rummy/
│   ├── engine.py
│   ├── cli.py
│   ├── meld.py
│   ├── move.py
│   ├── multiset.py
│   ├── rules.py
│   ├── state.py
│   ├── table.py
│   └── tiles.py
├── roadmap.md
├── DESIGN_CORE.md
├── SPRINT0_SPEC.md
├── engine/
├── ui/
├── bots/
├── rl/
└── tests/
```

---

## Sprint 1 — état actuel

Le moteur core (Sprint 1) est opérationnel en Python :
- `new_game(ruleset, rng_seed)` crée une partie déterministe (pioche ordonnée par seed).
- `is_legal_move(state, move)` valide `DRAW`, `PASS` et `PLAY`.
- `apply_move(state, move)` applique le coup, gère la pioche, les tours, l’historique et détecte la victoire (main vide).
- `replay_event_log(initial_state, events)` rejoue un historique pour vérifier le déterminisme.
- Un runner CLI minimal (`python -m rummy.cli --seed 123`) joue une partie automatique et affiche le vainqueur/état final.

Tests (pytest) couvrent la pioche, la validation des melds, la conservation et la reproductibilité des replays.

---

## Philosophie

Priorité absolue : **justesse des règles + invariants**.
Performance et RL viennent ensuite, en s’appuyant sur une base saine et testée.
