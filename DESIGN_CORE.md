# Core Design — Rummikub Engine

## Tuiles
- 52 types couleur/valeur + joker
- id stable (0–52)

## Multi-set
Vecteur de comptage par type de tuile.
Opérations : add, sub, inclusion, hash.

## Melds
- RUN : couleur fixe, valeurs consécutives, len >= 3
- GROUP : valeur fixe, couleurs distinctes, len 3–4
- Joker toujours explicitement assigné

## Table
Liste canonique de melds (source of truth).

## Coup (Move)
Types :
- DRAW
- PASS
- PLAY

PLAY :
- delta_from_hand (multiset)
- new_table (liste de melds)

Légalité :
- delta ⊆ main
- melds valides
- conservation stricte des tuiles
- règles de première pose respectées

## Invariants
- conservation globale
- aucune main négative
- melds toujours valides
- replays déterministes
