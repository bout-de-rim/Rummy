# Rummikub RL Playground

Ce projet implémente un moteur de jeu **Rummikub** rigoureux, déterministe et orienté
**apprentissage par renforcement**, avec une représentation explicite du plateau en *melds*
et des mains en *multi-sets*.

L’objectif est double :
1. Fournir un **core de jeu fiable**, jouable par des humains (CLI / GUI),
2. Servir de **socle expérimental** pour des agents RL (self-play, HRL, MCTS, etc.).

## Principes clés
- Plateau = liste de melds (RUN / GROUP)
- Mains et pioche = multi-sets
- Coup = table résultante + delta depuis la main
- Historique événementiel (event sourcing)
- Jokers explicitement assignés
