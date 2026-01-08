# Tests GUI (pygame) avec screenshots en headless

Ce guide décrit une stratégie fiable pour tester une GUI pygame en environnement **headless** (CI/CD ou machine sans affichage), en générant des **screenshots** .

---

## Objectifs

- Produire une image (PNG) d’un état UI déterministe, **sans ouvrir de fenêtre**.
- Faciliter des tests de non-régression visuelle (golden tests).
- Garder la logique de rendu séparée de la logique de jeu, pour des tests rapides et robustes.

---

## Pré-requis (principes de conception)

### 1) Rendu “pur” et testable
Isoler le rendu dans une fonction du type :

- Entrées : `surface`, `state` (et éventuellement `ui_config`)
- Sorties : dessin sur la surface (pas d’I/O, pas d’événements, pas de boucle)

Exemple de convention :
- `render(surface, state)` : dessine tout (fond, table, mains, HUD)
- `render_table(surface, table)`, `render_hand(surface, hand)`, etc.

### 2) Déterminisme
Pour que les screenshots soient comparables :
- Fixer une seed RNG pour l’état de jeu (et toute génération)
- Éviter toute dépendance au temps (animations, clignotements) en mode test
- Stabiliser la police (voir section “Fonts”)

---

## Mode headless : SDL “dummy”

Pygame repose sur SDL. En headless, on force SDL à utiliser un driver vidéo “dummy”.

### Variables d’environnement à définir
- `SDL_VIDEODRIVER=dummy`
- (optionnel) `SDL_AUDIODRIVER=dummy`

**Important :** ces variables doivent être définies **avant d’importer pygame** (ou avant toute initialisation display).

---

## Approche recommandée : rendu sur `pygame.Surface` (sans display)

Pour les tests, privilégier une surface hors display :

- Pas besoin de `pygame.display.set_mode`
- Fonctionne en CI sans serveur X/Wayland
- Plus stable pour les tests

### Pipeline de test
1. Construire un `GameState` de test (fixture)
2. Créer une `pygame.Surface((W,H))`
3. Appeler `render(surface, state)`
4. Sauvegarder : `pygame.image.save(surface, path)`
(5. Eventuellement, comparer à une image de référence)

---

