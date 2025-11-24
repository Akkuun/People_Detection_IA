#!/usr/bin/env python3
"""
🤖 Script AUTO-TUNING - Entraîner le VAE-GAN avec ajustement automatique des poids
Utilisation: python trainVAEGAN_autotuned.py
"""

import sys
import os

# Pour lancer l'entraînement avec auto-tuning, c'est simple:
# 1. Éditer trainVAEGAN.py et vérifier la CONFIG
# 2. Lancer: python trainVAEGAN.py
#
# Les poids s'ajusteront automatiquement à chaque epoch!

# ============================================
# CONFIGURATION AUTO-TUNING
# ============================================
# Dans trainVAEGAN.py, vous verrez:
#
#   CONFIG = {
#       ...
#       'auto_tune': True,           # ← Activer auto-tuning
#       'auto_tune_start_epoch': 2,  # ← Commencer à epoch 2
#       'auto_tune_strategy': 'smart',  # ← Stratégie: 'smart', 'conservative', 'aggressive'
#       ...
#   }
#
# C'est tout! Juste lancer python trainVAEGAN.py

# ============================================
# STRATEGIES DISPONIBLES
# ============================================
#
# 1. 'smart' (RECOMMANDÉ)
#    - Analyse la tendance de la loss
#    - Ajustements intelligents
#    - Bon compromis entre vitesse et stabilité
#
# 2. 'conservative' (POUR MUGSHOTS)
#    - Très stable, changements minimes
#    - Idéal pour haute fidélité
#    - Lent mais sûr
#
# 3. 'aggressive' (POUR STYLES CREATIFS)
#    - Changements importants
#    - Exploration plus rapide
#    - Peut être instable

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🤖 AUTO-TUNING SYSTEM POUR VAE-GAN")
    print("="*70)
    
    print("""
✅ Votre script trainVAEGAN.py a été modifié avec auto-tuning!

COMMENT ÇA FONCTIONNE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. LANCEZ L'ENTRAÎNEMENT (comme avant):
   $ python trainVAEGAN.py

2. AUTO-TUNING AGIT AUTOMATIQUEMENT:
   - À chaque epoch, la loss est enregistrée
   - Après epoch 2, les poids s'ajustent
   - Les ajustements sont affichés à chaque epoch

3. RÉSULTAT:
   - Meilleur modèle automatiquement obtenu
   - Poids optimisés sans intervention manuelle
   - Checkpoints sauvegardés comme avant


CONFIGURATION (dans trainVAEGAN.py):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pour MUGSHOTS (fidélité maximale):
   'auto_tune': True,
   'auto_tune_strategy': 'conservative',  # ← Très stable
   'auto_tune_start_epoch': 2,
   'recon_weight': 15.0,        # ← Très élevé
   'perceptual_weight': 0.5,    # ← Réduit
   'adv_weight': 0.1,           # ← Très faible
   'kl_weight': 0.005,          # ← Très réduit

Pour STYLES CREATIFS:
   'auto_tune': True,
   'auto_tune_strategy': 'aggressive',    # ← Plus dynamique
   'auto_tune_start_epoch': 2,
   'recon_weight': 5.0,         # ← Moins important
   'perceptual_weight': 2.0,    # ← Plus important
   'adv_weight': 0.5,           # ← Élevé
   'kl_weight': 0.05,           # ← Élevé


PENDANT L'ENTRAÎNEMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

À chaque epoch, vous verrez:

🔧 Weights adjusted (epoch 3):
   recon=10.200, perceptual=1.020, adv=0.250, kl=0.010

Cela signifie que les poids ont été ajustés basés sur:
- La tendance de la loss
- La volatilité 
- La stratégie choisie


COMMENT MODIFIER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Ouvrir trainVAEGAN.py
2. Rechercher la section CONFIG (autour de ligne 200)
3. Modifier les paramètres:
   - 'auto_tune': True/False (activer/désactiver)
   - 'auto_tune_strategy': 'smart'/'conservative'/'aggressive'
   - 'auto_tune_start_epoch': numéro d'epoch (2, 3, 5, etc)
4. Sauvegarder
5. Lancer python trainVAEGAN.py


COMMENT ÇA FONCTIONNE TECHNIQUEMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Classe WeightAutoTuner (intégrée dans trainVAEGAN.py):

1. record_loss(loss)
   → Enregistre la loss moyenne de chaque epoch
   → Garde un historique (10 dernières epochs)

2. suggest_weights(epoch)
   → Analyse l'historique
   → Calcule les tendances
   → Retourne les poids ajustés

3. Stratégies:
   - 'smart': Ajustements basés sur tendance + volatilité
   - 'conservative': Changements minimes, très stable
   - 'aggressive': Changements importants, exploration rapide


CLÉS D'OPTIMISATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Si loss AUGMENTE:
   → Reduit adv_weight (l'adversarial est trop fort)
   → Augmente recon_weight (focus sur reconstruction)

2. Si loss DIMINUE BIEN:
   → Augmente adv_weight (peut augmenter la difficulté)
   → Augmente perceptual_weight (améliore la qualité)

3. Si loss EST VOLATILE:
   → Reduit adv_weight (trop d'instabilité)
   → Garde une base stable


COMPARAISON AVANT/APRÈS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AVANT (manuel):
❌ Ajuster les poids manuellement
❌ Retrainer à chaque changement
❌ Difficult de trouver les bons poids
❌ Très chronophage

APRÈS (auto-tuning):
✅ Les poids s'ajustent automatiquement
✅ Pendant l'entraînement
✅ Sans intervention manuelle
✅ Optimisation continue


EXEMPLE DE SESSION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

$ python trainVAEGAN.py

🚀 Using device: cuda
🤖 Auto-tuning: True (strategy: smart)
✅ Dataset loaded. Total samples: 8523
✅ Models initialized (VAE-GAN)
🎓 Starting training for 10 epochs...

Epoch 1/10: 100%|████| 267/267 [05:32<00:00]
✅ Epoch 1 finished in 332.1s -- checkpoint saved

Epoch 2/10: 100%|████| 267/267 [05:28<00:00]
✅ Epoch 2 finished in 328.5s -- checkpoint saved

🔧 Weights adjusted (epoch 3):
   recon=10.200, perceptual=1.020, adv=0.243, kl=0.010
Epoch 3/10: 100%|████| 267/267 [05:30<00:00]
✅ Epoch 3 finished in 330.2s -- checkpoint saved

🔧 Weights adjusted (epoch 4):
   recon=10.400, perceptual=1.040, adv=0.242, kl=0.010
Epoch 4/10: 100%|████| 267/267 [05:28<00:00]
✅ Epoch 4 finished in 328.1s -- checkpoint saved

... et ainsi de suite jusqu'à l'epoch 10


RÉSULTATS ATTENDUS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Meilleure qualité de génération (sans ajustement manuel)
✅ Loss qui diminue progressivement
✅ Modèles bien équilibrés
✅ Moins de variance entre epochs


C'EST TOUT! 🎉

Lancez simplement: python trainVAEGAN.py

Et laissez l'auto-tuning s'occuper du reste! 🤖
    """)
    
    print("\n" + "="*70)
    print("Pour démarrer: python trainVAEGAN.py")
    print("="*70 + "\n")
