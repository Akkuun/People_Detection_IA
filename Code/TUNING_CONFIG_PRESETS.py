"""
🎯 CONFIGURATION AUTO-TUNING - Guide rapide
Modifiez la section CONFIG dans trainVAEGAN.py selon votre besoin
"""

# ============================================
# PRESET 1: MUGSHOTS (Recommandé)
# ============================================
# Pour générer des mugshots de haute fidélité
# Très stable, reconstruction prioritaire

CONFIG_MUGSHOTS = {
    'auto_tune': True,
    'auto_tune_strategy': 'conservative',  # Très stable
    'auto_tune_start_epoch': 2,
    
    'recon_weight': 15.0,       # ← Très élevé (priorité absolue)
    'perceptual_weight': 0.5,   # ← Réduit (moins important)
    'adv_weight': 0.1,          # ← Très faible (stabilité max)
    'kl_weight': 0.005,         # ← Très réduit
    
    'num_epochs': 20,  # Augmenter si vous avez du temps
    'batch_size': 32,
}

# ============================================
# PRESET 2: EQUILIBRE (Recommandé par défaut)
# ============================================
# Bon compromis entre qualité et variété
# Stratégie intelligente, adaptative

CONFIG_BALANCED = {
    'auto_tune': True,
    'auto_tune_strategy': 'smart',  # Ajustements intelligents
    'auto_tune_start_epoch': 2,
    
    'recon_weight': 10.0,       # Équilibré
    'perceptual_weight': 1.0,   # Équilibré
    'adv_weight': 0.25,         # Équilibré
    'kl_weight': 0.01,          # Équilibré
    
    'num_epochs': 20,
    'batch_size': 32,
}

# ============================================
# PRESET 3: STYLES CREATIFS
# ============================================
# Pour styles variés et créatifs
# Plus d'adversarial, plus de variété

CONFIG_CREATIVE = {
    'auto_tune': True,
    'auto_tune_strategy': 'aggressive',  # Changements importants
    'auto_tune_start_epoch': 2,
    
    'recon_weight': 5.0,        # Réduit
    'perceptual_weight': 2.0,   # Augmenté
    'adv_weight': 0.5,          # Très élevé
    'kl_weight': 0.05,          # Augmenté (variété)
    
    'num_epochs': 20,
    'batch_size': 32,
}  # ← Ajouté

# ============================================
# PRESET 4: SANS AUTO-TUNING (Original)
# ============================================
# Pour tester ou comparer avec l'original

CONFIG_NO_TUNING = {
    'auto_tune': False,  # ← Désactiver
    
    'recon_weight': 10.0,
    'perceptual_weight': 1.0,
    'adv_weight': 0.25,
    'kl_weight': 0.01,
    
    'num_epochs': 10,
    'batch_size': 32,
}

# ============================================
# COMMENT UTILISER
# ============================================
"""
1. Ouvrez trainVAEGAN.py
2. Trouvez la section CONFIG (autour de ligne 150-180)
3. Remplacez les valeurs selon votre cas:

   Pour MUGSHOTS:
   CONFIG = {
       'auto_tune': True,
       'auto_tune_strategy': 'conservative',
       'auto_tune_start_epoch': 2,
       'recon_weight': 15.0,
       'perceptual_weight': 0.5,
       'adv_weight': 0.1,
       'kl_weight': 0.005,
       ... (garder les autres comme avant)
   }

4. Sauvegardez
5. Lancez: python trainVAEGAN.py

Les poids s'ajusteront automatiquement à chaque epoch!
"""

# ============================================
# STRATEGIES EXPLIQUEES
# ============================================
"""
SMART (défaut, recommandé généraliste):
  - Analyse la tendance et la volatilité
  - Ajustements proportionnels
  - Bon pour la plupart des cas
  - Résultat: équilibré et stable

CONSERVATIVE (pour mugshots):
  - Changements très minimes (1-2%)
  - Très stable, prévisible
  - Lent mais sûr
  - Résultat: haute fidélité

AGGRESSIVE (pour styles créatifs):
  - Changements importants (5-10%)
  - Exploration rapide
  - Peut être instable
  - Résultat: varié mais moins stable
"""

# ============================================
# CLÉS D'OPTIMISATION
# ============================================
"""
1. recon_weight (reconstruction):
   - Élevé: images fidèles (pour mugshots)
   - Bas: images plus abstraites (pour styles)

2. perceptual_weight (qualité perceptuelle):
   - Élevé: images de meilleure qualité visuelle
   - Bas: reconstruction pixel-wise prioritaire

3. adv_weight (adversarial/réalisme):
   - Élevé: images plus réalistes, GAN plus compétitif
   - Bas: entraînement plus stable, moins de variance

4. kl_weight (latent space variété):
   - Élevé: plus de variété dans les images générées
   - Bas: moins de variance, plus de cohérence
"""

# ============================================
# EXEMPLE: MON PREMIER ENTRAINEMENT
# ============================================
"""
Étape 1: Commencez avec MUGSHOTS (conservative)
  - Configuration la plus stable
  - Bons résultats pour faces
  - Peu d'ajustement nécessaire

Étape 2: Lancez l'entraînement
  python trainVAEGAN.py

Étape 3: Observez l'auto-tuning
  À chaque epoch après le 2:
  "🔧 Weights adjusted (epoch X): ..."

Étape 4: Attendez la fin (10-20 epochs)
  Checkpoints sauvegardés à chaque epoch

Étape 5: Évaluez
  Les best checkpoints sont dans checkpoints_vaegan/

Étape 6: (Optionnel) Itérez
  Changez la stratégie ou les paramètres initiaux
  Relancez
  Comparez les résultats
"""

if __name__ == '__main__':
    print(__doc__)
    print("\n✅ Consultez trainVAEGAN.py pour voir la CONFIG complète!")
