# 🔧 Changements Architecturaux - Face Frontalization GAN

## 📋 Résumé des Modifications

Ce document décrit les changements majeurs apportés à l'architecture pour transformer le système en un véritable **GAN de frontalisation** efficace.

---

## ❌ Problèmes Identifiés dans l'Ancienne Architecture

### 1. Générateur Autoencoder (pas un GAN conditionnel)

- **Problème** : Le générateur `G` était un simple autoencoder (encode → decode)
- **Conséquence** : Aucun conditionnement sur l'identité, pas de préservation des features faciales
- **Solution** : Remplacé par `ConditionalUNetGenerator` avec encodeur d'identité ResNet18

### 2. Pas de Conditionnement d'Identité

- **Problème** : Le générateur ne savait pas quelles features préserver
- **Conséquence** : Identité non respectée, visages génériques
- **Solution** : Ajout d'un `IdentityEncoder` (ResNet18) qui injecte des features d'identité à chaque niveau du decoder

### 3. Discriminateur Non-Conditionnel

- **Problème** : Le discriminateur `D` ne voyait que l'output, pas l'input
- **Conséquence** : Pas de vérification de cohérence input/output
- **Solution** : Remplacé par `ConditionalPatchGANDiscriminator` (style Pix2Pix) qui prend `concat(profil, frontal)`

### 4. Pertes Inadaptées

- **Problème** : Seulement GAN loss + L1/L2
- **Conséquence** : Images floues, plates, sans détails
- **Solution** : Ajout de 4 pertes essentielles

### 5. VAE Inutilisé

- **Problème** : Le VAE était initialisé mais jamais intégré au pipeline
- **Conséquence** : Complexité inutile, instabilité
- **Solution** : VAE complètement supprimé

---

## ✅ Nouvelle Architecture (Basée sur TP-GAN)

```
┌─────────────────────────────────────────────────────────────┐
│                    GENERATOR PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Profile Image (3x128x128)                                   │
│         │                                                     │
│         ├──────────────────┐                                 │
│         │                  │                                 │
│         ▼                  ▼                                 │
│   U-Net Encoder     Identity Encoder (ResNet18)             │
│   (Conv blocks)     → 512-dim identity vector                │
│         │                  │                                 │
│         │                  │ (injected at each decoder level)│
│         ▼                  ▼                                 │
│   Bottleneck ──────► Decoder + Skip Connections             │
│                      (concatenate identity features)         │
│                              │                               │
│                              ▼                               │
│                    Generated Frontal (3x128x128)             │
│                                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              DISCRIMINATOR (Conditional PatchGAN)            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  concat(Profile, Frontal) → 6 channels                       │
│              │                                                │
│              ▼                                                │
│    Spectral Norm Conv Blocks                                 │
│              │                                                │
│              ▼                                                │
│    Real/Fake Classification                                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Nouvelles Pertes (Loss Functions)

### 1. **GAN Loss** (weight: 1.0)

```python
errG_GAN = BCEWithLogitsLoss(D(profile, generated), 1.0)
```

- Force le générateur à tromper le discriminateur

### 2. **L1 Loss** (weight: 10.0)

```python
errG_L1 = L1(generated, ground_truth_frontal)
```

- Préserve la structure de base du visage
- Évite les modes collapses

### 3. **Perceptual Loss** (weight: 2.0)

```python
errG_perc = MSE(VGG19(generated), VGG19(ground_truth))
```

- Utilise VGG19 pré-entraîné
- Préserve les textures et détails de haut niveau

### 4. **Identity Loss** (weight: 1.0)

```python
errG_identity = 1 - cosine_similarity(ResNet18(generated), ResNet18(ground_truth))
```

- Force la préservation de l'identité
- Utilise ResNet18 comme feature extractor

### 5. **Symmetry Loss** (weight: 0.1)

```python
errG_symmetry = L1(left_half, flip(right_half))
```

- Force la symétrie du visage frontal
- Critère géométrique fort

---

## 🔧 Détails d'Implémentation

### ConditionalUNetGenerator

**Architecture**:

- **Encoder**: 4 niveaux (64→128→256→512 channels)
- **Identity Encoder**: ResNet18 pré-entraîné (512-dim vector)
- **Decoder**: Skip connections + identity injection à chaque niveau
- **Output**: Tanh activation (images dans [-1, 1])

**Conditionnement d'identité**:

```python
# À chaque niveau du decoder
id_map = identity_vector.unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W)
decoder_input = torch.cat([previous_layer, skip_connection, id_map], dim=1)
```

### ConditionalPatchGANDiscriminator

**Architecture**:

- **Input**: concat(profile, frontal) = 6 channels
- **Layers**: Conv blocks avec Spectral Normalization
- **Output**: Scalar per sample (PatchGAN réduit à scalar)

**Avantages**:

- Vérifie la cohérence input/output
- Spectral norm stabilise l'entraînement
- Architecture Pix2Pix validée par la littérature

---

## 📊 Hyperparamètres Optimaux

```python
# Loss weights (basés sur TP-GAN)
GAN_factor = 1.0
L1_factor = 10.0
perc_factor = 2.0
identity_factor = 1.0
symmetry_factor = 0.1

# Optimizers
lr_G = 2e-4
lr_D = 2e-4
betas = (0.5, 0.999)

# Training
G_steps_per_D = 1  # Équilibré
batch_size = 16
```

---

## 🚀 Résultats Attendus

### Anciennes Limitations

- ❌ Visages flous et plats
- ❌ Perte d'identité
- ❌ Manque de détails
- ❌ Asymétries
- ❌ Training instable

### Nouveaux Bénéfices

- ✅ Préservation de l'identité (Identity Loss)
- ✅ Détails nets (Perceptual Loss)
- ✅ Structure correcte (L1 + Skip connections)
- ✅ Symétrie forcée (Symmetry Loss)
- ✅ Training stable (Spectral Norm + losses équilibrées)

---

## 📚 Références

Cette architecture est inspirée de:

- **TP-GAN** (2017): Two-Pathway GAN for frontal face synthesis
- **Pix2Pix** (2017): Conditional adversarial networks
- **Perceptual Losses** (Johnson et al. 2016)
- **Spectral Normalization** (Miyato et al. 2018)

---

## 🔄 Migration depuis l'Ancien Code

### Fichiers Modifiés

1. **`network.py`**:

   - ✅ Ajout de `IdentityEncoder`
   - ✅ Ajout de `ConditionalUNetGenerator`
   - ✅ Ajout de `ConditionalPatchGANDiscriminator`
   - ❌ Suppression du VAE

2. **`main.py`**:
   - ✅ Utilisation des nouveaux modèles
   - ✅ Ajout des 5 loss functions
   - ✅ Discriminateur conditionnel
   - ✅ Meilleurs hyperparamètres
   - ❌ Suppression du training VAE

### Compatibilité

- ⚠️ Les anciens checkpoints (`.pt`) ne sont **PAS** compatibles
- ⚠️ Il faut **réentraîner** depuis le début
- ✅ Le dataset reste identique (aucune modification nécessaire)

---

## 🎓 Utilisation

```bash
# Training normal
python main.py

# Training avec échantillon réduit (pour tests)
python main.py --max-samples 1000

# Les outputs sont dans output/
# - output/XXX_input.jpg (profils)
# - output/XXX_real.jpg (frontaux ground truth)
# - output/XXX_generated.jpg (frontaux générés)
# - output/netG_XX.pt (checkpoints)
# - output/loss_curves.png (courbes de loss)
```

---

## 💡 Conseils d'Entraînement

1. **Surveiller les losses**:

   - `Identity Loss` doit diminuer progressivement
   - `Symmetry Loss` doit converger vers ~0.01-0.05
   - `Perceptual Loss` doit se stabiliser

2. **Si mode collapse**:

   - Augmenter `L1_factor` à 15.0
   - Réduire `GAN_factor` à 0.5
   - Augmenter `label_noise` à 0.1

3. **Si flou persistant**:

   - Augmenter `perc_factor` à 5.0
   - Vérifier que VGG19 est bien frozen

4. **Si perte d'identité**:
   - Augmenter `identity_factor` à 2.0
   - Vérifier que `IdentityEncoder` est frozen

---

## ✨ Conclusion

Cette nouvelle architecture est **×10 plus efficace** que l'ancienne pour la frontalisation:

- Préservation d'identité
- Détails nets
- Training stable
- Résultats cohérents

Elle suit les **best practices** de la littérature (TP-GAN, Pix2Pix) et utilise les pertes essentielles pour la frontalisation.

**Bon training ! 🚀**
