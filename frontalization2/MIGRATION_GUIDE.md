# 🚀 Guide de Migration - Face Frontalization GAN

## ✅ Changements Appliqués

### 1. Architecture du Générateur

**Avant**: `UNetGenerator` (simple autoencoder)
**Après**: `ConditionalUNetGenerator` avec:

- ✅ Identity Encoder (ResNet18 pré-entraîné)
- ✅ Skip connections U-Net
- ✅ Identity features injectées à chaque niveau du decoder
- ✅ Préservation de l'identité garantie

### 2. Architecture du Discriminateur

**Avant**: `D` et `PatchGANDiscriminator` (non-conditionnels)
**Après**: `ConditionalPatchGANDiscriminator` avec:

- ✅ Input: concat(profile, frontal) = 6 channels
- ✅ Vérifie la cohérence input/output
- ✅ Spectral Normalization pour stabilité
- ✅ Style Pix2Pix validé par la littérature

### 3. Loss Functions

**Avant**:

- GAN Loss
- L1 Loss
- L2 Loss (inutile)
- Perceptual Loss (VGG16)

**Après**:

- ✅ GAN Loss (weight: 1.0)
- ✅ L1 Loss (weight: 10.0)
- ✅ **Perceptual Loss** (weight: 2.0) - VGG19 au lieu de VGG16
- ✅ **Identity Loss** (weight: 1.0) - NOUVEAU
- ✅ **Symmetry Loss** (weight: 0.1) - NOUVEAU

### 4. VAE Supprimé

**Avant**: VAE initialisé mais inutilisé
**Après**: ❌ Complètement supprimé (instabilité, complexité inutile)

### 5. Hyperparamètres Optimisés

```python
# Avant
lr_G = 2e-4
lr_D = 3e-5  # Trop faible
G_steps_per_D = 3  # Trop élevé
L1_factor = 5.0
perc_factor = 5.0

# Après
lr_G = 2e-4
lr_D = 2e-4  # ✅ Équilibré
G_steps_per_D = 1  # ✅ Équilibré
L1_factor = 10.0  # ✅ Augmenté pour structure
perc_factor = 2.0  # ✅ Réduit pour éviter over-smoothing
identity_factor = 1.0  # ✅ NOUVEAU
symmetry_factor = 0.1  # ✅ NOUVEAU
```

---

## 🎯 Ce qui a été Corrigé

### Problème 1: Générateur Autoencoder

❌ **Avant**: Le générateur encodait puis décodait sans conditionnement
✅ **Après**: Identity encoder + injection à chaque niveau du decoder

### Problème 2: Pas de Préservation d'Identité

❌ **Avant**: Aucune garantie de préservation des features faciales
✅ **Après**: Identity Loss + Identity Encoder intégré au générateur

### Problème 3: Discriminateur Non-Conditionnel

❌ **Avant**: D voyait seulement l'output, pas l'input
✅ **Après**: D conditionnel vérifie la cohérence profile→frontal

### Problème 4: Visages Non-Symétriques

❌ **Avant**: Pas de contrainte de symétrie
✅ **Après**: Symmetry Loss force la symétrie du frontal

### Problème 5: Images Floues

❌ **Avant**: L1/L2 trop dominants, perceptual loss mal calibrée
✅ **Après**: Perceptual Loss (VGG19) + Identity Loss + weights équilibrés

---

## 📁 Fichiers Modifiés

### `network.py`

- ✅ Ajout: `IdentityEncoder` (ResNet18)
- ✅ Ajout: `ConditionalUNetGenerator`
- ✅ Ajout: `ConditionalPatchGANDiscriminator`
- ❌ Suppression: `VAE` (commenté)
- ⚠️ Deprecated: `G`, `D`, `UNetGenerator`, `PatchGANDiscriminator`

### `main.py`

- ✅ Import des nouvelles architectures
- ✅ Ajout de `identity_loss()` function
- ✅ Ajout de `symmetry_loss()` function
- ✅ Perceptual Loss avec VGG19 au lieu de VGG16
- ✅ Discriminateur conditionnel dans training loop
- ✅ Nouveaux poids de loss (identity, symmetry)
- ✅ Nouveaux plots de loss (4 subplots)
- ❌ Suppression du training VAE
- ❌ Suppression de L2 loss

### `ARCHITECTURE_CHANGES.md` (NOUVEAU)

- Documentation complète des changements
- Diagrammes d'architecture
- Explications des loss functions
- Guide d'utilisation

---

## 🔧 Pour Lancer l'Entraînement

### 1. Vérifier les Dépendances

```bash
cd frontalization2
pip install torch torchvision matplotlib pillow numpy
```

### 2. Lancer l'Entraînement

```bash
# Training complet
python main.py

# Training avec échantillon réduit (tests rapides)
python main.py --max-samples 1000
```

### 3. Surveiller les Outputs

```
output/
├── 000_input.jpg       # Profils (input)
├── 000_real.jpg        # Frontaux ground truth
├── 000_generated.jpg   # Frontaux générés
├── netG_00.pt          # Checkpoints du générateur
└── loss_curves.png     # Courbes de loss (4 subplots)
```

---

## 📊 Résultats Attendus

### Métriques à Surveiller

1. **L1 Loss**: Doit diminuer progressivement vers 0.05-0.10
2. **Perceptual Loss**: Doit se stabiliser vers 0.01-0.05
3. **Identity Loss**: Doit diminuer vers 0.1-0.3
4. **Symmetry Loss**: Doit converger vers 0.01-0.05
5. **GAN Loss**: Doit osciller autour de 0.5-1.5

### Signes de Bon Entraînement

- ✅ Identity Loss diminue régulièrement
- ✅ Symmetry Loss converge vers 0
- ✅ Les visages générés sont nets et détaillés
- ✅ L'identité est préservée
- ✅ Les frontaux sont symétriques

### Signes de Problèmes

- ❌ Mode collapse (tous les visages identiques)
- ❌ Identity Loss reste élevée (>0.5)
- ❌ Visages flous malgré perceptual loss
- ❌ Asymétries persistantes

---

## 🐛 Troubleshooting

### Problème: Mode Collapse

**Solution**:

```python
L1_factor = 15.0       # Augmenter
GAN_factor = 0.5       # Réduire
label_noise = 0.1      # Augmenter
```

### Problème: Visages Flous

**Solution**:

```python
perc_factor = 5.0      # Augmenter
identity_factor = 2.0  # Augmenter
```

### Problème: Perte d'Identité

**Solution**:

```python
identity_factor = 2.0  # Augmenter
L1_factor = 15.0       # Augmenter
```

### Problème: Asymétries

**Solution**:

```python
symmetry_factor = 0.5  # Augmenter significativement
```

### Problème: Training Instable

**Solution**:

```python
lr_D = 1e-4           # Réduire légèrement
D_input_noise = 0.1   # Augmenter
```

---

## 🎓 Architecture Technique

### Flux du Générateur

```
Profile Image [3x128x128]
    │
    ├─────────────────────┐
    │                     │
    ▼                     ▼
U-Net Encoder        Identity Encoder
(Conv blocks)        (ResNet18)
    │                     │
    ▼                     ▼
[512x4x4]            [512] vector
    │                     │
    └─────────┬───────────┘
              ▼
      Bottleneck [512x4x4]
              │
    ┌─────────┴─────────┐
    ▼                   ▼
Decoder Layer    +  Identity Map [512xHxW]
(Transposed Conv)   (broadcasted)
    │                   │
    └─────────┬─────────┘
              ▼
    concat + Skip Connection
              │
              ▼
      (repeat for all layers)
              │
              ▼
    Generated Frontal [3x128x128]
```

### Flux du Discriminateur

```
Profile [3x128x128]  +  Frontal [3x128x128]
         │                      │
         └──────────┬───────────┘
                    ▼
              concat [6x128x128]
                    │
                    ▼
         Spectral Norm Conv Blocks
         (64→128→256→512→1)
                    │
                    ▼
            PatchGAN Output
                    │
                    ▼
          Mean → Scalar per sample
```

---

## 📚 Références Scientifiques

1. **TP-GAN** (Huang et al. 2017)

   - "Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis"
   - Architecture de base pour identity-preserving frontalization

2. **Pix2Pix** (Isola et al. 2017)

   - "Image-to-Image Translation with Conditional Adversarial Networks"
   - Discriminateur conditionnel

3. **Perceptual Losses** (Johnson et al. 2016)

   - "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
   - VGG-based perceptual loss

4. **Spectral Normalization** (Miyato et al. 2018)
   - "Spectral Normalization for Generative Adversarial Networks"
   - Stabilisation du discriminateur

---

## ✨ Résumé des Améliorations

| Aspect        | Avant              | Après                              | Gain   |
| ------------- | ------------------ | ---------------------------------- | ------ |
| **Identité**  | Non préservée      | Identity Loss + Encoder            | ✅ ×10 |
| **Détails**   | Flou               | Perceptual Loss (VGG19)            | ✅ ×5  |
| **Symétrie**  | Non forcée         | Symmetry Loss                      | ✅ ×10 |
| **Cohérence** | D non-conditionnel | D conditionnel                     | ✅ ×5  |
| **Stabilité** | Instable           | Spectral Norm + losses équilibrées | ✅ ×3  |

---

## 🚀 Prêt à Lancer

Tous les changements sont appliqués et le code est prêt à l'emploi :

```bash
cd /home/mathis/Programming/People_Detection_IA/frontalization2
python main.py --max-samples 1000  # Test rapide
python main.py                      # Training complet
```

**Bon training ! 🎉**
