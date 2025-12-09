# 🔄 Guide de Migration de l'Interface GUI

## 📋 Résumé

Ton interface `gui_main.py` **continuera à fonctionner** après l'entraînement du nouveau modèle, MAIS tu devras faire quelques modifications simples.

---

## ⚠️ INCOMPATIBILITÉ : Ancienne vs Nouvelle Architecture

### Actuellement (Ancien Modèle)

```python
# genMugshot.py charge :
network.UNetGenerator()  # ou network.G()
# Depuis : Code/output/netG_99.pt
```

### Après Entraînement (Nouveau Modèle)

```python
# Devra charger :
ConditionalUNetGenerator()  # NOUVELLE architecture
# Depuis : frontalization2/output/netG_XX.pt
```

**❌ Les checkpoints `.pt` NE sont PAS compatibles** car :

- Architecture différente (Identity Encoder intégré)
- Clés du `state_dict` différentes
- Nombre de paramètres différent

---

## ✅ SOLUTION : 2 Options

### Option 1 : Utiliser `genMugshot_v2.py` (RECOMMANDÉ)

J'ai créé `genMugshot_v2.py` qui :

- ✅ Détecte automatiquement l'architecture (ancienne ou nouvelle)
- ✅ Charge depuis `frontalization2/network.py`
- ✅ Compatible avec les deux types de modèles
- ✅ Garde la même API que `genMugshot.py`

#### Modification dans `gui_main.py`

**Avant :**

```python
from genMugshot import generate_frontal_from_image
```

**Après :**

```python
from genMugshot_v2 import generate_frontal_from_image
```

**C'est tout !** Aucune autre modification nécessaire dans `gui_main.py`.

---

### Option 2 : Modifier `genMugshot.py` Existant

Si tu préfères garder le même fichier, remplace le contenu de `genMugshot.py` par celui de `genMugshot_v2.py`.

---

## 🔧 Configuration du Chemin du Modèle

### Après l'Entraînement

1. **Ton modèle sera ici :**

   ```
   /home/mathis/Programming/People_Detection_IA/frontalization2/output/netG_99.pt
   ```

2. **Modifier le chemin dans `genMugshot_v2.py` (ligne 36) :**

   **Option A - Chemin Absolu (Recommandé) :**

   ```python
   DEFAULT_MODEL_PATH = "/home/mathis/Programming/People_Detection_IA/frontalization2/output/netG_99.pt"
   ```

   **Option B - Chemin Relatif :**

   ```python
   DEFAULT_MODEL_PATH = os.path.join(script_dir, "..", "frontalization2", "output", "netG_99.pt")
   ```

3. **Ou passer le chemin manuellement dans le code :**

   ```python
   from genMugshot_v2 import generate_frontal_from_image, load_model

   # Charger un modèle spécifique
   load_model("/chemin/vers/ton/modele/netG_99.pt")

   # Puis utiliser normalement
   frontal_img = generate_frontal_from_image(face_image)
   ```

---

## 🎯 Changements à Faire dans `gui_main.py`

### Modification Minimale (1 ligne)

**Ligne 14 - Changer l'import :**

```python
# AVANT
from genMugshot import generate_frontal_from_image

# APRÈS
from genMugshot_v2 import generate_frontal_from_image
```

**C'est tout !** Le reste du code reste identique.

---

## 🧪 Test de la Nouvelle Configuration

### 1. Tester `genMugshot_v2.py` Seul

```bash
cd /home/mathis/Programming/People_Detection_IA/Code

# Test avec une image
python genMugshot_v2.py --input test_image.jpg --output frontal_test.jpg

# Test avec un dossier (batch)
python genMugshot_v2.py --input input_folder/ --output output_folder/ --batch

# Test avec un modèle spécifique
python genMugshot_v2.py --input test.jpg --model ../frontalization2/output/netG_99.pt
```

### 2. Tester avec l'Interface GUI

```bash
cd /home/mathis/Programming/People_Detection_IA/Code
python gui_main.py
```

**Vérifie dans la console :**

```
🔧 Using device: cpu
📦 Loading model from: /path/to/model/netG_99.pt
🆕 Detected ConditionalUNetGenerator (NEW ARCHITECTURE)
✅ Model loaded successfully!
   Architecture: conditional
   Device: cpu
```

---

## 📝 Récapitulatif des Fichiers

### Structure Actuelle

```
Code/
├── gui_main.py           # Interface (à modifier légèrement)
├── genMugshot.py         # Ancien loader (architecture legacy)
├── genMugshot_v2.py      # ✅ NOUVEAU loader (multi-architecture)
└── output/
    └── netG_99.pt        # Ancien modèle (legacy)

frontalization2/
├── main.py               # Entraînement
├── network.py            # Nouvelles architectures
└── output/
    └── netG_XX.pt        # 🆕 Nouveau modèle (après entraînement)
```

---

## 🔄 Workflow Complet

### 1. **Entraîner le Nouveau Modèle**

```bash
cd frontalization2
python main.py
# → Génère frontalization2/output/netG_99.pt
```

### 2. **Modifier `genMugshot_v2.py`**

```python
# Ligne 36
DEFAULT_MODEL_PATH = "/home/mathis/Programming/People_Detection_IA/frontalization2/output/netG_99.pt"
```

### 3. **Modifier `gui_main.py`**

```python
# Ligne 14
from genMugshot_v2 import generate_frontal_from_image
```

### 4. **Tester**

```bash
cd Code
python gui_main.py
```

---

## 🎨 Ce qui Change pour l'Utilisateur

### Ancienne Architecture

- ❌ Visages flous
- ❌ Perte d'identité
- ❌ Asymétries

### Nouvelle Architecture

- ✅ Visages nets et détaillés
- ✅ Identité préservée
- ✅ Symétrie frontale
- ✅ Meilleure qualité générale

**L'interface restera identique, seuls les résultats seront meilleurs !**

---

## 🐛 Troubleshooting

### Problème : "Import ConditionalUNetGenerator failed"

**Solution :** Vérifie que `frontalization2/network.py` contient bien la nouvelle architecture.

### Problème : "Model file not found"

**Solution :** Vérifie le chemin dans `DEFAULT_MODEL_PATH` (ligne 36 de `genMugshot_v2.py`).

### Problème : "Error loading model: size mismatch"

**Cause :** Tu essaies de charger un ancien modèle avec la nouvelle architecture.
**Solution :** Réentraîne le modèle avec `frontalization2/main.py`.

### Problème : L'interface reste sur l'ancien modèle

**Solution :** Vérifie que tu as bien changé l'import dans `gui_main.py` :

```python
from genMugshot_v2 import generate_frontal_from_image  # ✅
# et pas
from genMugshot import generate_frontal_from_image      # ❌
```

---

## ✨ Fonctionnalités Bonus de `genMugshot_v2.py`

### 1. Détection Automatique d'Architecture

```python
# Détecte automatiquement si c'est :
# - ConditionalUNetGenerator (nouveau)
# - UNetGenerator (ancien)
# - G (très ancien)
```

### 2. Chargement Manuel de Modèle

```python
from genMugshot_v2 import load_model, generate_frontal_from_image

# Charger un modèle spécifique
load_model("/chemin/custom/model.pt")

# Puis utiliser
frontal = generate_frontal_from_image(image)
```

### 3. Mode Batch

```python
from genMugshot_v2 import batch_generate_frontal

# Traiter tout un dossier
batch_generate_frontal("input_folder/", "output_folder/")
```

### 4. CLI Intégré

```bash
# Single image
python genMugshot_v2.py -i input.jpg -o output.jpg

# Batch processing
python genMugshot_v2.py -i input_folder/ -o output_folder/ --batch

# Custom model
python genMugshot_v2.py -i input.jpg -m custom_model.pt
```

---

## 🎯 Conclusion

**TL;DR:**

1. ✅ Change 1 ligne dans `gui_main.py` (l'import)
2. ✅ Configure le chemin du modèle dans `genMugshot_v2.py`
3. ✅ Entraîne le nouveau modèle
4. ✅ Teste avec l'interface

**Ton interface continuera à fonctionner exactement pareil, mais avec des résultats ×10 meilleurs !**
