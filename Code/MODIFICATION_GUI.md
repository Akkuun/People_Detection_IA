# 📝 Modification Exacte à Faire dans gui_main.py

## 🎯 Une Seule Ligne à Changer !

### Fichier : `Code/gui_main.py`

---

## AVANT (Ligne 14)

```python
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import Button, Label, Frame, filedialog, StringVar
from PIL import Image, ImageTk
import os
import time
import threading
import subprocess
from ultralytics import YOLO
from Utility.CaptureFace import CaptureFace
from Utility.MugshotPipeline import MugshotPipeline
try:
    from genMugshot import generate_frontal_from_image      # ❌ ANCIEN
    GENMODEL_AVAILABLE = True
except Exception as e:
    print(f"Warning: genMugshot module not available: {e}")
    GENMODEL_AVAILABLE = False
```

---

## APRÈS (Ligne 14)

```python
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import Button, Label, Frame, filedialog, StringVar
from PIL import Image, ImageTk
import os
import time
import threading
import subprocess
from ultralytics import YOLO
from Utility.CaptureFace import CaptureFace
from Utility.MugshotPipeline import MugshotPipeline
try:
    from genMugshot_v2 import generate_frontal_from_image   # ✅ NOUVEAU
    GENMODEL_AVAILABLE = True
except Exception as e:
    print(f"Warning: genMugshot_v2 module not available: {e}")
    GENMODEL_AVAILABLE = False
```

---

## 🔍 Changements Exacts

### Ligne 14

**AVANT :**

```python
from genMugshot import generate_frontal_from_image
```

**APRÈS :**

```python
from genMugshot_v2 import generate_frontal_from_image
```

### Ligne 19 (optionnel - message d'erreur)

**AVANT :**

```python
print(f"Warning: genMugshot module not available: {e}")
```

**APRÈS :**

```python
print(f"Warning: genMugshot_v2 module not available: {e}")
```

---

## ✅ C'est Tout !

**Aucune autre modification nécessaire dans `gui_main.py` !**

Le reste du code (lignes 195-200) reste identique :

```python
# Cette partie NE CHANGE PAS
if GENMODEL_AVAILABLE:
    def generate_frontal_async(img_array, save_path, orient):
        try:
            # Générer l'image frontale
            frontal_img = generate_frontal_from_image(img_array)  # ✅ Fonctionne pareil
            if frontal_img is not None:
                # Sauvegarder l'image frontale générée
                frontal_path = save_path.replace('.jpg', '_frontal_generated.jpg')
                cv2.imwrite(frontal_path, frontal_img)
                print(f"Image frontale générée: {frontal_path}")
        except Exception as e:
            print(f"Error in frontal generation: {e}")
```

---

## 🎨 Interface Visuelle

L'interface reste **exactement la même** :

- ✅ Même fenêtre
- ✅ Mêmes boutons (Screenshot / Mugshot)
- ✅ Même workflow
- ✅ Mêmes fichiers générés

**MAIS** :

- ✅ Meilleure qualité des visages frontaux
- ✅ Identité mieux préservée
- ✅ Symétrie correcte

---

## 🧪 Test Rapide

### 1. Faire la Modification

```bash
cd /home/mathis/Programming/People_Detection_IA/Code
nano gui_main.py  # ou ton éditeur préféré
```

**Ligne 14 :** Change `genMugshot` → `genMugshot_v2`

### 2. Tester

```bash
python gui_main.py
```

### 3. Vérifier dans la Console

Tu devrais voir :

```
🔧 Using device: cpu
📦 Loading model from: /path/to/model.pt
🆕 Detected ConditionalUNetGenerator (NEW ARCHITECTURE)
✅ Model loaded successfully!
```

### 4. Utiliser l'Interface

- Clique sur "Mugshot"
- Le visage frontal sera généré avec le nouveau modèle
- Fichier créé : `mugshot_XXX_frontal_generated.jpg`

---

## 📊 Comparaison Visuelle

### Ancien Flow (genMugshot.py)

```
gui_main.py
    ↓
genMugshot.py
    ↓
network.UNetGenerator()  ← Ancien modèle
    ↓
Code/output/netG_99.pt
```

### Nouveau Flow (genMugshot_v2.py)

```
gui_main.py
    ↓
genMugshot_v2.py
    ↓
ConditionalUNetGenerator()  ← Nouveau modèle
    ↓
frontalization2/output/netG_99.pt
```

---

## 💡 Pourquoi Ça Marche ?

`genMugshot_v2.py` garde **exactement la même signature** :

```python
def generate_frontal_from_image(input_image, output_path=None):
    """
    Args:
        input_image: numpy array (cv2) ou chemin
        output_path: où sauvegarder (optionnel)

    Returns:
        numpy array (BGR) ou None
    """
```

→ Ton `gui_main.py` appelle la fonction **exactement pareil**
→ Seul l'**intérieur** de la fonction a changé (nouveau modèle)

---

## 🔄 Rollback (si besoin)

Si tu veux revenir à l'ancien modèle :

```python
# Ligne 14
from genMugshot import generate_frontal_from_image  # Retour à l'ancien
```

---

## ✨ RÉSUMÉ

**1 ligne à changer :**

```python
# gui_main.py ligne 14
from genMugshot_v2 import generate_frontal_from_image
```

**+ 1 configuration :**

```python
# genMugshot_v2.py ligne 36
DEFAULT_MODEL_PATH = "/chemin/vers/frontalization2/output/netG_99.pt"
```

**= Interface prête avec nouveau modèle ! 🎉**
