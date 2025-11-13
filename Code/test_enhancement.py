#!/usr/bin/env python3
"""
Script pour tester les différentes options d'amélioration d'image
"""

import cv2
import os
from Utility.CaptureFace import CaptureFace

def test_enhancement_options():
    """Tester les différentes options d'amélioration sur une image existante"""
    
    # Dossier des images sauvegardées
    saved_dir = "SavedImages"
    
    # Prendre la dernière image de mugshot comme test
    mugshot_files = [f for f in os.listdir(saved_dir) if f.startswith('mugshot_')]
    if not mugshot_files:
        print("❌ Aucune image de mugshot trouvée pour le test")
        return
    
    # Prendre la dernière image
    latest_file = sorted(mugshot_files)[-1]
    image_path = os.path.join(saved_dir, latest_file)
    
    print(f"🖼️  Test sur l'image: {latest_file}")
    
    # Charger l'image originale
    original = cv2.imread(image_path)
    if original is None:
        print("❌ Impossible de charger l'image")
        return
    
    # Créer le générateur
    generator = CaptureFace()
    
    # Tester les différentes options
    options = {
        "none": "Aucune amélioration",
        "minimal": "Amélioration minimale", 
        "normal": "Amélioration normale"
    }
    
    print("\n🔍 Comparaison des améliorations:")
    print("-" * 40)
    
    for option, description in options.items():
        # Appliquer l'amélioration
        enhanced, orientation = generator.create_mugshot(original, enhance=option)
        
        # Sauvegarder le résultat
        output_file = f"SavedImages/test_enhance_{option}_{latest_file}"
        cv2.imwrite(output_file, enhanced)
        
        print(f"✅ {description:20} -> {output_file}")
    
    print(f"\n📊 Images sauvegardées dans {saved_dir}/")
    print("💡 Comparez visuellement pour choisir la meilleure option!")

if __name__ == "__main__":
    test_enhancement_options()
