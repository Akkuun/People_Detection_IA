#!/usr/bin/env python3
"""
Script de test pour valider le pipeline de mugshots
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
from Utility.MugshotPipeline import MugshotPipeline
from Utility.CaptureFace import CaptureFace

def create_test_face():
    """Créer une image de test simple pour simuler un visage"""
    # Créer une image test avec des formes géométriques pour simuler un visage
    img = np.zeros((200, 150, 3), dtype=np.uint8)
    
    # Fond gris
    img[:] = (50, 50, 50)
    
    # Visage (cercle)
    cv2.circle(img, (75, 100), 60, (180, 160, 140), -1)
    
    # Yeux
    cv2.circle(img, (60, 80), 8, (0, 0, 0), -1)
    cv2.circle(img, (90, 80), 8, (0, 0, 0), -1)
    
    # Nez
    cv2.circle(img, (75, 100), 3, (150, 130, 110), -1)
    
    # Bouche
    cv2.ellipse(img, (75, 120), (15, 8), 0, 0, 180, (80, 60, 60), -1)
    
    return img

def test_pipeline():
    """Tester le pipeline complet"""
    print("🧪 Test du pipeline de mugshots...")
    
    try:
        # Initialiser le pipeline
        pipeline = MugshotPipeline()
        print("✅ Pipeline initialisé avec succès")
        
        # Créer une image de test
        test_face = create_test_face()
        print("✅ Image de test créée")
        
        # Tester la classification d'orientation
        orientation = pipeline.classify_orientation(test_face)
        print(f"✅ Classification d'orientation : {orientation}")
        
        # Tester le traitement complet
        processed_face, final_orientation = pipeline.process_mugshot(test_face)
        print(f"✅ Traitement complet réussi : {final_orientation}")
        
        if processed_face is not None:
            # Sauvegarder l'image de test
            os.makedirs("SavedImages", exist_ok=True)
            cv2.imwrite("SavedImages/test_pipeline_input.jpg", test_face)
            cv2.imwrite("SavedImages/test_pipeline_output.jpg", processed_face)
            print("✅ Images de test sauvegardées dans SavedImages/")
            
            print("\n🎉 Tous les tests sont passés avec succès !")
            print("📸 Le pipeline est prêt pour la détection en temps réel")
            
        else:
            print("❌ Erreur : L'image traitée est None")
            
    except Exception as e:
        print(f"❌ Erreur lors du test : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
