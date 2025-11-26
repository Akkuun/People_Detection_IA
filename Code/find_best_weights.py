#!/usr/bin/env python3
"""
🎯 QUICK TEST: Trouver les meilleurs poids en 5 epochs chacun
Compare 3 configurations pour trouver celle qui donne la meilleure qualité
"""

import subprocess
import json
import os
import shutil
from datetime import datetime

# ============================================
# 3 CONFIGURATIONS À TESTER
# ============================================

CONFIGS = {
    "recon_heavy": {
        'description': '🔴 RECON-HEAVY: Reconstruction maximale (mugshots)',
        'recon_weight': 25.0,
        'perceptual_weight': 1.5,
        'adv_weight': 0.15,
        'kl_weight': 0.008,
        'num_epochs': 5,
        'auto_tune_strategy': 'conservative'
    },
    
    "balanced": {
        'description': '🟡 BALANCED: Équilibre général',
        'recon_weight': 15.0,
        'perceptual_weight': 1.5,
        'adv_weight': 0.2,
        'kl_weight': 0.01,
        'num_epochs': 5,
        'auto_tune_strategy': 'smart'
    },
    
    "gan_stable": {
        'description': '🟢 GAN-STABLE: GAN très stable, reconstruction correcte',
        'recon_weight': 12.0,
        'perceptual_weight': 1.2,
        'adv_weight': 0.12,
        'kl_weight': 0.008,
        'num_epochs': 5,
        'auto_tune_strategy': 'conservative'
    }
}

def modify_config(config_name, params):
    """Modifie trainVAEGAN.py avec les nouveaux paramètres"""
    
    print(f"\n{'='*60}")
    print(f"🔧 Modification CONFIG pour: {config_name}")
    print(f"{'='*60}")
    
    filepath = 'trainVAEGAN.py'
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Remplacer les poids
    replacements = {
        "'recon_weight': 10.0,": f"'recon_weight': {params['recon_weight']},",
        "'perceptual_weight': 1.0,": f"'perceptual_weight': {params['perceptual_weight']},",
        "'adv_weight': 0.25,": f"'adv_weight': {params['adv_weight']},",
        "'kl_weight': 0.01,": f"'kl_weight': {params['kl_weight']},",
        "'num_epochs': 10,": f"'num_epochs': {params['num_epochs']},",
        "'auto_tune_strategy': 'smart',": f"'auto_tune_strategy': '{params['auto_tune_strategy']}',"
    }
    
    for old, new in replacements.items():
        if old in content:
            content = content.replace(old, new)
            print(f"  ✓ {old} → {new}")
        else:
            print(f"  ⚠ Pattern not found: {old}")
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✅ CONFIG modifiée\n")

def run_training(config_name):
    """Lance l'entraînement"""
    
    print(f"🚀 Lancement de l'entraînement: {config_name}")
    print(f"{'='*60}")
    
    # Créer dossier de sauvegarde pour ce test
    test_dir = f"test_weights/{config_name}"
    os.makedirs(test_dir, exist_ok=True)
    
    # Lancer l'entraînement
    try:
        result = subprocess.run(
            ["python3", "trainVAEGAN.py"],
            capture_output=True,
            text=True,
            timeout=3600  # 1 heure max
        )
        
        # Sauvegarder les logs
        log_file = f"{test_dir}/train.log"
        with open(log_file, 'w') as f:
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
        
        print(f"\n✅ Entraînement terminé!")
        print(f"📝 Logs sauvegardés dans: {log_file}")
        
        # Copier les images générées
        if os.path.exists('samples_vaegan'):
            dst = f"{test_dir}/samples"
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree('samples_vaegan', dst)
            print(f"📸 Images sauvegardées dans: {dst}")
        
        # Copier le meilleur modèle
        if os.path.exists('checkpoints_vaegan/best_vaegan_model.pth'):
            shutil.copy(
                'checkpoints_vaegan/best_vaegan_model.pth',
                f"{test_dir}/best_model.pth"
            )
            print(f"💾 Meilleur modèle copié")
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"❌ Entraînement dépassé (timeout)")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def extract_final_losses(log_file):
    """Extrait les loss finales du fichier log"""
    
    final_losses = {}
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Chercher la dernière ligne avec les loss values
        for line in reversed(lines):
            if 'loss_G_recon' in line:
                # Parser les loss values
                import re
                recon_match = re.search(r'loss_G_recon=([\d.]+)', line)
                adv_match = re.search(r'loss_G_adv=([\d.]+)', line)
                kl_match = re.search(r'loss_G_kl=([\d.]+)', line)
                
                if recon_match:
                    final_losses['recon'] = float(recon_match.group(1))
                if adv_match:
                    final_losses['adv'] = float(adv_match.group(1))
                if kl_match:
                    final_losses['kl'] = float(kl_match.group(1))
                
                break
    except Exception as e:
        print(f"  ⚠ Impossible de parser les losses: {e}")
    
    return final_losses

def main():
    """Exécute le test complet"""
    
    print("\n" + "="*60)
    print("🎯 QUICK WEIGHT FINDER - Test 3 configurations")
    print("="*60)
    
    results = {}
    
    for config_name, config_params in CONFIGS.items():
        print(f"\n\n{config_params['description']}")
        
        # Modifier la CONFIG
        modify_config(config_name, config_params)
        
        # Lancer l'entraînement
        success = run_training(config_name)
        
        if success:
            # Extraire les losses
            log_file = f"test_weights/{config_name}/train.log"
            losses = extract_final_losses(log_file)
            results[config_name] = {
                'config': config_params,
                'losses': losses,
                'dir': f"test_weights/{config_name}"
            }
            
            print(f"\n📊 Résultats finaux:")
            for loss_type, value in losses.items():
                print(f"  - loss_G_{loss_type}: {value:.4f}")
        
        print(f"\n{'='*60}")
    
    # Résumé final
    print("\n\n" + "="*60)
    print("📊 RÉSUMÉ FINAL")
    print("="*60)
    
    summary_file = "test_weights/RESULTS.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Test Date: {datetime.now()}\n\n")
        
        for config_name, result in results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"Configuration: {config_name}\n")
            f.write(f"{'='*60}\n")
            
            f.write(f"Parameters:\n")
            for param, value in result['config'].items():
                f.write(f"  - {param}: {value}\n")
            
            f.write(f"\nFinal Losses:\n")
            for loss_type, value in result['losses'].items():
                f.write(f"  - loss_G_{loss_type}: {value:.4f}\n")
            
            f.write(f"\nSamples directory: {result['dir']}/samples\n")
    
    print(f"\n✅ Résultats sauvegardés dans: {summary_file}")
    print(f"\n📂 Dossiers de test créés dans: test_weights/")
    print(f"   - test_weights/recon_heavy/samples/")
    print(f"   - test_weights/balanced/samples/")
    print(f"   - test_weights/gan_stable/samples/")
    
    print(f"\n💡 Conseils:")
    print(f"   1. Ouvrez les dossiers samples pour comparer les images")
    print(f"   2. Lisez test_weights/RESULTS.txt pour les loss values")
    print(f"   3. Choisissez la meilleure config")
    print(f"   4. Copiez la CONFIG dans trainVAEGAN.py")
    print(f"   5. Lancez un entraînement long (30 epochs) avec cette config")

if __name__ == '__main__':
    main()
