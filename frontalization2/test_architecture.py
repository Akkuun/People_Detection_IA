#!/usr/bin/env python3
"""
Script de test rapide pour vérifier l'architecture du Frontalization GAN
"""

import torch
import torch.nn.functional as F
from network import ConditionalUNetGenerator, ConditionalPatchGANDiscriminator, IdentityEncoder

def test_identity_encoder():
    """Test de l'Identity Encoder"""
    print("🔧 Test Identity Encoder...")
    
    encoder = IdentityEncoder()
    encoder.eval()
    
    # Test forward pass
    x = torch.randn(2, 3, 128, 128)  # Batch de 2 images
    with torch.no_grad():
        features = encoder(x)
    
    assert features.shape == (2, 512), f"Shape incorrecte: {features.shape}"
    print(f"   ✅ Output shape: {features.shape}")
    print(f"   ✅ Identity Encoder OK\n")

def test_conditional_generator():
    """Test du Générateur Conditionnel"""
    print("🔧 Test Conditional U-Net Generator...")
    
    generator = ConditionalUNetGenerator()
    generator.eval()
    
    # Test forward pass
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        output = generator(x)
    
    assert output.shape == (2, 3, 128, 128), f"Shape incorrecte: {output.shape}"
    assert output.min() >= -1 and output.max() <= 1, "Output pas dans [-1, 1]"
    print(f"   ✅ Output shape: {output.shape}")
    print(f"   ✅ Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"   ✅ Conditional U-Net Generator OK\n")

def test_conditional_discriminator():
    """Test du Discriminateur Conditionnel"""
    print("🔧 Test Conditional PatchGAN Discriminator...")
    
    discriminator = ConditionalPatchGANDiscriminator()
    discriminator.eval()
    
    # Test forward pass
    profile = torch.randn(2, 3, 128, 128)
    frontal = torch.randn(2, 3, 128, 128)
    
    with torch.no_grad():
        output = discriminator(profile, frontal)
    
    assert output.shape == (2,), f"Shape incorrecte: {output.shape}"
    print(f"   ✅ Output shape: {output.shape}")
    print(f"   ✅ Conditional PatchGAN Discriminator OK\n")

def test_losses():
    """Test des Loss Functions"""
    print("🔧 Test Loss Functions...")
    
    # Mock images
    real = torch.randn(2, 3, 128, 128)
    fake = torch.randn(2, 3, 128, 128)
    
    # L1 Loss
    l1_loss = F.l1_loss(fake, real)
    print(f"   ✅ L1 Loss: {l1_loss.item():.5f}")
    
    # Symmetry Loss
    left_half = fake[:, :, :, :64]
    right_half = fake[:, :, :, 64:]
    right_flipped = torch.flip(right_half, dims=[3])
    sym_loss = F.l1_loss(left_half, right_flipped)
    print(f"   ✅ Symmetry Loss: {sym_loss.item():.5f}")
    
    # Identity Loss (cosine similarity)
    feat1 = torch.randn(2, 512)
    feat2 = torch.randn(2, 512)
    cos_sim = F.cosine_similarity(feat1, feat2, dim=1).mean()
    id_loss = 1.0 - cos_sim
    print(f"   ✅ Identity Loss: {id_loss.item():.5f}")
    
    print(f"   ✅ All Loss Functions OK\n")

def test_full_pipeline():
    """Test du Pipeline Complet"""
    print("🔧 Test Full Pipeline...")
    
    # Créer les modèles
    generator = ConditionalUNetGenerator()
    discriminator = ConditionalPatchGANDiscriminator()
    identity_encoder = IdentityEncoder()
    
    # Mode eval
    generator.eval()
    discriminator.eval()
    identity_encoder.eval()
    
    # Mock data
    profile = torch.randn(2, 3, 128, 128)
    frontal_real = torch.randn(2, 3, 128, 128)
    
    with torch.no_grad():
        # Generator forward
        frontal_fake = generator(profile)
        
        # Discriminator forward (conditional)
        d_real = discriminator(profile, frontal_real)
        d_fake = discriminator(profile, frontal_fake)
        
        # Identity features
        id_real = identity_encoder(frontal_real)
        id_fake = identity_encoder(frontal_fake)
    
    print(f"   ✅ Generator output: {frontal_fake.shape}")
    print(f"   ✅ D(real): {d_real.shape}, mean={d_real.mean():.3f}")
    print(f"   ✅ D(fake): {d_fake.shape}, mean={d_fake.mean():.3f}")
    print(f"   ✅ Identity features: {id_real.shape}")
    print(f"   ✅ Full Pipeline OK\n")

def count_parameters(model):
    """Compte le nombre de paramètres d'un modèle"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_sizes():
    """Affiche la taille des modèles"""
    print("📊 Model Sizes...")
    
    generator = ConditionalUNetGenerator()
    discriminator = ConditionalPatchGANDiscriminator()
    identity_encoder = IdentityEncoder()
    
    g_params = count_parameters(generator)
    d_params = count_parameters(discriminator)
    i_params = count_parameters(identity_encoder)
    
    print(f"   Generator: {g_params:,} parameters ({g_params/1e6:.2f}M)")
    print(f"   Discriminator: {d_params:,} parameters ({d_params/1e6:.2f}M)")
    print(f"   Identity Encoder: {i_params:,} parameters ({i_params/1e6:.2f}M)")
    print(f"   Total: {(g_params+d_params+i_params):,} parameters ({(g_params+d_params+i_params)/1e6:.2f}M)")
    print()

def main():
    print("=" * 60)
    print("🚀 Test de l'Architecture Frontalization GAN")
    print("=" * 60)
    print()
    
    try:
        test_identity_encoder()
        test_conditional_generator()
        test_conditional_discriminator()
        test_losses()
        test_full_pipeline()
        test_model_sizes()
        
        print("=" * 60)
        print("✅ TOUS LES TESTS SONT PASSÉS !")
        print("=" * 60)
        print()
        print("🎉 L'architecture est prête pour l'entraînement !")
        print("📝 Pour lancer l'entraînement:")
        print("   python main.py --max-samples 1000  (test rapide)")
        print("   python main.py                      (training complet)")
        print()
        
    except Exception as e:
        print("=" * 60)
        print("❌ ERREUR DÉTECTÉE !")
        print("=" * 60)
        print(f"\n{type(e).__name__}: {e}\n")
        import traceback
        traceback.print_exc()
        print()

if __name__ == "__main__":
    main()
