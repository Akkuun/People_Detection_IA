import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torchvision import models


def weights_init(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


''' Identity Encoder - ResNet18 pour encoder l'identité '''
class IdentityEncoder(nn.Module):
    def __init__(self, identity_dim=512):
        super(IdentityEncoder, self).__init__()
        # Utiliser ResNet18 pré-entraîné sur ImageNet
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remplacer la couche FC par Identity pour obtenir les features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.identity_dim = identity_dim
        
    def forward(self, x):
        # ResNet attend des images 224x224
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.encoder(x)  # retourne un vecteur 512-dim


''' U-Net Generator avec conditionnement d'identité '''
class ConditionalUNetGenerator(nn.Module):
    def __init__(self):
        super(ConditionalUNetGenerator, self).__init__()
        
        # Identity encoder
        self.identity_encoder = IdentityEncoder()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder avec conditionnement d'identité
        # dec4: 512 (bottleneck) + 512 (skip e4) + 512 (identity) = 1536
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(1536, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # dec3: 512 (dec4) + 256 (skip e3) + 512 (identity) = 1280
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(1280, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # dec2: 256 (dec3) + 128 (skip e2) + 512 (identity) = 896
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(896, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # dec1: 128 (dec2) + 64 (skip e1) + 512 (identity) = 704
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(704, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder d'identité
        id_vec = self.identity_encoder(x)  # [B, 512]
        
        # Encoder
        e1 = self.enc1(x)      # [B, 64, 64, 64]
        e2 = self.enc2(e1)     # [B, 128, 32, 32]
        e3 = self.enc3(e2)     # [B, 256, 16, 16]
        e4 = self.enc4(e3)     # [B, 512, 8, 8]

        # Bottleneck
        b = self.bottleneck(e4)  # [B, 512, 4, 4]

        # Decoder avec skip connections + identity conditioning
        # Expand identity vector to spatial maps
        id_map_4x4 = id_vec.unsqueeze(2).unsqueeze(3).repeat(1, 1, 4, 4)
        d4 = torch.cat([b, e4, id_map_4x4], dim=1)  # [B, 1536, 4, 4]
        d4 = self.dec4(d4)  # [B, 512, 8, 8]
        
        id_map_8x8 = id_vec.unsqueeze(2).unsqueeze(3).repeat(1, 1, 8, 8)
        d3 = torch.cat([d4, e3, id_map_8x8], dim=1)  # [B, 1280, 8, 8]
        d3 = self.dec3(d3)  # [B, 256, 16, 16]
        
        id_map_16x16 = id_vec.unsqueeze(2).unsqueeze(3).repeat(1, 1, 16, 16)
        d2 = torch.cat([d3, e2, id_map_16x16], dim=1)  # [B, 896, 16, 16]
        d2 = self.dec2(d2)  # [B, 128, 32, 32]
        
        id_map_32x32 = id_vec.unsqueeze(2).unsqueeze(3).repeat(1, 1, 32, 32)
        d1 = torch.cat([d2, e1, id_map_32x32], dim=1)  # [B, 704, 32, 32]
        d1 = self.dec1(d1)  # [B, 64, 64, 64]

        # Final layer
        out = self.final(d1)  # [B, 3, 128, 128]
        return out


''' Generator network for 128x128 RGB images (DEPRECATED - use ConditionalUNetGenerator) '''
class G(nn.Module):
    
    def __init__(self):
        super(G, self).__init__()
        
        self.main = nn.Sequential(
            # Input HxW = 128x128
            nn.Conv2d(3, 16, 4, 2, 1), # Output HxW = 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, 2, 1), # Output HxW = 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), # Output HxW = 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), # Output HxW = 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), # Output HxW = 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1), # Output HxW = 2x2
            nn.MaxPool2d((2,2)),
            # At this point, we arrive at our low D representation vector, which is 512 dimensional.

            nn.ConvTranspose2d(512, 256, 4, 1, 0, bias = False), # Output HxW = 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False), # Output HxW = 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False), # Output HxW = 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias = False), # Output HxW = 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias = False), # Output HxW = 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1, bias = False), # Output HxW = 128x128
            nn.Tanh()
        )

    
    def forward(self, input):
        output = self.main(input)
        return output


''' Discriminateur conditionnel PatchGAN (Pix2Pix style) '''
class ConditionalPatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(ConditionalPatchGANDiscriminator, self).__init__()
        # Le discriminateur prend en entrée CONCAT(profil, frontal) = 6 channels
        # Cela permet au D de vérifier la cohérence entre l'entrée et la sortie
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))
        )

    def forward(self, profile, frontal):
        # Concaténer le profil et le frontal
        x = torch.cat([profile, frontal], dim=1)  # [B, 6, H, W]
        out = self.model(x)
        # Réduire spatial map à un scalaire par échantillon
        out = out.view(out.size(0), -1).mean(1)
        return out


''' Discriminator network for 128x128 RGB images (DEPRECATED - use ConditionalPatchGANDiscriminator) '''
class D(nn.Module):
    
    def __init__(self):
        super(D, self).__init__()
        # Apply spectral_norm to discriminator conv layers to regularize D
        self.main = nn.Sequential(
                      spectral_norm(nn.Conv2d(3, 16, 4, 2, 1)),
                      nn.LeakyReLU(0.2, inplace = True),
                      spectral_norm(nn.Conv2d(16, 32, 4, 2, 1)),
                      nn.BatchNorm2d(32),
                      nn.LeakyReLU(0.2, inplace = True),
                      spectral_norm(nn.Conv2d(32, 64, 4, 2, 1)),
                      nn.BatchNorm2d(64),
                      nn.LeakyReLU(0.2, inplace = True),
                      spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
                      nn.BatchNorm2d(128),
                      nn.LeakyReLU(0.2, inplace = True),
                      spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
                      nn.BatchNorm2d(256),
                      nn.LeakyReLU(0.2, inplace = True),
                      spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
                      nn.BatchNorm2d(512),
                      nn.LeakyReLU(0.2, inplace = True),
                      spectral_norm(nn.Conv2d(512, 1, 4, 2, 1, bias = False)),
                      # No Sigmoid here: we'll use BCEWithLogitsLoss for stability
                      )
    
    
    def forward(self, input):
        output = self.main(input)
        # output shape: (N, 1, H, W) for a patch discriminator
        # Reduce to one probability per sample by averaging over spatial dimensions
        output = output.view(output.size(0), -1).mean(1)
        return output


class UNetGenerator(nn.Module):
    """DEPRECATED: Use ConditionalUNetGenerator instead for better results"""
    def __init__(self):
        super(UNetGenerator, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections
        d4 = self.dec4(b) + e4
        d3 = self.dec3(d4) + e3
        d2 = self.dec2(d3) + e2
        d1 = self.dec1(d2) + e1

        # Final layer
        out = self.final(d1)
        return out

class PatchGANDiscriminator(nn.Module):
    """DEPRECATED: Use ConditionalPatchGANDiscriminator instead"""
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()
        # Apply spectral norm here as well for the PatchGAN discriminator
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1))
        )

    def forward(self, x):
        out = self.model(x)
        # return raw logits (no sigmoid) and reduce spatial map to a scalar per-sample
        out = out.view(out.size(0), -1).mean(1)
        return out

# Suppression du VAE - non nécessaire pour la frontalization, ajoutait de l'instabilité
# Le conditionnement d'identité via ResNet18 est suffisant et plus efficace
