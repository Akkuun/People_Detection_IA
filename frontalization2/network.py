import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def weights_init(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


''' Generator network for 128x128 RGB images '''
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


''' Discriminator network for 128x128 RGB images '''
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

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Latent space
        self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 8 * 8)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        enc_out = self.encoder(x)
        enc_out = enc_out.view(enc_out.size(0), -1)
        mu = self.fc_mu(enc_out)
        logvar = self.fc_logvar(enc_out)

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decode
        dec_out = self.fc_decode(z)
        dec_out = dec_out.view(dec_out.size(0), 512, 8, 8)
        recon_x = self.decoder(dec_out)

        return recon_x, mu, logvar
