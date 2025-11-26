# vae_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Conv block utilitaire
def conv_block(in_c, out_c, k=4, s=2, p=1, norm=True, activation=True):
    layers = [nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_c, affine=True))
    if activation:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

def deconv_block(in_c, out_c, k=4, s=2, p=1, norm=True, activation=True):
    layers = [nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_c, affine=True))
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class VAEGenerator(nn.Module):
    """
    VAE-like encoder-decoder generator for image-to-image tasks.
    Input: image 3x128x128
    Output: reconstructed image 3x128x128, plus mu/logvar
    """
    def __init__(self, nc=3, ngf=64, latent_dim=256):
        super().__init__()
        self.nc = nc
        self.ngf = ngf
        self.latent_dim = latent_dim

        # Encoder: 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.enc1 = conv_block(nc, ngf, norm=False)          # 64x64
        self.enc2 = conv_block(ngf, ngf*2)                   # 32x32
        self.enc3 = conv_block(ngf*2, ngf*4)                 # 16x16
        self.enc4 = conv_block(ngf*4, ngf*8)                 # 8x8
        self.enc5 = conv_block(ngf*8, ngf*8)                 # 4x4

        # Flatten final feature map and produce mu/logvar
        self.fc_mu = nn.Linear(ngf*8*4*4, latent_dim)
        self.fc_logvar = nn.Linear(ngf*8*4*4, latent_dim)

        # Project latent back to feature map
        self.fc_dec = nn.Linear(latent_dim, ngf*8*4*4)

        # Decoder
        self.dec5 = deconv_block(ngf*8, ngf*8)   # 4->8
        self.dec4 = deconv_block(ngf*8, ngf*4)   # 8->16
        self.dec3 = deconv_block(ngf*4, ngf*2)   # 16->32
        self.dec2 = deconv_block(ngf*2, ngf)     # 32->64
        # final layer
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # init weights (if tu as init_weights utilitaire, tu peux l'appeler après instanciation)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x):
        x = self.enc1(x)  # B x ngf x 64 x 64
        x = self.enc2(x)  # B x ngf*2 x 32 x 32
        x = self.enc3(x)  # B x ngf*4 x 16 x 16
        x = self.enc4(x)  # B x ngf*8 x 8 x 8
        x = self.enc5(x)  # B x ngf*8 x 4 x 4
        B, C, H, W = x.shape
        x = x.view(B, -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        B = z.size(0)
        x = self.fc_dec(z)
        x = x.view(B, self.ngf*8, 4, 4)
        x = self.dec5(x)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# pour compatibilité import dans ton training:
# from model.vae_generator import VAEGenerator
