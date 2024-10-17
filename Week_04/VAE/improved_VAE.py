import torch.nn as nn
import torch
import torch.nn.functional as F
# VAE 정의
class my_VAE(nn.Module):
    def __init__(self, latent_dim1=50, latent_dim2=50, beta=5e-3):
        super(my_VAE, self).__init__()
        self.beta = beta
 
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1024), 
            nn.ReLU(),
        )
        
        self.fc_mu1 = nn.Linear(1024, latent_dim1)
        self.fc_logvar1 = nn.Linear(1024, latent_dim1)

        self.fc_mu2 = nn.Linear(latent_dim1, latent_dim2)
        self.fc_logvar2 = nn.Linear(latent_dim1, latent_dim2)

        self.decoder_input2 = nn.Linear(latent_dim2, 1024)

        self.decoder = nn.Sequential(
            nn.Linear(1024, 256 * 3 * 3),
            nn.ReLU(),
            nn.Unflatten(1, (256, 3, 3)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=2, output_padding=0),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu1 = self.fc_mu1(h)
        logvar1 = self.fc_logvar1(h)
        z1 = self.reparameterize(mu1, logvar1)
        mu2 = self.fc_mu2(z1)
        logvar2 = self.fc_logvar2(z1)
        z2 = self.reparameterize(mu2, logvar2)
        return z1, mu1, logvar1, z2, mu2, logvar2

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z1, z2):
        h = F.relu(self.decoder_input2(z2))
        recon_x = self.decoder(h)
        return recon_x

    def forward(self, x):
        z1, mu1, logvar1, z2, mu2, logvar2 = self.encode(x)
        recon_x = self.decode(z1, z2)
        return recon_x, mu1, logvar1, mu2, logvar2

def loss_function(recon_x, x, mu1, logvar1, mu2, logvar2, beta=5e-3):
    recon_x = recon_x.view_as(x)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
    KLD2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
    return BCE + beta * (KLD1 + KLD2)