import torch
from VAE.base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class ConditionalVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 img_size:int = 64,
                 **kwargs) -> None:
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size

        self.embed_class = nn.Linear(num_classes, img_size * img_size)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        in_channels += 1 # To account for the extra label channel
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)


        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        # # for image 64
        # self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1] * 4)

        # for image 105
        self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1] )

        hidden_dims.reverse()

        # for image 64
        for i in range(len(hidden_dims) - 1):

            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        # For image 64
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                            kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(hidden_dims[-1], out_channels=1, kernel_size=5, padding=0),  # 32x32 -> 28x28
        )
        

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        # for Image 64
        result = result.view(-1, 512, 1, 1)
        # for Image 105
        # result = result.view(-1, 512, 4, 4)

        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        y = kwargs['labels'].float()
        embedded_class = self.embed_class(y)
        embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embedded_input = self.embed_data(input)

        x = torch.cat([embedded_input, embedded_class], dim = 1)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, y], dim = 1)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        
        recons_logits = args[0]
        x = args[1]
        mu = args[2]
        log_var = args[3]
        B = x.size(0)

        M_N = kwargs['M_N']  # Account for the minibatch samples from the dataset


        # Reconstruction: sum over pixels, average over batch
        recon_loss = F.binary_cross_entropy_with_logits(
            recons_logits, x, reduction="sum"
        ) / B

        # KL (positive)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / B

        loss = recon_loss + M_N * kld
        return {"loss": loss, "Reconstruction_Loss": recon_loss, "KLD": kld}


    def sample(self,
               num_samples:int,
               current_device: int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        y = kwargs['labels'].float()
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, **kwargs)[0]


if __name__ == "__main__":
    # Test ConditionalVAE
    batch_size = 8
    in_channels = 3
    img_size = 64
    num_classes = 10
    latent_dim = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConditionalVAE(
        in_channels=in_channels,
        num_classes=num_classes,
        latent_dim=latent_dim,
        img_size=img_size
    ).to(device)

    x = torch.randn(batch_size, in_channels, img_size, img_size).to(device)
    labels = torch.zeros(batch_size, num_classes).to(device)  
    labels[:, 0] = 1  # Example: all samples belong to class 0
    output = model(x, labels=labels)



