import torch.nn as nn
import torch
from net import STgramMFN, ArcMarginProduct


BATCH_NORM_EPSILON = 1e-5

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ContrastiveHead(nn.Module):
    def __init__(self, channels_in, out_dim=128, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i != num_layers - 1:
                dim, relu = channels_in, True
            else:
                dim, relu = out_dim, False
            self.layers.append(nn.Linear(channels_in, dim, bias=False))
            bn = nn.BatchNorm1d(dim, eps=BATCH_NORM_EPSILON, affine=True)
            if i == num_layers - 1:
                nn.init.zeros_(bn.bias)
            self.layers.append(bn)
            if relu:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for b in self.layers:
            x = b(x)
        return x

class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to
    obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average
    pooling layer.
    """

    def __init__(self, encoder, projection_dim, n_features):
        """Original SimCLR Implementation from Spijkervet Code"""
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i)
        # where σ is a ReLU non-linearity.
        # For v2 we have an added layer
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j


class SimCLRv2(nn.Module):
    """SimCLRv2 Implementation
        Using ResNet architecture from Pytorch converter which includes projection head.
    """
    def __init__(self, num_class=41, c_dim=128, pretrained_weights: str = None, use_arcface=False):
        super(SimCLRv2, self).__init__()
        self.use_arcface = use_arcface
        self.encoder = STgramMFN(num_class, c_dim=c_dim, use_arcface=use_arcface)
        self.projector = ContrastiveHead(channels_in=c_dim)
        self.classifier = nn.Linear(c_dim, num_class)

        if pretrained_weights:
            self.encoder.load_state_dict(torch.load(pretrained_weights, map_location='cpu')['encoder'])
            self.projector.load_state_dict(torch.load(pretrained_weights, map_location='cpu')['projector'])

    def forward(self, x_wav, x_mel, label=None):
        # # x_wav [B, N, 1, L] x_mel [B, N, 1, M, F]
        # b, n = x_wav.size(0), x_wav.size(1)
        # # reshape
        # x_wav = x_wav.reshape(b*n, x_wav.size(2), x_wav.size(3))
        # x_mel = x_mel.reshape(b*n, x_mel.size(2), x_mel.size(3), x_mel.size(4))

        # h
        out, h = self.encoder(x_wav, x_mel, label)
        logits = out if self.use_arcface else self.classifier(h)
        # z
        z = self.projector(h)
        return logits, z


class SimCLRv2_ft(nn.Module):
    """Take a pretrained SimCLRv2 Model and Finetune with linear layer"""
    def __init__(self, simclrv2_model, n_classes=41, c_dim=128, pretrain=True, use_arcface=False, m=0.5, s=30, sub=1):
        super(SimCLRv2_ft, self).__init__()
        self.encoder = simclrv2_model.encoder
        # From v2 paper, we just need the first layer from projector
        self.projector = torch.nn.Sequential(*(list(simclrv2_model.projector.children())[0][:2])) if pretrain else nn.Identity()
        # contrain used
        self.contrain_projector = ContrastiveHead(channels_in=c_dim)
        # Hack
        linear_in_features = self.projector[0].out_features if pretrain else c_dim
        self.linear = nn.Linear(linear_in_features, n_classes)
        self.arcface = ArcMarginProduct(in_features=linear_in_features, out_features=n_classes,
                                        m=m, s=s, sub=sub) if use_arcface else use_arcface

    def forward(self, x_wav, x_mel, label=None):
        out, h = self.encoder(x_wav, x_mel)
        z = self.contrain_projector(h)
        h_prime = self.projector(h)
        y_hat = self.arcface(h_prime, label) if self.arcface else self.linear(h_prime)
        return y_hat, h_prime, z