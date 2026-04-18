import torch
import torch.nn as nn


# -------------------------
# Safe GroupNorm
# -------------------------
def group_norm(channels):

    groups = min(8, channels)

    while channels % groups != 0:
        groups -= 1

    return nn.GroupNorm(groups, channels)


# -------------------------
# Residual Block
# -------------------------
class ResidualBlock(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.gn1 = group_norm(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.gn2 = group_norm(out_c)

        self.relu = nn.ReLU(inplace=True)

        if in_c != out_c:
            self.skip = nn.Conv2d(in_c, out_c, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):

        identity = self.skip(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out = out + identity
        out = self.relu(out)

        return out


# -------------------------
# Attention Gate
# -------------------------
class AttentionBlock(nn.Module):

    def __init__(self, g_c, x_c, inter_c):
        super().__init__()

        self.Wg = nn.Conv2d(g_c, inter_c, 1)
        self.Wx = nn.Conv2d(x_c, inter_c, 1)

        self.psi = nn.Sequential(
            nn.Conv2d(inter_c, 1, 1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g1 = self.Wg(g)
        x1 = self.Wx(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# -------------------------
# Attention Residual UNet
# -------------------------
class AttentionResUNet(nn.Module):

    def __init__(self, in_channels=384, out_channels=3):
        super().__init__()

        # Encoder
        self.enc1 = ResidualBlock(in_channels, 128)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualBlock(128, 256)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualBlock(256, 512)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(512, 1024),
            ResidualBlock(1024,1024)
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att1 = AttentionBlock(512, 512, 256)
        self.dec1 = ResidualBlock(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att2 = AttentionBlock(256, 256, 128)
        self.dec2 = ResidualBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 = AttentionBlock(128, 128, 64)
        self.dec3 = ResidualBlock(256, 128)

        # Output
        self.out = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):

        # -------------------------
        # Global Residual (Photon Trick)
        # -------------------------
        input_mean = x.mean(dim=1, keepdim=True)
        input_mean = input_mean.repeat(1, 3, 1, 1)

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder stage 1
        u1 = self.up1(b)
        a1 = self.att1(u1, e3)
        d1 = torch.cat([u1, a1], dim=1)
        d1 = self.dec1(d1)

        # Decoder stage 2
        u2 = self.up2(d1)
        a2 = self.att2(u2, e2)
        d2 = torch.cat([u2, a2], dim=1)
        d2 = self.dec2(d2)

        # Decoder stage 3
        u3 = self.up3(d2)
        a3 = self.att3(u3, e1)
        d3 = torch.cat([u3, a3], dim=1)
        d3 = self.dec3(d3)

        # Output
        out = self.out(d3)

        # Global Residual Addition
        out = out + input_mean

        return out