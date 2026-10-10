"""Small forecasters of the B1 residual."""
import torch
from torch import nn
import torch.nn.functional as F


def features(x, mask, mu, horizon):
    """x: (B, K, H, W) rotated context (NaN → 0 upstream), mask (B, 1, H, W), mu (1, 1, H, W), horizon (B,)
    in frames.
    """
    last = x[:, -1:]
    diffs = (x[:, :-1] - last) * mask
    hmap = (horizon.float() / 160.0).view(-1, 1, 1, 1).expand_as(last)
    return torch.cat([diffs, (last - 1) * mask, mask, mu.expand_as(last), hmap], 1)


def block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(4, co), nn.GELU(),
                         nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(4, co), nn.GELU())


class UNetSmall(nn.Module):
    def __init__(self, k=5, base=16):
        super().__init__()
        cin = (k - 1) + 1 + 1 + 1 + 1
        self.e1, self.e2, self.e3 = block(cin, base), block(base, 2 * base), block(2 * base, 4 * base)
        self.mid = block(4 * base, 4 * base)
        self.d3, self.d2, self.d1 = block(8 * base, 2 * base), block(4 * base, base), block(2 * base, base)
        self.out = nn.Conv2d(base, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)  # start exactly at B1

    def forward(self, x, mask, mu, horizon):
        f = features(x, mask, mu, horizon)
        e1 = self.e1(f)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        m = self.mid(F.avg_pool2d(e3, 2))
        d3 = self.d3(torch.cat([F.interpolate(m, scale_factor=2, mode="bilinear"), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2, mode="bilinear"), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear"), e1], 1))
        return self.out(d1)


class ConvLSTMCell(nn.Module):
    def __init__(self, ci, ch):
        super().__init__()
        self.ch = ch
        self.gates = nn.Conv2d(ci + ch, 4 * ch, 3, padding=1)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.gates(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, k=5, hidden=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(4, hidden, 3, stride=2, padding=1), nn.GELU(),
                                 nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.cell = ConvLSTMCell(hidden, hidden)
        self.dec = nn.Sequential(nn.Conv2d(hidden + 4, hidden, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.out = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, mask, mu, horizon):
        B, K, H, W = x.shape
        last = x[:, -1:]
        hmap = (horizon.float() / 160.0).view(-1, 1, 1, 1).expand(B, 1, H, W)
        mus = mu.expand(B, 1, H, W)
        h = c = None
        for k in range(K):
            step = torch.cat([(x[:, k:k + 1] - last) * mask if k < K - 1 else (last - 1) * mask, mask, mus, hmap], 1)
            e = self.enc(step)
            if h is None:
                h = torch.zeros_like(e)
                c = torch.zeros_like(e)
            h, c = self.cell(e, (h, c))
        up = F.interpolate(h, size=(H, W), mode="bilinear")
        full = torch.cat([up, (last - 1) * mask, mask, mus, hmap], 1)
        return self.out(self.dec(full))


MODELS = {"unet": UNetSmall, "convlstm": ConvLSTM}


def n_params(m):
    return sum(p.numel() for p in m.parameters())
