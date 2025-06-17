import math
import torch
from torch import nn
import torch.nn.functional as F
from models.subNets.position_embedding import PositionalEmbedding


def FFT_function(x, lengths):
    batch = x.size()[0]
    rx = torch.zeros_like(x) + 0.0 * 1j
    for i in range(batch):
        l = lengths[i]
        rx[i, :l, :] = torch.fft.fft(x[i, :l, :], dim=0)
    return rx


def IFFT_function(x, lengths):
    batch = x.size()[0]
    rx = torch.zeros_like(x) + 0.0 * 1j
    for i in range(batch):
        l = lengths[i]
        rx[i, :l, :] = torch.fft.ifft(x[i, :l, :], dim=0)
    return rx


class Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return x


class Self_Fusion(nn.Module):
    def __init__(self, dim, hid_dim=40, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)

        self.Ww = nn.Sequential(
            nn.Linear(dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, dim))
        self.Wb = nn.Sequential(
            nn.Linear(dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, dim))
        
        nn.init.xavier_normal_(self.Ww[0].weight)
        nn.init.xavier_normal_(self.Wb[0].weight)

    def relu(self, z):
        magnitude = torch.abs(z)
        phase = torch.angle(z)
        activated = torch.where(
            magnitude > 0,
            magnitude * torch.exp(1j * phase),
            torch.tensor(0.0 + 0.0 * 1j, device=z.device).detach()
        )
        return activated

    def forward(self, x, len_x):
        batch = x.size()[0]
        x = self.dropout(x)
        c = torch.sum(x, dim=1) / len_x.unsqueeze(-1)
        x_freq = FFT_function(x, len_x)
        W = self.Ww(c) + torch.tensor(0.0 * 1j, device=x.device).detach()
        b = self.Wb(c) + torch.tensor(0.0 * 1j, device=x.device).detach()
        filtered = x_freq * (1 + W.unsqueeze(1)) + b.unsqueeze(1)
        activated = self.relu(filtered)
        x_recon = IFFT_function(activated, len_x)
        x_recon = x_recon.real
        return x_recon


class Cross_Fusion(nn.Module):
    def __init__(self, dim, hid_dim=40, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.W1 = nn.Sequential(
            nn.Linear(dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, dim))
        self.B1 = nn.Sequential(
            nn.Linear(dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, dim))
        
        self.W2 = nn.Sequential(
            nn.Linear(dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, dim))
        self.B2 = nn.Sequential(
            nn.Linear(dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, dim))
        
        nn.init.xavier_normal_(self.W1[0].weight)
        nn.init.xavier_normal_(self.B1[0].weight)
        nn.init.xavier_normal_(self.W2[0].weight)
        nn.init.xavier_normal_(self.B2[0].weight)

    def relu(self, z):
        magnitude = torch.abs(z)
        phase = torch.angle(z)
        activated = torch.where(
            magnitude > 0,
            magnitude * torch.exp(1j * phase),
            torch.tensor(0.0 + 0.0 * 1j, device=z.device).detach()
        )
        return activated

    def forward(self, x, y, z, len_x, len_y, len_z):
        batch = x.size()[0]
        x = self.dropout(x)
        y = self.dropout(y)
        z = self.dropout(z)
        x_freq = FFT_function(x, len_x)
        c1 = torch.sum(y, dim=1) / len_y.unsqueeze(-1)
        c2 = torch.sum(z, dim=1) / len_z.unsqueeze(-1)
        W1 = self.W1(c1) + torch.tensor(0.0 * 1j, device=x.device).detach()
        b1 = self.B1(c1) + torch.tensor(0.0 * 1j, device=x.device).detach()
        W2 = self.W2(c2) + torch.tensor(0.0 * 1j, device=x.device).detach()
        b2 = self.B2(c2) + torch.tensor(0.0 * 1j, device=x.device).detach()
        filtered = x_freq * (1 + (W1.unsqueeze(1) + W2.unsqueeze(1))/2.0) + (b1.unsqueeze(1) + b2.unsqueeze(1))/2.0
        activated = self.relu(filtered)
        x_recon = IFFT_function(activated, len_x)
        x_recon = x_recon.real
        return x_recon


class MultiModalFourier(nn.Module):
    def __init__(self, modal_dim, hid_dim, dropout=0., device='cuda'):
        super().__init__()
        self.device = device
        self.norm = Norm(modal_dim)

        self.other2video = Cross_Fusion(modal_dim, hid_dim, dropout)
        self.other2audio = Cross_Fusion(modal_dim, hid_dim, dropout)
        self.other2text = Cross_Fusion(modal_dim, hid_dim, dropout)

        self.text2text = Self_Fusion(modal_dim, hid_dim, dropout)
        self.audio2audio = Self_Fusion(modal_dim, hid_dim, dropout)
        self.video2video = Self_Fusion(modal_dim, hid_dim, dropout)

    def forward(self, xt, xa, xv, len_t, len_a, len_v):
        xtn = self.norm(xt)
        xan = self.norm(xa)
        xvn = self.norm(xv)

        x_o2t = self.other2text(xtn, xan, xvn, len_t, len_a, len_v)
        x_o2a = self.other2audio(xan, xtn, xvn, len_a, len_t, len_v)
        x_o2v = self.other2video(xvn, xtn, xan, len_v, len_t, len_a)
        x_t2t = self.text2text(xtn, len_t)
        x_a2a = self.audio2audio(xan, len_a)
        x_v2v = self.video2video(xvn, len_v)

        ft = x_o2t + x_t2t
        fa = x_o2a + x_a2a
        fv = x_o2v + x_v2v
        ft = ft + xt
        fa = fa + xa
        fv = fv + xv
        return ft, fa, fv


class FeedForwardAll(nn.Module):
    def __init__(self, m_dim, mult=2, dropout=0.):
        super().__init__()
        self.text_net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(m_dim, (m_dim // mult)),
            nn.ReLU(),
            nn.Linear(m_dim // mult, m_dim),
        )
        self.audio_net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(m_dim, m_dim // mult),
            nn.ReLU(),
            nn.Linear(m_dim // mult, m_dim),
        )
        self.video_net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(m_dim, m_dim // mult),
            nn.ReLU(),
            nn.Linear(m_dim // mult, m_dim),
        )
        self.norm = Norm(m_dim)

    def forward(self, xt, xa, xv):
        ft = self.norm(xt)
        fa = self.norm(xa)
        fv = self.norm(xv)
        ft = self.text_net(ft)
        fa = self.audio_net(fa)
        fv = self.video_net(fv)
        out_t = ft + xt
        out_a = fa + xa
        out_v = fv + xv
        return out_t, out_a, out_v


class Simple_Transformer_Block(nn.Module):
    def __init__(self, depth, modal_dim, hid_dim, att_dropout=0., ff_expansion=2,
                 ff_dropout=0., device='cuda'):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList([])
        self.depth = depth

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiModalFourier(modal_dim=modal_dim, hid_dim=hid_dim, dropout=att_dropout, device=device),
                FeedForwardAll(modal_dim, mult=ff_expansion, dropout=ff_dropout)
            ]))

    def forward(self, xt, xa, xv, len_t, len_a, len_v):
        for attn, ff in self.layers:
            xt, xa, xv = attn(xt, xa, xv, len_t, len_a, len_v)
            xt, xa, xv = ff(xt, xa, xv)
        return xt, xa, xv


class Simple_Transformer(nn.Module):
    def __init__(self, device, depth, modal_dim, hid_dim, attn_dropout=0., ff_expansion=4,
                 ff_dropout=0., learnable_pos_emb=False, emb_dropout=0., max_len=500):
        super().__init__()
        self.device = device
        self.modal_dim = modal_dim
        self.pos_embed = PositionalEmbedding(modal_dim, max_seq_len=max_len, dropout=emb_dropout,
                                             learnable=learnable_pos_emb)
        self.transformer = Simple_Transformer_Block(depth, modal_dim, hid_dim, attn_dropout, ff_expansion, ff_dropout)

    def forward(self, modality_inputs, modality_masks, modality_lengths):
        mask_t, mask_a, mask_v = modality_masks
        xt, xa, xv = modality_inputs
        len_t, len_a, len_v = modality_lengths
        xt = self.pos_embed(xt) * mask_t
        xa = self.pos_embed(xa) * mask_a
        xv = self.pos_embed(xv) * mask_v

        out_t, out_a, out_v = self.transformer(xt, xa, xv, len_t, len_a, len_v)
        out_t = out_t * mask_t
        out_a = out_a * mask_a
        out_v = out_v * mask_v
        return out_t, out_a, out_v
