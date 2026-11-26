"""Cross-attention transformer, from architecture/transformer.py (Bergner et al., ICLR 2023). Unchanged."""

import math
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def compute_attn(self, q, k):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        return self.dropout(torch.softmax(attn, dim=-1))

    def forward(self, q, k, v):
        return torch.matmul(self.compute_attn(q, k), v)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_token, H, D, D_k, D_v, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.n_token, self.H, self.D_k, self.D_v = n_token, H, D_k, D_v
        self.q = nn.Parameter(torch.empty((1, n_token, D)))
        v = math.sqrt(1 / D_k)
        nn.init.uniform_(self.q, a=-v, b=v)
        self.q_w = nn.Linear(D, H * D_k, bias=False)
        self.k_w = nn.Linear(D, H * D_k, bias=False)
        self.v_w = nn.Linear(D, H * D_v, bias=False)
        self.fc  = nn.Linear(H * D_v, D, bias=False)
        self.attention  = ScaledDotProductAttention(D_k ** 0.5, attn_dropout)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(D, eps=1e-6)

    def get_attn(self, x):
        B, len_seq = x.shape[:2]
        q = self.q_w(self.q).view(1, self.n_token, self.H, self.D_k)
        k = self.k_w(x).view(B, len_seq, self.H, self.D_k)
        return self.attention.compute_attn(q.transpose(1, 2), k.transpose(1, 2))

    def get_logits(self, x):
        """Attention logits BEFORE the softmax, mean over heads and tokens: (B, L).

        The softmax weights of a patch depend on the other patches of its
        group; the logits do not, so they can be compared across groups and
        drawn as a heatmap over the whole image.
        """
        B, len_seq = x.shape[:2]
        q = self.q_w(self.q).view(1, self.n_token, self.H, self.D_k).transpose(1, 2)
        k = self.k_w(x).view(B, len_seq, self.H, self.D_k).transpose(1, 2)
        a = torch.matmul(q / self.attention.temperature, k.transpose(2, 3))
        return a.mean(dim=(1, 2))

    def forward(self, x):
        B, len_seq = x.shape[:2]
        q = self.q_w(self.q).view(1, self.n_token, self.H, self.D_k)
        k = self.k_w(x).view(B, len_seq, self.H, self.D_k)
        v = self.v_w(x).view(B, len_seq, self.H, self.D_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        x = self.attention(q, k, v)
        x = x.transpose(1, 2).contiguous().view(B, self.n_token, -1)
        x = self.dropout(self.fc(x))
        x = x + self.q
        return self.layer_norm(x)


class MLP(nn.Module):
    def __init__(self, D, D_inner, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(D, D_inner)
        self.w_2 = nn.Linear(D_inner, D)
        self.layer_norm = nn.LayerNorm(D, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.dropout(self.w_2(torch.relu(self.w_1(x))))
        return self.layer_norm(x + residual)


class Transformer(nn.Module):
    def __init__(self, n_token, H, D, D_k, D_v, D_inner,
                 attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.crs_attn = MultiHeadCrossAttention(
            n_token, H, D, D_k, D_v, attn_dropout, dropout)
        self.mlp = MLP(D, D_inner, dropout)

    def get_scores(self, x):
        """Attention averaged over heads, then over tokens."""
        attn = self.crs_attn.get_attn(x)
        return attn.mean(dim=1).transpose(1, 2).mean(-1)

    def get_logits(self, x):
        return self.crs_attn.get_logits(x)

    def forward(self, x):
        return self.mlp(self.crs_attn(x))


# ------------------------------------------------------------------
# Self-attention aggregator, as in ips-self-attention.ipynb.
# Patches attend to each other; the score of a patch is the attention it
# RECEIVES (column sum of the attention matrix, mean over heads); the image
# token is the mean of the output tokens. Used only by the ablation runs.
# ------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, H, D, D_k, D_v, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.H, self.D_k, self.D_v = H, D_k, D_v
        self.q_w = nn.Linear(D, H * D_k, bias=False)
        self.k_w = nn.Linear(D, H * D_k, bias=False)
        self.v_w = nn.Linear(D, H * D_v, bias=False)
        self.fc  = nn.Linear(H * D_v, D, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(D, eps=1e-6)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_w(x).view(B, L, self.H, self.D_k).transpose(1, 2)
        k = self.k_w(x).view(B, L, self.H, self.D_k).transpose(1, 2)
        v = self.v_w(x).view(B, L, self.H, self.D_v).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q / self.D_k ** 0.5, k.transpose(2, 3)), dim=-1)
        out = torch.matmul(self.attn_dropout(attn), v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.H * self.D_v)
        out = self.layer_norm(self.dropout(self.fc(out)) + x)
        return out, attn                                  # attn: (B, H, L, L)


class TransformerSA(nn.Module):
    def __init__(self, H, D, D_k, D_v, D_inner, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.sa  = MultiHeadSelfAttention(H, D, D_k, D_v, attn_dropout, dropout)
        self.mlp = MLP(D, D_inner, dropout)

    def get_scores(self, x):
        _, attn = self.sa(x)
        return attn.mean(dim=1).sum(dim=1)                # (B, L)

    def get_logits(self, x):
        return None                                       # depends on the group

    def forward(self, x):
        out, _ = self.sa(x)
        return self.mlp(out).mean(dim=1, keepdim=True)    # (B, 1, D)
