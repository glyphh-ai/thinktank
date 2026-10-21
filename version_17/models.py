"""
Models — one bank of tiny models, four ways to answer a circuit
===============================================================
Bank:  16 tiny MLPs (2 -> 8 -> 1), each trained alone on one gate's
       4-row truth table, then frozen. The bottom rung of the ladder.

A  monolithic   transformer reads the whole circuit, outputs the bit.
B  looped       step-by-step loop; one shared learned executor MLP.
C  layered      same loop; executor = learned soft routing over the
                frozen bank: r = sum_k softmax(router[op])_k * tiny_k(a, b)
C0 layered-rnd  C, but the bank is 16 frozen RANDOM MLPs (never trained
                on gates) — separates "what the tiny models know" from
                "the shape of the architecture".

B, C, C0 share the same loop and the same learned operand addressing
(address token -> soft weights over the 6 slots). Only the executor
differs. All looped values are probabilities; no rounding between steps.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from circuits import N_GATES, N_OPS, N_SLOTS, P1, P2, TABLE

HID = 8


# ============================================================================
# The bank of tiny models
# ============================================================================
class Bank(nn.Module):
    """N_GATES independent 2->HID->1 MLPs, evaluated in one batched call."""

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W1 = nn.Parameter(torch.randn(N_GATES, 2, HID, generator=g))
        self.b1 = nn.Parameter(torch.randn(N_GATES, HID, generator=g) * 0.1)
        self.W2 = nn.Parameter(torch.randn(N_GATES, HID, generator=g) / math.sqrt(HID))
        self.b2 = nn.Parameter(torch.zeros(N_GATES))

    def forward(self, a, b):
        """a, b: (B,) probabilities -> (B, N_GATES) output probabilities."""
        ab = torch.stack([a, b], -1)                                  # (B,2)
        h = torch.tanh(torch.einsum("bi,kih->bkh", ab, self.W1) + self.b1)
        return torch.sigmoid((h * self.W2).sum(-1) + self.b2)        # (B,K)


def train_bank(seed=0, steps=2000):
    """Each tiny model learns its own gate, alone, on 4 rows. Then frozen."""
    bank = Bank(seed)
    a = torch.tensor([0., 0., 1., 1.])
    b = torch.tensor([0., 1., 0., 1.])
    y = torch.tensor(TABLE.T, dtype=torch.float32)                   # (4,K)
    opt = torch.optim.Adam(bank.parameters(), lr=3e-2)
    for _ in range(steps):
        loss = F.binary_cross_entropy(bank(a, b), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return freeze(bank)


def random_bank(seed=0):
    return freeze(Bank(seed))


def freeze(bank):
    for p in bank.parameters():
        p.requires_grad_(False)
    return bank


# ============================================================================
# Looped models (B, C, C0)
# ============================================================================
class Looped(nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # address token -> logits over the 6 slots
        self.addr = nn.Parameter(torch.randn(N_SLOTS, N_SLOTS, generator=g) * 0.01)

    def read(self, slots, tok):
        return (F.softmax(self.addr[tok], -1) * slots).sum(-1)

    def forward(self, x, ops, addr):
        """x (B,4) bits, ops (B,D), addr (B,D,2) -> (B,) P(answer = 1)."""
        slots = torch.cat([x.float(), torch.zeros(len(x), 2)], 1)
        for i in range(ops.shape[1]):
            a = self.read(slots, addr[:, i, 0])
            b = self.read(slots, addr[:, i, 1])
            r = self.execute(ops[:, i], a, b)
            slots = torch.cat([slots[:, :P1], r[:, None], slots[:, P1:P2]], 1)
        return r


class SharedExecutor(Looped):
    """B: one learned MLP executes every op."""

    def __init__(self, seed, emb=16, hid=32):
        super().__init__(seed)
        torch.manual_seed(seed)
        self.op_emb = nn.Embedding(N_OPS, emb)
        self.mlp = nn.Sequential(nn.Linear(emb + 2, hid), nn.Tanh(), nn.Linear(hid, 1))

    def execute(self, op, a, b):
        h = torch.cat([self.op_emb(op), a[:, None], b[:, None]], -1)
        return torch.sigmoid(self.mlp(h)).squeeze(-1)


class Routed(Looped):
    """C / C0: learned soft routing over a frozen bank of tiny models."""

    def __init__(self, seed, bank):
        super().__init__(seed)
        g = torch.Generator().manual_seed(seed + 1)
        self.bank = bank
        self.router = nn.Parameter(torch.randn(N_OPS, N_GATES, generator=g) * 0.01)

    def execute(self, op, a, b):
        w = F.softmax(self.router[op], -1)                            # (B,K)
        return (w * self.bank(a, b)).sum(-1)


# ============================================================================
# Monolithic transformer (A)
# ============================================================================
# vocab: CLS, BIT0, BIT1, OP0..OP9, AD0..AD5
CLS, BIT, OP = 0, 1, 3
AD = OP + N_OPS
VOCAB = AD + N_SLOTS


def tokenize(x, ops, addr):
    """-> (B, 1 + 4 + 3D) tokens: CLS, the 4 bits, then (op, a, b) per step."""
    steps = torch.stack([OP + ops, AD + addr[..., 0], AD + addr[..., 1]], -1)
    return torch.cat([torch.full((len(x), 1), CLS), BIT + x, steps.flatten(1)], 1)


def sinusoid(T, C):
    pos = torch.arange(T)[:, None]
    div = torch.exp(torch.arange(0, C, 2) * (-math.log(10000.0) / C))
    pe = torch.zeros(T, C)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Monolithic(nn.Module):
    def __init__(self, seed, C=64, layers=2, heads=4):
        super().__init__()
        torch.manual_seed(seed)
        self.C = C
        self.emb = nn.Embedding(VOCAB, C)
        layer = nn.TransformerEncoderLayer(C, heads, 2 * C, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(C)
        self.head = nn.Linear(C, 1)

    def forward(self, x, ops, addr):
        t = tokenize(x, ops, addr)
        h = self.emb(t) + sinusoid(t.shape[1], self.C)
        h = self.ln(self.enc(h))
        return torch.sigmoid(self.head(h[:, 0])).squeeze(-1)


def build(name, seed, bank=None):
    if name == "A":
        return Monolithic(seed)
    if name == "B":
        return SharedExecutor(seed)
    if name in ("C", "C0"):
        return Routed(seed, bank)
    raise ValueError(name)
