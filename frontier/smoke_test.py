"""
smoke_test.py — validate the dual-objective pipeline on CPU with a tiny model.

Same logic the Qwen3 fine-tune uses (LM next-token loss + Ada-role token loss,
masked, on the prepared data), but a ~tiny GRU so it runs in seconds on CPU.
This proves the data->batches->dual-loss->step->eval path is correct before any
GPU spend. Run as CI smoke too.

Run:  python smoke_test.py
"""

import json, torch, torch.nn as nn
from prepare_data import ALL_ROLES

ROLES = ["O"] + ALL_ROLES
RID = {r: i for i, r in enumerate(ROLES)}
PAD = 0


def load(path="data/facts.jsonl", limit=2000):
    facts = [json.loads(l) for i, l in enumerate(open(path)) if i < limit]
    vocab = {"<pad>": 0, "<unk>": 1}
    for f in facts:
        for w, _ in f["word_roles"]:
            vocab.setdefault(w, len(vocab))
    return facts, vocab


def batchify(facts, vocab, T=12):
    X, R = [], []
    for f in facts:
        ids = [vocab.get(w, 1) for w, _ in f["word_roles"]][:T]
        rs = [RID[r] for _, r in f["word_roles"]][:T]
        ids += [PAD] * (T - len(ids)); rs += [-100] * (T - len(rs))  # -100 = ignore
        X.append(ids); R.append(rs)
    return torch.tensor(X), torch.tensor(R)


class Tiny(nn.Module):
    def __init__(self, V, d=64, n_roles=len(ROLES)):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=PAD)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.lm = nn.Linear(d, V)
        self.role = nn.Linear(d, n_roles)

    def forward(self, x):
        h, _ = self.gru(self.emb(x))
        return self.lm(h), self.role(h)


def main():
    facts, vocab = load()
    X, R = batchify(facts, vocab)
    n = int(0.9 * len(X)); Xtr, Rtr, Xva, Rva = X[:n], R[:n], X[n:], R[n:]
    model = Tiny(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    lossf_pad = nn.CrossEntropyLoss(ignore_index=PAD)

    def step(Xb, Rb, lam=1.0, train=True):
        lm, role = model(Xb)
        # LM: predict next token
        lm_loss = lossf_pad(lm[:, :-1].reshape(-1, lm.size(-1)), Xb[:, 1:].reshape(-1))
        # ROLE: tag current token (the Ada-in-the-loop objective)
        role_loss = lossf(role.reshape(-1, role.size(-1)), Rb.reshape(-1))
        loss = lm_loss + lam * role_loss
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        return lm_loss.item(), role_loss.item()

    print("step  lm_loss  role_loss")
    for s in range(200):
        bi = torch.randint(0, len(Xtr), (64,))
        l, r = step(Xtr[bi], Rtr[bi])
        if s % 50 == 0 or s == 199:
            print(f"{s:4d}  {l:7.3f}  {r:7.3f}")

    # eval role tagging accuracy on val
    with torch.no_grad():
        _, role = model(Xva)
        pred = role.argmax(-1)
        mask = Rva != -100
        acc = (pred[mask] == Rva[mask]).float().mean().item()
    print(f"\nval role-tag accuracy: {acc:.1%}")
    assert acc > 0.5, "pipeline produced no learning — smoke test FAILED"
    print("SMOKE TEST PASSED — dual-objective pipeline logic is correct.")


if __name__ == "__main__":
    main()
