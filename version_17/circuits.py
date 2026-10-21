"""
Circuits — the ground truth
===========================
The 16 two-input boolean functions, and random straight-line circuits
built from them. Every circuit has an exact answer, so every claim about
a model is checkable to the bit.

A circuit: 4 input bits x0..x3, then D steps. Step i computes
    r_i = op_i(slot[a_i], slot[b_i])
over 6 slots: X0..X3 (the inputs), P1 (previous result), P2 (result
before that). One operand of every step after the first is P1, so the
answer really sits D gates deep. Ops are drawn from the 10 gates that
depend on BOTH inputs (no constants, no projections) — a constant or a
projection would cut the chain and fake the depth.
"""

import numpy as np

# Gate k's truth table: bit (2a + b) of k is gate_k(a, b).
N_GATES = 16
TABLE = np.array([[(k >> (2 * a + b)) & 1 for a in (0, 1) for b in (0, 1)]
                  for k in range(N_GATES)])

def gate(k, a, b):
    return TABLE[k, 2 * a + b]

# The 10 gates whose output depends on both inputs.
DEEP_GATES = [k for k in range(N_GATES)
              if any(gate(k, 0, b) != gate(k, 1, b) for b in (0, 1))
              and any(gate(k, a, 0) != gate(k, a, 1) for a in (0, 1))]
N_OPS = len(DEEP_GATES)  # circuit op token i means gate DEEP_GATES[i]

N_IN = 4
X0, P1, P2 = 0, 4, 5
N_SLOTS = 6


def random_circuit(depth, rng):
    """Returns (x, ops, addr) — x (4,), ops (D,), addr (D, 2)."""
    x = rng.integers(0, 2, N_IN)
    ops = rng.integers(0, N_OPS, depth)
    addr = np.zeros((depth, 2), dtype=np.int64)
    for i in range(depth):
        if i == 0:
            addr[i] = rng.choice(N_IN, 2, replace=False)
        else:
            pool = list(range(N_IN)) + ([P2] if i >= 2 else [])
            pair = [P1, rng.choice(pool)]
            if rng.random() < 0.5:
                pair.reverse()
            addr[i] = pair
    return x, ops, addr


def run(x, ops, addr):
    """Exact evaluation. Returns the final bit."""
    slots = list(x) + [0, 0]
    for op, (a, b) in zip(ops, addr):
        r = gate(DEEP_GATES[op], slots[a], slots[b])
        slots[P2], slots[P1] = slots[P1], r
    return slots[P1]


def shortcut(x, ops, addr):
    """Best guess from the LAST step alone, not executing anything before
    it: if the fresh operand forces the gate's output (AND with 0, OR with
    1, ...) answer that, else guess 1. ~0.70 at any depth — the floor a
    model can reach without executing the circuit."""
    op, (a, b) = ops[-1], addr[-1]
    if len(ops) == 1:
        return run(x, ops, addr)
    k = DEEP_GATES[op]
    if a == P1:
        outs = {gate(k, p, x[b] if b < N_IN else 0) for p in (0, 1)}
    else:
        outs = {gate(k, x[a] if a < N_IN else 0, p) for p in (0, 1)}
    if b == P2 or a == P2:  # fresh operand is P2 — also unknown
        return 1
    return outs.pop() if len(outs) == 1 else 1


def make_set(n, depth, rng):
    """n circuits of one depth, as stacked arrays + exact answers."""
    cs = [random_circuit(depth, rng) for _ in range(n)]
    x = np.stack([c[0] for c in cs])
    ops = np.stack([c[1] for c in cs])
    addr = np.stack([c[2] for c in cs])
    y = np.array([run(*c) for c in cs])
    return x, ops, addr, y


if __name__ == "__main__":
    names = {1: "NOR", 6: "XOR", 7: "NAND", 8: "AND", 9: "XNOR", 14: "OR"}
    print("deep gates:", [(k, names.get(k, f"g{k}")) for k in DEEP_GATES])
    rng = np.random.default_rng(0)
    for d in (1, 4, 16):
        x, ops, addr, y = make_set(4000, d, rng)
        sc = np.mean([shortcut(*c) == t for *c, t in zip(x, ops, addr, y)])
        print(f"depth {d:2d}: P(y=1) = {y.mean():.3f} | last-step shortcut acc = {sc:.3f}")
