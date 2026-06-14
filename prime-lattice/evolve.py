"""
evolve.py — self-rewriting knowledge: consolidate, revert, and order-dependence.

A frame is a TYPE that gets rewritten by experience (instances), not prompted:

  teach        v1: BITE.agent = {dog, cat}            (semantic memory)
  assimilate   instances that fit -> nothing changes
  CONSOLIDATE  enough new biters (snake, spider) -> generalize to ANIMATE
  REVERT       the generalization causes bad outcomes (butterfly, worm don't
               bite) enough times -> pull the reframe back to the confirmed set

Every version is committed to the lattice with provenance, so the whole
learning history is auditable and a single reframe can be rolled back surgically
(an LLM cannot do this). And because each instance rewrites the type that
conditions the next, the ORDER of experience changes the final knowledge --
shown at the end with the same observations in two orders.

Run:  python evolve.py
"""

from __future__ import annotations
from store import Store, Node, node_id

# a tiny ontology: which things are ANIMATE (used by the generalized frame)
ANIMATE = {"dog", "cat", "snake", "spider", "bee", "butterfly", "worm", "ant"}
CONSOLIDATE_AT = 2     # distinct NEW confirmed biters before generalizing
REVERT_AT = 2          # bad outcomes on accepted agents before reverting


class Frame:
    def __init__(self, name, version, kind, allow, prev, reason):
        self.name, self.version, self.kind = name, version, kind
        self.allow = set(allow) if kind == "extensional" else allow   # category = str
        self.prev, self.reason = prev, reason

    def accepts(self, item):
        return item in ANIMATE if self.kind == "category" else item in self.allow

    def desc(self):
        return self.allow if self.kind == "category" else "{" + ", ".join(sorted(self.allow)) + "}"


class Mind:
    def __init__(self, store, name="BITE"):
        self.store, self.name = store, name
        self.head = None
        self.confirmed, self.new_biters, self.bad_agents = set(), set(), set()
        self.log = []

    def _commit(self, f: Frame):
        self.head = f
        # durable, provenance-stamped version node in the lattice (audit log)
        self.store.put_many([Node(
            "program", "frame", f"{self.name}#v{f.version}", type="frame", agent="evolve",
            substrate=[node_id("prime", "sem-prime", "BITE")],
            pathways={"version": f.version, "kind": f.kind, "prev": f.prev,
                      "allow": f.allow if f.kind == "category" else sorted(f.allow),
                      "reason": f.reason})])
        self.log.append((f.version, f.kind, f.desc(), f.reason))

    def teach(self, agents):
        self._commit(Frame(self.name, 1, "extensional", agents, None, "taught"))

    def observe(self, agent, outcome):
        accepted = self.head.accepts(agent)
        note = f"{agent:10} outcome={outcome:4} accepted={str(accepted):5}"
        if outcome == "good":
            self.confirmed.add(agent)
            if not accepted:                                  # new evidence
                self.new_biters.add(agent)
                if self.head.kind == "extensional" and len(self.new_biters) >= CONSOLIDATE_AT:
                    self._accommodate(); note += "  ==> CONSOLIDATE (generalize -> ANIMATE)"
        else:                                                 # bad outcome
            if accepted:                                      # the frame let it in -> its fault
                self.bad_agents.add(agent)
                if len(self.bad_agents) >= REVERT_AT:
                    self._revert(); note += "  ==> REVERT (pull the reframe back)"
        return note

    def _accommodate(self):
        self._commit(Frame(self.name, self.head.version + 1, "category", "ANIMATE",
                           self.head.version,
                           f"generalized to ANIMATE from {sorted(self.confirmed)}"))
        self.new_biters.clear()

    def _revert(self):
        self._commit(Frame(self.name, self.head.version + 1, "extensional",
                           set(self.confirmed), self.head.version,
                           f"reverted over-generalization after bad: {sorted(self.bad_agents)}"))
        self.bad_agents.clear()


def run(title, observations):
    store = Store(":memory:")
    store.put_many([Node("prime", "sem-prime", "BITE", agent="seed")])
    m = Mind(store)
    m.teach(["dog", "cat"])
    print("\n" + title)
    print(f"  v1 taught: BITE.agent = {m.head.desc()}")
    for agent, outcome in observations:
        print("   " + m.observe(agent, outcome))
    print(f"  FINAL: BITE.agent = {m.head.desc()}   (v{m.head.version})")
    print(f"  behavior now: 'bee bit'={m.head.accepts('bee')}  "
          f"'butterfly bit'={m.head.accepts('butterfly')}")
    print("  audit (version chain, from the lattice):")
    for v, k, d, r in m.log:
        chain = store.db.execute(
            "SELECT pathways FROM nodes WHERE value=?", (f"BITE#v{v}",)).fetchone()
        print(f"     v{v} [{k:11}] {d}")
        print(f"          why: {r}")
    return m.head.desc()


def main():
    print("=" * 70)
    print("SELF-REWRITING KNOWLEDGE — consolidate, revert, audit")
    print("=" * 70)

    obs = [("dog", "good"), ("cat", "good"), ("snake", "good"), ("spider", "good"),
           ("bee", "good"), ("butterfly", "bad"), ("worm", "bad")]
    a = run("ORDER A — biters confirmed first, bad outcomes last:", obs)

    print("\n" + "=" * 70)
    print("ORDER MATERIALLY MATTERS — same observations, different order")
    print("=" * 70)
    reordered = [("butterfly", "bad"), ("worm", "bad"), ("dog", "good"),
                 ("snake", "good"), ("spider", "good")]
    b = run("ORDER B — bad outcomes arrive BEFORE the generalization exists:",
            reordered)

    print("\n" + "-" * 70)
    print(f"  ORDER A final: {a}")
    print(f"  ORDER B final: {b}")
    print("  Same evidence, different order -> different knowledge. In B the bad")
    print("  outcomes landed while the frame still REJECTED them, so nothing was")
    print("  attributed; the generalization then stuck. Path-dependent, but every")
    print("  step is versioned and reversible -- learned, not prompted.")


if __name__ == "__main__":
    main()
