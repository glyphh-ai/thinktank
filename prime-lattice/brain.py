"""
brain.py — the controller. It inherits capabilities from the systems in the
lattice and calls them through a typed interface.

The brain implements nothing itself. Each tower-system advertises a typed
CAPABILITY in the lattice -- a signature (consumes -> produces) over a system.
The controller reads those signatures, inherits the whole repertoire, and
AUTO-COMPOSES a pipeline by matching types (produces -> consumes). No control
flow is hardcoded: ask it to get from `text` to `affect` and it discovers
read -> comprehend -> appraise -> feel on its own.

Add a new system + capability node and the brain inherits it for free -- the
same "rules as data" move, now for capabilities. The controller is itself a
system in the lattice (substrate = the capability nodes).

Run:  python brain.py
"""

from __future__ import annotations
import json
from collections import deque
from store import Store, Node, node_id
import compose as Cmp
import reader as R
import emotion as E


# ---------------------------------------------------------------------------
# Register a typed capability INTO THE LATTICE (a typed interface over a system)
# ---------------------------------------------------------------------------
def register(store, name, consumes, produces, system_id):
    store.put_many([Node("program", "capability", name, type="capability",
                        substrate=[system_id], agent="a:brain",
                        pathways={"consumes": consumes, "produces": produces})])


class Brain:
    """The controller: inherits capabilities from the lattice, dispatches by type."""

    def __init__(self, store: Store, fns: dict):
        self.store, self.fns = store, fns
        self.caps = self._inherit()

    def _inherit(self) -> dict:
        """Read every capability the lattice advertises -- the brain's repertoire."""
        caps = {}
        for name, pathways in self.store.db.execute(
                "SELECT value, pathways FROM nodes WHERE kind='capability'"):
            p = json.loads(pathways)
            caps[name] = (p["consumes"], p["produces"])
        return caps

    def plan(self, src_type: str, goal_type: str) -> list[str] | None:
        """Auto-compose a pipeline by chaining produces -> consumes (typed)."""
        q, seen = deque([(src_type, [])]), {src_type}
        while q:
            t, chain = q.popleft()
            if t == goal_type:
                return chain
            for name, (consumes, produces) in self.caps.items():
                if consumes == t and produces not in seen:
                    seen.add(produces)
                    q.append((produces, chain + [name]))
        return None

    def invoke(self, src_type: str, goal_type: str, value):
        chain = self.plan(src_type, goal_type)
        if chain is None:
            return None, None
        trace = src_type
        for name in chain:
            value = self.fns[name](value)
            trace += f" -({name})-> {self.caps[name][1]}"
        return value, trace


def build_brain():
    store, reader, comp = Cmp.build_world()

    sys_gram = node_id("system", "grammar", "english")
    sys_comp = node_id("system", "comprehension", "core")
    sys_emo = node_id("system", "emotion", "core")

    # the executable core (fixed); the typed signatures live in the lattice
    fns = {
        "read":       lambda text: reader.parse(text),
        "comprehend": lambda tree: comp.comprehend(tree) if tree else None,
        "appraise":   lambda m: Cmp.appraise(m),
        "feel":       lambda a: E.recognize(a) if a and a["valence"] != "neutral" else "calm",
    }
    register(store, "read",       "text",      "structure", sys_gram)
    register(store, "comprehend", "structure", "meaning",   sys_comp)
    register(store, "appraise",   "meaning",   "appraisal", sys_comp)
    register(store, "feel",       "appraisal", "affect",    sys_emo)

    # the controller is itself a system in the lattice
    cap_ids = [r[0] for r in store.db.execute(
        "SELECT id FROM nodes WHERE kind='capability'")]
    store.put_many([Node("system", "brain", "executive", type="controller",
                         substrate=cap_ids, agent="a:brain")])
    return store, fns


def main():
    store, fns = build_brain()
    brain = Brain(store, fns)

    print("=" * 68)
    print("THE BRAIN — inherits typed capabilities, auto-composes by type")
    print("=" * 68)
    print(f"  inherited capabilities: "
          + ", ".join(f"{n}({c}->{p})" for n, (c, p) in brain.caps.items()))

    print("\n  ask: text -> affect   (no pipeline hardcoded; brain plans it)")
    result, trace = brain.invoke("text", "affect", "the dog bit the cat")
    print(f"     plan : {trace}")
    print(f"     'the dog bit the cat' -> feels: {result}")

    print("\n" + "-" * 68)
    print("  INHERIT A NEW CAPABILITY — add a system, brain uses it, zero code")
    print("  adding capability:  narrate (affect -> report)")
    register(store, "narrate", "affect", "report",
             node_id("system", "emotion", "core"))
    fns["narrate"] = lambda affect: f"\"the subject feels {affect}.\""
    brain = Brain(store, fns)                       # re-inherit from the lattice

    result, trace = brain.invoke("text", "report", "the cat won")
    print(f"     plan : {trace}")
    print(f"     'the cat won' -> {result}")
    print("\n  => the controller implements nothing; it inherits capabilities")
    print("     from the lattice and calls them through one typed interface.")
    print("     New system -> new capability -> new behaviour, no code change.")


if __name__ == "__main__":
    main()
