"""
ground.py — grounded understanding, measured.

The experiment: take a corpus, break every sentence into the lattice as FACTS,
then measure understanding by what can be COMPOSED from those facts — answering
questions that require chaining 2+ facts (so we measure understanding, not
memory).

  corpus sentence   --(real agents)-->   FACT in the lattice
  question          --(deterministic)-->  compose facts -> answer
  understanding     =                      answer accuracy

The corpus is a controlled world (people move between places, carry objects).
"Where is the X?" needs two supporting facts: who took X, and where they then
went. The agents never see the answer; they only extract facts. Composition
reconstructs world-state from the facts and answers. No model at answer time.

Roles:
  * REAL AGENTS do the open-ended work: natural language -> structured facts.
  * The LATTICE stores facts (entities=primes, facts=programs, story=system).
  * COMPOSITION (deterministic) answers by replaying facts -> world state.

Run:  python ground.py                 # generate + reference-decompose + measure
      python ground.py --agent-facts F # measure using agent-produced facts (JSON)
"""

from __future__ import annotations
import json
import random
import sys
from store import Store, Node, node_id

ACTORS = ["mary", "john", "sandra", "daniel"]
PLACES = ["kitchen", "garden", "office", "bedroom", "hallway", "bathroom"]
OBJECTS = ["milk", "apple", "football", "book"]

GO = ["went to the", "journeyed to the", "moved to the", "travelled to the"]
TAKE = ["picked up the", "grabbed the", "took the", "got the"]
DROP = ["dropped the", "left the", "put down the", "discarded the"]


# ===========================================================================
# WORLD + CORPUS — a controlled generator with hidden ground truth.
# ===========================================================================
def simulate(events):
    """Replay (actor, action, arg) events -> final object locations.
    This is BOTH the gold oracle and the composition engine."""
    actor_loc, holder, obj_loc = {}, {}, {}
    for actor, action, arg in events:
        if action == "go":
            actor_loc[actor] = arg
            for o, h in holder.items():
                if h == actor:
                    obj_loc[o] = arg
        elif action == "take":
            holder[arg] = actor
            obj_loc[arg] = actor_loc.get(actor)
        elif action == "drop":
            holder[arg] = None
            obj_loc[arg] = actor_loc.get(actor)
    return obj_loc


def generate(n_stories: int, seed: int = 7):
    rng = random.Random(seed)
    stories = []
    for _ in range(n_stories):
        events, sentences = [], []
        obj = rng.choice(OBJECTS)
        steps = rng.randint(4, 7)
        took = False
        for i in range(steps):
            actor = rng.choice(ACTORS)
            r = rng.random()
            if not took or r < 0.5:                       # go
                place = rng.choice(PLACES)
                events.append((actor, "go", place))
                sentences.append(f"{actor.title()} {rng.choice(GO)} {place}.")
                if not took and i == steps - 2:           # ensure obj is taken late
                    events.append((actor, "take", obj))
                    sentences.append(f"{actor.title()} {rng.choice(TAKE)} {obj}.")
                    took = True
            else:                                         # take/drop
                events.append((actor, "take", obj))
                sentences.append(f"{actor.title()} {rng.choice(TAKE)} {obj}.")
                took = True
        gold = simulate(events).get(obj)
        if gold:
            stories.append({"sentences": sentences,
                            "question": f"Where is the {obj}?",
                            "object": obj, "answer": gold})
    return stories


# ===========================================================================
# DECOMPOSITION — sentence -> fact. (Reference parser; agents do the real one.)
# ===========================================================================
def reference_decompose(sentence: str) -> dict | None:
    s = sentence.lower().rstrip(".")
    words = s.split()
    actor = words[0]
    target = words[-1]
    if any(v.split()[0] in s for v in GO) and "to the" in s:
        return {"actor": actor, "action": "go", "arg": target}
    if any(v.split()[0] in s for v in TAKE):
        return {"actor": actor, "action": "take", "arg": target}
    if any(v.split()[0] in s for v in DROP):
        return {"actor": actor, "action": "drop", "arg": target}
    return None


# ===========================================================================
# GROUND INTO THE LATTICE + COMPOSE AN ANSWER.
# ===========================================================================
def ground_story(store: Store, idx: int, facts: list[dict]):
    ents = {f["actor"] for f in facts} | {f["arg"] for f in facts}
    store.put_many([Node("prime", "entity", e, agent="a:ground") for e in ents])
    fact_nodes = []
    for j, f in enumerate(facts):
        fact_nodes.append(Node("program", "fact", f"s{idx}:{j}:{f['actor']}:{f['action']}:{f['arg']}",
                               type=f["action"], agent="a:agent",
                               substrate=[node_id("prime", "entity", f["actor"]),
                                          node_id("prime", "entity", f["arg"])],
                               pathways={"fact": f, "ord": j}))
    store.put_many(fact_nodes)
    store.put_many([Node("system", "story", f"story-{idx}", type="episode",
                         substrate=[n.id for n in fact_nodes] or
                                   [node_id("prime", "entity", next(iter(ents)))],
                         agent="a:ground")])


def compose_answer(facts: list[dict], obj: str) -> str | None:
    events = [(f["actor"], f["action"], f["arg"]) for f in facts]
    return simulate(events).get(obj)


# ===========================================================================
# MEASURE understanding = composition accuracy over the corpus.
# ===========================================================================
def measure(stories, decode_facts):
    store = Store(":memory:")
    correct = total = no_parse = 0
    for i, st in enumerate(stories):
        facts = decode_facts(i, st)
        if facts is None:
            no_parse += 1
            continue
        ground_story(store, i, facts)
        ans = compose_answer(facts, st["object"])
        total += 1
        correct += (ans == st["answer"])
    return correct, total, no_parse, store


def main():
    stories = generate(200)
    print("=" * 68)
    print("GROUNDED UNDERSTANDING — measured by fact composition")
    print("=" * 68)
    print(f"  corpus: {len(stories)} stories, "
          f"{sum(len(s['sentences']) for s in stories)} sentences, "
          f"each Q needs 2+ composed facts\n")

    # Mode 1: reference decomposer (reproducible baseline / self-test)
    def ref(i, st):
        out = [reference_decompose(x) for x in st["sentences"]]
        return out if all(out) else None

    if len(sys.argv) > 2 and sys.argv[1] == "--agent-facts":
        agent_facts = json.load(open(sys.argv[2]))   # {story_idx: [facts]}
        def decode(i, st): return agent_facts.get(str(i))
        label = "REAL AGENTS"
    else:
        decode, label = ref, "reference parser"

    correct, total, no_parse, store = measure(stories, decode)
    n_facts = store.db.execute(
        "SELECT count(*) FROM nodes WHERE kind='fact'").fetchone()[0]
    print(f"  decomposer    : {label}")
    print(f"  facts grounded into lattice: {n_facts} fact-nodes")
    acc = 100 * correct / total if total else 0
    print(f"  questions answered by composition: {correct}/{total}  "
          f"= {acc:.1f}% understanding")
    if no_parse:
        print(f"  (skipped {no_parse} stories with unparsed sentences)")
    print("\n  understanding here = reconstructing world-state from grounded")
    print("  facts and composing the answer. No model runs at answer time;")
    print("  the agents only did build-time extraction (NL -> facts).")


if __name__ == "__main__":
    main()
