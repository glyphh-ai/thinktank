"""
hard.py — a genuinely hard grounding task.

Harder than ground.py in three ways:
  1. HANDOFFS: an object is taken, then GIVEN to someone, who then moves --
     answering needs a 3+ fact chain (take -> give -> go), not 2.
  2. DISTRACTORS: irrelevant actors/objects move around; the system must
     ignore them and follow only the chain that touches the queried object.
  3. HARDER EXTRACTION: "give X to Y" is a 3-slot relation (actor, object,
     recipient) -- agents must get the recipient right or the chain breaks.

Same pipeline: real agents extract facts -> lattice; deterministic composition
replays world-state -> answer. Accuracy is expected to DROP below 100%; the
drop, and where it comes from, is the honest measure of difficulty.

Run:  python hard.py                       # reference ceiling
      python hard.py --agent-facts F.json  # measure on real-agent facts
"""

from __future__ import annotations
import json
import random
import sys
from store import Store
import ground

ACTORS = ["mary", "john", "sandra", "daniel", "fred", "bill"]
PLACES = ["kitchen", "garden", "office", "bedroom", "hallway", "bathroom", "cellar"]
OBJECTS = ["milk", "apple", "football", "book", "key", "box"]

GO = ["went to the", "journeyed to the", "moved to the", "travelled to the"]
TAKE = ["picked up the", "grabbed the", "took the", "got the"]
GIVE = ["gave the", "handed the", "passed the"]   # ... "to <recipient>"


def simulate(facts):
    """Replay facts (with handoffs) -> object locations. Gold AND composer."""
    actor_loc, holder, obj_loc = {}, {}, {}
    for f in facts:
        a, act, arg = f["actor"], f["action"], f["arg"]
        if act == "go":
            actor_loc[a] = arg
            for o, h in holder.items():
                if h == a:
                    obj_loc[o] = arg
        elif act == "take":
            holder[arg] = a
            obj_loc[arg] = actor_loc.get(a)
        elif act == "give":
            holder[arg] = f.get("to")
            obj_loc[arg] = actor_loc.get(a)
    return obj_loc


def generate(n_stories: int, seed: int = 11):
    rng = random.Random(seed)
    stories = []
    for _ in range(n_stories):
        a1, a2 = rng.sample(ACTORS, 2)
        obj = rng.choice(OBJECTS)
        facts, sentences = [], []

        def go(actor):
            p = rng.choice(PLACES)
            facts.append({"actor": actor, "action": "go", "arg": p})
            sentences.append(f"{actor.title()} {rng.choice(GO)} {p}.")

        def distractor():
            actor = rng.choice(ACTORS)
            if rng.random() < 0.5:
                go(actor)
            else:
                o = rng.choice([x for x in OBJECTS if x != obj])
                facts.append({"actor": actor, "action": "take", "arg": o})
                sentences.append(f"{actor.title()} {rng.choice(TAKE)} {o}.")

        # the real chain, interleaved with distractors
        go(a1); distractor()
        facts.append({"actor": a1, "action": "take", "arg": obj})
        sentences.append(f"{a1.title()} {rng.choice(TAKE)} {obj}.")
        distractor()
        facts.append({"actor": a1, "action": "give", "arg": obj, "to": a2})
        sentences.append(f"{a1.title()} {rng.choice(GIVE)} {obj} to {a2.title()}.")
        distractor()
        go(a2)                                  # a2 carries the object away
        distractor()

        gold = simulate(facts).get(obj)
        if gold:
            stories.append({"sentences": sentences, "object": obj, "answer": gold})
    return stories


def reference_decompose(sentence: str):
    s = sentence.lower().rstrip(".")
    w = s.split()
    actor, target = w[0], w[-1]
    if any(v.split()[0] in s for v in GIVE) and " to " in s:
        # "<actor> gave the <obj> to <recipient>"
        return {"actor": actor, "action": "give", "arg": w[-3], "to": w[-1]}
    if any(v.split()[0] in s for v in GO) and "to the" in s:
        return {"actor": actor, "action": "go", "arg": target}
    if any(v.split()[0] in s for v in TAKE):
        return {"actor": actor, "action": "take", "arg": target}
    return None


def measure(stories, decode):
    store = Store(":memory:")
    correct = total = no_parse = 0
    misses = []
    for i, st in enumerate(stories):
        facts = decode(i, st)
        if facts is None:
            no_parse += 1
            continue
        ground.ground_story(store, i, facts)
        ans = simulate(facts).get(st["object"])
        total += 1
        if ans == st["answer"]:
            correct += 1
        elif len(misses) < 8:
            misses.append(f"story {i} [{st['object']}]: said {ans}, gold {st['answer']}")
    return correct, total, no_parse, misses, store


def main():
    stories = generate(200)
    print("=" * 68)
    print("TRULY HARD — handoffs (3+ hops) + distractors")
    print("=" * 68)
    print(f"  corpus: {len(stories)} stories, "
          f"{sum(len(s['sentences']) for s in stories)} sentences")
    print(f"  each answer needs take -> give -> go, through distractors\n")

    if len(sys.argv) > 2 and sys.argv[1] == "--agent-facts":
        af = json.load(open(sys.argv[2]))
        decode = lambda i, st: af.get(str(i))
        label = "REAL AGENTS"
    else:
        def decode(i, st):
            out = [reference_decompose(x) for x in st["sentences"]]
            return out if all(out) else None
        label = "reference parser"

    correct, total, no_parse, misses, store = measure(stories, decode)
    acc = 100 * correct / total if total else 0
    print(f"  decomposer: {label}")
    print(f"  understanding (composition accuracy): {correct}/{total} = {acc:.1f}%")
    if no_parse:
        print(f"  unparsed stories skipped: {no_parse}")
    for m in misses:
        print(f"     miss: {m}")


if __name__ == "__main__":
    main()
