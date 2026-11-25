"""sample_frontier.py — 12 unseen-domain cases per move, prompts apart from gold.

Run: python sample_frontier.py OUT_DIR
"""

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1]
rows = json.load(open(f"{HERE}/data/test_unseen.json"))
rng = random.Random(7)
pick = []
for k in ("apply", "distinguish", "reverse", "overrule"):
    pick += rng.sample([s for s in rows if s["kind"] == k], 12)
rng.shuffle(pick)
json.dump([dict(i=i, decision=s["decision"], kind=s["kind"]) for i, s in enumerate(pick)],
          open(f"{out}/frontier_gold.json", "w"))
with open(f"{out}/frontier_prompts.txt", "w") as f:
    f.write("\n\n".join(f"=== CASE {i} ===\n{s['prompt']}" for i, s in enumerate(pick)))
print(len(pick))
