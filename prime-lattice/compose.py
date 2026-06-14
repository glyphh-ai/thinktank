"""
compose.py — the composition SYSTEM: bind the towers into a situation.

Each tower answers part of a situation. Composition is the system that fuses
them into one bound representation -- producing capability no single tower has:

    reader      -> structure   (who did what)
    comprehend  -> meaning     (BITE(agent=dog, patient=cat))
    emotion     -> affect      (the cat feels ...)
    COMPOSE     -> situation    structure + meaning + affect + cause, bound

The situation is itself a node whose substrate spans all three tower-systems
(multi-tower inheritance). This is the first test of capability-from-stacking:
"what does X feel, and why?" needs comprehension AND emotion, composed.

Run:  python compose.py
"""

from __future__ import annotations
from store import Store, Node, node_id
import reader as R
import comprehend as C
import emotion as E

VERB_VALENCE = {"BITE": "bad", "WIN": "good", "SEE": "neutral",
                "SLEEP": "neutral", "SIT": "neutral", "LOSE": "bad"}


def add_verb(store: Store, word: str, name: str, roles: list[str]):
    store.put_many([
        Node("program", "word", word, type="verb", agent="a:lex",
             substrate=[node_id("prime", "category", "verb")]),
        Node("program", "lex-sem", word, type="meaning", agent="a:sem",
             substrate=[node_id("prime", "sem-prime", "KIND")],
             pathways={"meaning": {"t": "verb", "name": name, "roles": roles}}),
    ])


def build_world() -> tuple[Store, R.Reader, C.Comprehender]:
    store = Store(":memory:")
    R.seed_grammar(store)
    store.put_many([R._rule_node("NP", ["NP", "conj", "NP"])])
    # give the grammar a system node so other towers can compose onto it
    rule_ids = [r[0] for r in store.db.execute(
        "SELECT id FROM nodes WHERE kind='rule'")]
    store.put_many([Node("system", "grammar", "english", type="grammar",
                         substrate=rule_ids, agent="a:gram")])
    C.seed_comprehension(store)
    E.seed_emotion(store)
    add_verb(store, "bit", "BITE", ["agent", "patient"])
    add_verb(store, "won", "WIN", ["agent"])
    return store, R.Reader(store), C.Comprehender(store)


def appraise(meaning: dict) -> dict | None:
    """Bridge: a comprehended event -> an appraisal (the seam to emotion)."""
    if meaning.get("t") != "pred":
        return None
    valence = VERB_VALENCE.get(meaning["name"], "neutral")
    roles = meaning["roles"]
    experiencer = roles.get("patient") or roles.get("agent")
    a = {"valence": valence, "experiencer": experiencer}
    if "patient" in roles and "agent" in roles:
        a["cause"] = "other"
    if valence == "good":
        a["want"] = True
    return a


class Composer:
    """The composition system: run the towers and bind their outputs."""
    def __init__(self, store, reader, comp):
        self.store, self.reader, self.comp = store, reader, comp

    def situation(self, sentence: str) -> dict:
        tree = self.reader.parse(sentence)
        meaning = self.comp.comprehend(tree) if tree else None
        appraisal = appraise(meaning) if meaning else None
        feeling = (E.recognize(appraisal)
                   if appraisal and appraisal["valence"] != "neutral" else "calm")
        sit = {"text": sentence, "structure": tree, "meaning": meaning,
               "appraisal": appraisal, "feeling": feeling}
        self._persist(sit)
        return sit

    def _persist(self, sit):
        # a situation node whose substrate spans all three tower-systems
        self.store.put_many([Node(
            "program", "situation", sit["text"], type="situation", agent="a:compose",
            substrate=[node_id("system", "grammar", "english"),
                       node_id("system", "comprehension", "core"),
                       node_id("system", "emotion", "core")])])

    @staticmethod
    def answer(sit: dict) -> str:
        m = sit["meaning"]
        if not m or m.get("t") != "pred":
            return "(no situation)"
        who = sit["appraisal"]["experiencer"]["label"] if sit["appraisal"] else "?"
        roles = m["roles"]
        if "agent" in roles and "patient" in roles:
            why = f"{roles['agent']['label']} {m['name']} {roles['patient']['label']}"
        else:
            why = f"{roles.get('agent', {}).get('label','?')} {m['name']}"
        return f"{who} feels {sit['feeling'].upper()}  (because: {why})"


def main():
    store, reader, comp = build_world()
    composer = Composer(store, reader, comp)

    print("=" * 68)
    print("COMPOSITION SYSTEM — bind the towers into one situation")
    print("=" * 68)
    for s in ["the dog bit the cat", "the cat won", "the big cat saw a dog"]:
        sit = composer.situation(s)
        print(f"\n  \"{s}\"")
        print(f"     structure: {R.render(sit['structure'])}")
        print(f"     meaning  : {C.render(sit['meaning'])}")
        print(f"     affect   : {sit['feeling']}")
        print(f"     COMPOSED -> {Composer.answer(sit)}")

    print("\n" + "-" * 68)
    print("  CAPABILITY FROM STACKING — 'what does X feel and why?' needs")
    print("  comprehension AND emotion, bound by composition. No single tower")
    print("  answers it; the situation node spans all three tower-systems.")


if __name__ == "__main__":
    main()
