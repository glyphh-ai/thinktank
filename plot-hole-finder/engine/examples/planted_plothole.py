"""Demo: localize a planted plot hole with the glyphh cortex.

A tiny three-chapter "novel" with two planted inconsistencies:

  1. Continuity error -- Elena's eye colour is established as GREEN in ch.1 and
     ch.5, then silently becomes BLUE in ch.12 with no licensing event.
  2. Coreference -- a stray "she" mention in ch.12 must be resolved back to
     Elena across the whole work.

The point: cheap layer-level probing flags the *contested* attribute without
any pairwise scan; we then drill into the segments to localize it; and the
cortex routes the stray mention to the right entity. No LLM, no prose -- just
VSA algebra over the extracted facts.

Run:  python -m examples.planted_plothole   (from the engine/ directory)
"""

from __future__ import annotations

import glyphh as g

DIM = 10_000


def build_cortex() -> tuple[g.Cortex, g.Codebook, g.CleanupMemory]:
    cb = g.Codebook(DIM, seed=7)
    eye_values = g.value_cleanup(cb, ["GREEN", "BLUE", "BROWN"])

    # --- Elena's timeline (entity layer) ---
    elena = g.Layer("Elena", DIM)
    elena.add(
        g.Segment("ch1:elena", DIM, narrative_time=1)
        .add(g.make_role(cb, "EYE_COLOR", "GREEN"))
        .add(g.make_role(cb, "HAIR", "RED"))
        .add(g.make_role(cb, "LOCATION", "VILLAGE"))
    )
    elena.add(
        g.Segment("ch5:elena", DIM, narrative_time=5)
        .add(g.make_role(cb, "EYE_COLOR", "GREEN"))
        .add(g.make_role(cb, "LOCATION", "CITY"))
    )
    elena.add(
        g.Segment("ch12:elena", DIM, narrative_time=12)
        .add(g.make_role(cb, "EYE_COLOR", "BLUE"))  # <-- planted plot hole
        .add(g.make_role(cb, "LOCATION", "CASTLE"))
    )

    # --- A second entity, so routing has to discriminate ---
    marcus = g.Layer("Marcus", DIM)
    marcus.add(
        g.Segment("ch2:marcus", DIM, narrative_time=2)
        .add(g.make_role(cb, "EYE_COLOR", "BROWN"))
        .add(g.make_role(cb, "HAIR", "BLACK"))
        .add(g.make_role(cb, "LOCATION", "VILLAGE"))
    )

    cortex = g.Cortex("the-work", DIM).add(elena).add(marcus)
    return cortex, cb, eye_values


def main() -> None:
    cortex, cb, eye_values = build_cortex()

    print("== glyphh: planted plot-hole demo ==\n")

    # 1) Cheap triage: probe each entity layer for a contested attribute.
    print("[1] Layer-level triage for contested EYE_COLOR (no pairwise scan):")
    for name, layer in cortex.layers.items():
        hits = layer.contested(cb, "EYE_COLOR", eye_values)
        values = ", ".join(f"{m.label}({m.similarity:.2f})" for m in hits)
        flag = "  <-- CONTESTED" if len(hits) > 1 else ""
        print(f"    {name:8s}: {values}{flag}")

    # 2) Localize: drill into the contested layer's segments.
    print("\n[2] Localizing the contradiction in Elena's segments:")
    groups = cortex.layers["Elena"].localize(cb, "EYE_COLOR", eye_values)
    for value, segs in groups.items():
        where = ", ".join(s.name for s in segs)
        print(f"    EYE_COLOR={value:6s} in {where}")
    if len(groups) > 1:
        print("    => continuity error: no licensing event between these states.")

    # 3) Coreference: route a stray "she" mention back to an entity.
    print("\n[3] Coreference routing for a stray mention "
          "(eyes=BLUE, location=CASTLE):")
    mention = (
        g.Segment("ch12:she", DIM, narrative_time=12)
        .add(g.make_role(cb, "EYE_COLOR", "BLUE"))
        .add(g.make_role(cb, "LOCATION", "CASTLE"))
    )
    routed = cortex.route(mention.vector)
    print(f"    routed to: {routed.label} (sim={routed.similarity:.2f}, "
          f"margin={routed.margin:.2f})")


if __name__ == "__main__":
    main()
