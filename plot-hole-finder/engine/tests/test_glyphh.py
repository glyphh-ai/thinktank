"""Tests for the glyphh hierarchical HDC engine."""

import numpy as np
import pytest

import glyphh as g

DIM = 10_000


@pytest.fixture
def cb() -> g.Codebook:
    return g.Codebook(DIM, seed=1)


def test_bind_is_invertible(cb):
    a, b = cb["A"], cb["B"]
    bound = g.bind(a, b)
    assert g.similarity(bound, a) < 0.1  # bind randomizes similarity
    assert np.array_equal(g.unbind(bound, b), a)  # ... but is invertible


def test_bundle_preserves_similarity(cb):
    parts = [cb[s] for s in "ABCD"]
    bundled = g.bundle(parts)
    for p in parts:
        assert g.similarity(bundled, p) > 0.2  # bundle stays close to its parts


def test_codebook_is_reproducible():
    a = g.Codebook(DIM, seed=42)["X"]
    b = g.Codebook(DIM, seed=42)["X"]
    assert np.array_equal(a, b)


def test_segment_probe_recovers_value(cb):
    values = g.value_cleanup(cb, ["GREEN", "BLUE", "BROWN"])
    seg = (
        g.Segment("s", DIM)
        .add(g.make_role(cb, "EYE_COLOR", "GREEN"))
        .add(g.make_role(cb, "LOCATION", "CITY"))
    )
    match = seg.probe(cb, "EYE_COLOR", values)
    assert match.label == "GREEN"
    assert match.margin > 0.1  # clean, confident read


def test_layer_flags_contested_attribute(cb):
    values = g.value_cleanup(cb, ["GREEN", "BLUE", "BROWN"])
    layer = g.Layer("Elena", DIM)
    layer.add(g.Segment("ch1", DIM).add(g.make_role(cb, "EYE_COLOR", "GREEN")))
    layer.add(g.Segment("ch5", DIM).add(g.make_role(cb, "EYE_COLOR", "GREEN")))
    layer.add(g.Segment("ch12", DIM).add(g.make_role(cb, "EYE_COLOR", "BLUE")))

    contested = layer.contested(cb, "EYE_COLOR", values)
    labels = {m.label for m in contested}
    assert labels == {"GREEN", "BLUE"}  # both surface above the noise floor


def test_layer_not_contested_when_consistent(cb):
    values = g.value_cleanup(cb, ["GREEN", "BLUE", "BROWN"])
    layer = g.Layer("Stable", DIM)
    layer.add(g.Segment("ch1", DIM).add(g.make_role(cb, "EYE_COLOR", "GREEN")))
    layer.add(g.Segment("ch5", DIM).add(g.make_role(cb, "EYE_COLOR", "GREEN")))
    assert [m.label for m in layer.contested(cb, "EYE_COLOR", values)] == ["GREEN"]


def test_layer_localizes_contradiction(cb):
    values = g.value_cleanup(cb, ["GREEN", "BLUE", "BROWN"])
    layer = g.Layer("Elena", DIM)
    layer.add(g.Segment("ch1", DIM).add(g.make_role(cb, "EYE_COLOR", "GREEN")))
    layer.add(g.Segment("ch12", DIM).add(g.make_role(cb, "EYE_COLOR", "BLUE")))
    groups = layer.localize(cb, "EYE_COLOR", values)
    assert {s.name for s in groups["GREEN"]} == {"ch1"}
    assert {s.name for s in groups["BLUE"]} == {"ch12"}


def test_cortex_routes_mention_to_right_entity(cb):
    elena = g.Layer("Elena", DIM)
    elena.add(
        g.Segment("e1", DIM)
        .add(g.make_role(cb, "EYE_COLOR", "GREEN"))
        .add(g.make_role(cb, "HAIR", "RED"))
        .add(g.make_role(cb, "LOCATION", "VILLAGE"))
    )
    marcus = g.Layer("Marcus", DIM)
    marcus.add(
        g.Segment("m1", DIM)
        .add(g.make_role(cb, "EYE_COLOR", "BROWN"))
        .add(g.make_role(cb, "HAIR", "BLACK"))
        .add(g.make_role(cb, "LOCATION", "CITY"))
    )
    cortex = g.Cortex("w", DIM).add(elena).add(marcus)

    mention = (
        g.Segment("she", DIM)
        .add(g.make_role(cb, "HAIR", "RED"))
        .add(g.make_role(cb, "LOCATION", "VILLAGE"))
    )
    routed = cortex.route(mention.vector)
    assert routed.label == "Elena"
    assert routed.margin > 0.0


def test_resonator_factors_a_product(cb):
    subj_labels = ["ELENA", "MARCUS"]
    rel_labels = ["HOLDS", "BREAKS"]
    obj_labels = ["AMULET", "SWORD"]
    codebooks = [
        [cb[s] for s in subj_labels],
        [cb[s] for s in rel_labels],
        [cb[s] for s in obj_labels],
    ]
    product = g.bind(cb["ELENA"], cb["BREAKS"], cb["AMULET"])
    decoded, _ = g.resonate(product, codebooks, [subj_labels, rel_labels, obj_labels])
    assert decoded == ["ELENA", "BREAKS", "AMULET"]
