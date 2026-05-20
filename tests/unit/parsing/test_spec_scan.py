# pylint: disable=missing-docstring
from __future__ import annotations

from zetta_utils.parsing.spec_scan import TypeRef, extract_types


def test_flat_spec():
    spec = {"@type": "build_cv_layer", "path": "gs://x"}
    res = extract_types(spec)
    assert res.types == (TypeRef(name="build_cv_layer", version=None),)
    assert res.has_dynamic_types is False


def test_nested_dicts_and_lists():
    spec = {
        "@type": "outer",
        "child": {"@type": "inner_a", "@version": "1.2"},
        "children": [
            {"@type": "inner_b"},
            {"@type": "inner_c"},
            {"unrelated": True},
        ],
    }
    res = extract_types(spec)
    names = res.names()
    assert names == {"outer", "inner_a", "inner_b", "inner_c"}
    inner_a = next(t for t in res.types if t.name == "inner_a")
    assert inner_a.version == "1.2"
    assert res.has_dynamic_types is False


def test_dynamic_type_marked():
    spec = {"@type": ["not", "a", "string"]}
    res = extract_types(spec)
    assert res.has_dynamic_types is True
    assert res.types == ()


def test_no_at_type_keys():
    spec = {"a": 1, "b": [2, 3], "c": {"d": "e"}}
    res = extract_types(spec)
    assert res.types == ()
    assert res.has_dynamic_types is False


def test_tuples_walked():
    spec = ({"@type": "in_tuple"},)
    res = extract_types(spec)
    assert res.names() == {"in_tuple"}


def test_top_level_list():
    spec = [{"@type": "x"}, {"@type": "y"}]
    res = extract_types(spec)
    assert res.names() == {"x", "y"}
