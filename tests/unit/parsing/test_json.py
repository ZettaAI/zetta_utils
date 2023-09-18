from io import StringIO

from zetta_utils.parsing import json


def test_str_roundtrip():
    assert json.loads(json.dumps({"1": "2"})) == {"1": "2"}


def test_str_tuple_roundtrip():
    assert json.loads(json.dumps((1, "2"))) == (1, "2")


def test_str_nested_tuple_roundtrip():
    assert json.loads(json.dumps((1, "2", (3, "4")))) == (1, "2", (3, "4"))


def test_fp_roundtrip():
    fp = StringIO()
    json.dump({"1": "2"}, fp)
    fp.seek(0)
    assert json.load(fp) == {"1": "2"}


def test_fp_tuple_roundtrip():
    fp = StringIO()
    json.dump((1, "2"), fp)
    fp.seek(0)
    assert json.load(fp) == (1, "2")


def test_fp_nested_tuple_roundtrip():
    fp = StringIO()
    json.dump((1, "2", (3, "4")), fp)
    fp.seek(0)
    assert json.load(fp) == (1, "2", (3, "4"))
