"""Answer lookup, one test per DDXPlus evidence type. Uses hand-built records, so each rule is
checked in isolation (no dependence on which patients happen to be sampled).
Run: python3 tests/test_evidence_types.py"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddx.data import Patient, answer, get_patient, kb, parse_evidences, reveal  # noqa: E402
from ddx.game import _vote_info, match_condition  # noqa: E402
from ddx.text import menu_codes, render_answer, render_reveal  # noqa: E402


def P(tokens):
    return Patient("test", -1, 40, "F", "GERD", "E_53", parse_evidences(tokens, kb()))


def test_binary_present_and_absent():
    p = P(["E_201"])
    a = answer(p, "E_201")
    assert a.value is True and a.positive and render_answer(a).endswith("-> Yes")
    b = answer(p, "E_91")
    assert b.value is False and not b.positive and render_answer(b).endswith("-> No")


def test_categorical_numeric_with_default():
    p = P(["E_53", "E_56_@_7"])
    a = answer(p, "E_56")                         # pain intensity given
    assert a.value == 7 and a.positive and "7 on a 0-10 scale" in render_answer(a)
    d = answer(p, "E_59")                         # pain onset speed absent -> default 0, not positive
    assert d.value == kb().evidence["E_59"].default == 0 and not d.positive


def test_categorical_coded_with_default():
    # E_204 recent travel: default V_10 ("N") = no travel
    assert not answer(P([]), "E_204").positive and answer(P([]), "E_204").value == "V_10"
    assert not answer(P(["E_204_@_V_10"]), "E_204").positive
    other = next(v for v in kb().evidence["E_204"].values if v not in ("V_10", "V_11"))
    t = answer(P([f"E_204_@_{other}"]), "E_204")
    assert t.positive and t.value == other and "V_" not in render_answer(t)
    # E_130 skin colour: explicit "NA" (V_11) counts as not positive
    assert not answer(P(["E_129", "E_130_@_V_11"]), "E_130").positive


def test_multichoice_absent_is_none_and_present_lists_values():
    p = P(["E_53", "E_55_@_V_14", "E_55_@_V_15"])
    a = answer(p, "E_55")
    assert a.value == ("V_14", "V_15") and a.positive
    assert "V_" not in render_answer(a) and "," in render_answer(a)
    m = answer(p, "E_57")                         # absent multi-choice -> empty, shown as "None"
    assert m.value == () and not m.positive and render_answer(m).endswith("-> None")
    na = answer(P(["E_53", "E_54_@_V_11"]), "E_54")   # explicit NA dropped
    assert na.value == () and not na.positive


def test_followups_only_revealed_when_parent_is_yes():
    no = reveal(P([]), "E_53")
    assert [a.code for a in no] == ["E_53"] and not no[0].positive
    yes = reveal(P(["E_53", "E_56_@_4"]), "E_53")
    assert [a.code for a in yes] == ["E_53"] + kb().children["E_53"]
    by = {a.code: a for a in yes}
    assert by["E_56"].value == 4 and by["E_58"].value == 0     # missing detail -> default, still shown
    text = render_reveal(yes)
    assert text.count("\n   - ") == len(kb().children["E_53"])


def test_followups_not_askable_and_dangling_parents_are():
    menu = set(menu_codes())
    kids = {c for v in kb().children.values() for c in v}
    assert not (menu & kids) and len(kids) == 14
    for c in ["E_37", "E_69", "E_87", "E_118", "E_123"]:   # their parents (E_36, E_68, E_85, E_117, E_122) are missing
        assert c in menu and kb().evidence[c].parent is None


def test_all_multichoice_items_are_followups():
    """Multi-choice items are never asked directly, so their `positive` flag never drives a round."""
    for c, e in kb().evidence.items():
        if e.dtype == "M":
            assert e.parent is not None, c


def test_full_record_roundtrip_random_patients():
    """Random test-split patients: every reveal of every askable question renders without codes."""
    import pandas as pd  # noqa: F401
    rng = random.Random(7)
    for row in rng.sample(range(130000), 30):
        p = get_patient("test", row)
        for c in menu_codes():
            t = render_reveal(reveal(p, c))
            assert "E_" not in t and "V_" not in t, (row, c, t)


def test_vote_resolution():
    rng = random.Random(0)
    assert _vote_info([], rng) == (None, None)
    assert _vote_info([("x", 0.5)], rng) == ("x", "single")
    assert _vote_info([("x", .1), ("x", .2), ("y", .9)], rng) == ("x", "majority")
    assert _vote_info([("x", .1), ("y", .9)], rng) == ("y", "confidence")
    w, how = _vote_info([("x", .5), ("y", .5)], rng)
    assert how == "coin" and w in ("x", "y")


def test_condition_matching():
    for n in kb().condition_names:
        assert match_condition(n) == n
    assert match_condition("not a disease") is None and match_condition("") is None


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except Exception as ex:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(ex)[:300])
    sys.exit(fails)
