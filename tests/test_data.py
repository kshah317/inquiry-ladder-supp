"""Checks for the loader, answer lookup, text templates and maps. Run: python3 tests/test_data.py"""
import ast
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddx.data import answer, get_patient, kb, load_split, reveal  # noqa: E402
from ddx.text import code_for_qid, menu_codes, qid_map, render_menu, render_opening, render_reveal  # noqa: E402
from ddx.sample import load_sample  # noqa: E402


def test_kb_counts():
    k = kb()
    assert len(k.evidence) == 223 and len(k.conditions) == 49
    types = pd.Series([e.dtype for e in k.evidence.values()]).value_counts().to_dict()
    assert types == {"B": 208, "C": 10, "M": 5}, types
    assert len(menu_codes()) == 209                       # 223 - 14 follow-ups
    assert sum(len(v) for v in k.children.values()) == 14
    assert len(set(qid_map().values())) == 209 and code_for_qid()["Q001"] == menu_codes()[0]


def test_initial_evidence_is_binary_and_askable():
    df = load_split("test")
    codes = set(df.INITIAL_EVIDENCE.unique())
    assert all(kb().evidence[c].dtype == "B" for c in codes)
    assert codes <= set(menu_codes())


def test_answer_roundtrip_on_sample():
    """Every token in a patient's record must be recovered by answer(); nothing extra invented."""
    s = load_sample("main")
    df = load_split("test")
    for row in s.row:
        p = get_patient("test", int(row))
        tokens = ast.literal_eval(df.EVIDENCES.iloc[int(row)])
        for tok in tokens:
            code, _, val = tok.partition("_@_")
            a = answer(p, code)
            if not val:
                assert a.value is True
            elif kb().evidence[code].dtype == "M":
                assert val in a.value or val == "V_11"
            else:
                assert str(a.value) == val
        # an item absent from the record is "no"/default
        absent = [c for c in menu_codes() if c not in p.record][:5]
        assert all(not answer(p, c).positive for c in absent)
        assert answer(p, p.initial_evidence).positive


def test_reveal_bundles_followups():
    df = load_split("test")
    row = int(df.index[df.EVIDENCES.str.contains("'E_53'")][0])
    p = get_patient("test", row)
    out = reveal(p, "E_53")
    assert [a.code for a in out] == ["E_53"] + kb().children["E_53"]
    text = render_reveal(out)
    assert "->" in text and "0-10 scale" in text


def test_text_has_no_dataset_codes_or_bad_translations():
    menu = render_menu()
    assert "E_" not in menu and "V_" not in menu
    for bad in ["heartbreaking", "haunting", "palace", "whoooping", "live in in "]:
        assert bad not in menu
    p = get_patient("test", int(load_sample("main").row[0]))
    assert "E_" not in render_opening(p)


def test_maps_complete():
    c = pd.read_csv(ROOT / "prereg" / "condition_specialty.csv")
    e = pd.read_csv(ROOT / "prereg" / "evidence_specialty.csv")
    assert set(c.condition) == set(kb().conditions) and len(c) == 49
    assert set(e.code) == set(kb().evidence) and len(e) == 223
    assert set(c.specialty) == {"Cardiology", "Pulmonology", "Infectious disease", "Other"}


def test_sample_stratified():
    s = load_sample("main")
    assert len(s) == 294 and s.PATHOLOGY.value_counts().eq(6).all() and s.split.eq("test").all()
    p = load_sample("pilot")
    assert len(p) == 98 and p.split.eq("validate").all() and p.head(49).PATHOLOGY.nunique() == 49


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except Exception as ex:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(ex))
    sys.exit(fails)
