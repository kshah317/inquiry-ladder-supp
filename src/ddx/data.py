"""Load DDXPlus and answer questions about a hidden patient record.

Everything here is deterministic: no LLM is involved. The game engine asks
`reveal(patient, code)` and gets back exactly what the patient record says.

DDXPlus encoding (checked against release_evidences.json, Oct 2026):
  * 223 evidence items: 208 binary (B), 10 categorical (C), 5 multi-choice (M).
  * In a patient's EVIDENCES list:
      'E_201'          -> binary item is present (answer "yes")
      'E_56_@_6'       -> categorical item with a numeric value (0-10 scales)
      'E_204_@_V_10'   -> categorical item with a coded value (here V_10 = "N")
      'E_55_@_V_29'    -> multi-choice item; one token per chosen value
  * An item that is absent from the list takes its `default_value`
    (0 / "NA" / "N"), i.e. "no" or "not applicable".
  * 14 items are follow-ups of a parent (code_question != own code):
      E_53 (pain) -> E_54..E_59;  E_129 (skin) -> E_130..E_136;  E_151 (swelling) -> E_152.
    Five more items point to a parent code that does not exist in the release
    (E_36, E_68, E_85, E_117, E_122); we treat those items as stand-alone questions.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SPLIT_FILES = {"test": "release_test_patients", "validate": "release_validate_patients"}


# ----------------------------------------------------------------------------- knowledge base
@dataclass(frozen=True)
class Evidence:
    code: str            # DDXPlus code, e.g. "E_201"
    question: str        # English question text (raw, before our wording fixes)
    dtype: str           # "B", "C" or "M"
    is_history: bool     # DDXPlus "is_antecedent": medical history / risk factor vs current symptom
    parent: str | None   # parent code for follow-up items, else None
    default: object      # value when the item is absent from the record
    values: tuple        # possible values (empty for binary)
    value_text: dict     # value code -> English meaning (raw)


@dataclass(frozen=True)
class Condition:
    name: str            # name as used in PATHOLOGY column (e.g. "Larygospasm" - typo is in the data)
    icd10: str
    severity: int        # 1 = most severe ... 5 = mildest


class KnowledgeBase:
    def __init__(self, raw_dir: Path = RAW):
        ev = json.loads((raw_dir / "release_evidences.json").read_text(encoding="utf-8"))
        co = json.loads((raw_dir / "release_conditions.json").read_text(encoding="utf-8"))
        self.evidence: dict[str, Evidence] = {}
        for code, v in ev.items():
            parent = v["code_question"] if v["code_question"] != code else None
            if parent is not None and parent not in ev:   # dangling parent -> stand-alone item
                parent = None
            self.evidence[code] = Evidence(
                code=code,
                question=v["question_en"],
                dtype=v["data_type"],
                is_history=bool(v["is_antecedent"]),
                parent=parent,
                default=v["default_value"],
                values=tuple(v.get("possible-values", [])),
                value_text={k: m["en"] for k, m in v.get("value_meaning", {}).items()},
            )
        self.children: dict[str, list[str]] = {}
        for e in self.evidence.values():
            if e.parent:
                self.children.setdefault(e.parent, []).append(e.code)
        for p in self.children:
            self.children[p].sort(key=_code_num)
        self.conditions: dict[str, Condition] = {
            v["cond-name-eng"]: Condition(v["cond-name-eng"], v["icd10-id"].upper(), int(v["severity"]))
            for v in co.values()
        }

    @property
    def top_level(self) -> list[str]:
        """Codes that can be asked directly (follow-ups are revealed with their parent)."""
        return sorted((c for c, e in self.evidence.items() if e.parent is None), key=_code_num)

    @property
    def condition_names(self) -> list[str]:
        return sorted(self.conditions)


def _code_num(code: str) -> int:
    return int(code.split("_")[1])


@lru_cache(maxsize=1)
def kb() -> KnowledgeBase:
    return KnowledgeBase()


# ----------------------------------------------------------------------------- patients
@dataclass
class Patient:
    split: str
    row: int                       # row index in the split file = stable patient id
    age: int
    sex: str                       # "M" / "F"
    pathology: str
    initial_evidence: str
    record: dict = field(repr=False)   # code -> True | value | set(values)

    @property
    def pid(self) -> str:
        return f"{self.split}:{self.row}"


def parse_evidences(tokens: list[str], kb_: KnowledgeBase) -> dict:
    rec: dict = {}
    for tok in tokens:
        if "_@_" not in tok:
            rec[tok] = True
            continue
        code, val = tok.split("_@_")
        e = kb_.evidence[code]
        if not str(val).startswith("V_"):
            val = int(val)
        if e.dtype == "M":
            rec.setdefault(code, set()).add(val)
        else:
            rec[code] = val
    return rec


@lru_cache(maxsize=2)
def load_split(split: str) -> pd.DataFrame:
    """Raw split as a DataFrame. Row index = patient id. EVIDENCES left as strings (parsed lazily)."""
    return pd.read_csv(RAW / SPLIT_FILES[split])


def get_patient(split: str, row: int) -> Patient:
    r = load_split(split).iloc[row]
    return Patient(
        split=split,
        row=int(row),
        age=int(r.AGE),
        sex=str(r.SEX),
        pathology=str(r.PATHOLOGY),
        initial_evidence=str(r.INITIAL_EVIDENCE),
        record=parse_evidences(ast.literal_eval(r.EVIDENCES), kb()),
    )


# ----------------------------------------------------------------------------- answer lookup
@dataclass(frozen=True)
class Answer:
    code: str
    dtype: str
    value: object      # bool (B) | int or value code (C) | tuple of value codes, maybe empty (M)
    positive: bool     # True if the answer is anything other than "no / none / default"


def answer(p: Patient, code: str) -> Answer:
    e = kb().evidence[code]
    if e.dtype == "B":
        v = code in p.record
        return Answer(code, "B", v, v)
    if e.dtype == "C":
        v = p.record.get(code, e.default)
        return Answer(code, "C", v, v != e.default and v not in ("V_10", "V_11"))
    # multi-choice
    vals = tuple(sorted(p.record.get(code, set()) - {"V_11"}, key=_code_num))
    return Answer(code, "M", vals, bool(vals))


def reveal(p: Patient, code: str) -> list[Answer]:
    """What the patient says when the panel asks `code`.

    Asking a parent question (pain, skin lesions, swelling) returns the parent
    answer and, if it is "yes", all of its follow-up answers in one turn.
    """
    first = answer(p, code)
    out = [first]
    if first.positive:
        out += [answer(p, c) for c in kb().children.get(code, [])]
    return out
