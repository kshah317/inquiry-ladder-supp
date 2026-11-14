"""Turn DDXPlus records into the plain text that agents see.

Design choices (see the paper, Appendix A):
  * Neutral question IDs (Q001...) instead of DDXPlus codes (E_201), so a model
    cannot lean on memorised dataset codes. The mapping is fixed and saved.
  * The menu lists current symptoms first, then medical history, each in code order.
  * Wording fixes from prereg/english_fixes.yaml are applied everywhere.
"""
from __future__ import annotations

from functools import lru_cache

import yaml

from .data import ROOT, Answer, Patient, kb

FIXES_PATH = ROOT / "prereg" / "english_fixes.yaml"
SEX_WORD = {"M": "male", "F": "female"}


@lru_cache(maxsize=1)
def fixes() -> dict:
    return yaml.safe_load(FIXES_PATH.read_text(encoding="utf-8"))


def question_text(code: str) -> str:
    return fixes()["questions"].get(code, kb().evidence[code].question).strip()


def value_text(code: str, value) -> str:
    if isinstance(value, int):
        return str(value)
    return fixes()["values"].get(value, kb().evidence[code].value_text.get(value, value))


def condition_display(name: str) -> str:
    return fixes()["conditions"].get(name, name)


# ----------------------------------------------------------------------------- question IDs
@lru_cache(maxsize=1)
def menu_codes() -> list[str]:
    """Askable DDXPlus codes in menu order: symptoms first, then history."""
    top = kb().top_level
    ev = kb().evidence
    return [c for c in top if not ev[c].is_history] + [c for c in top if ev[c].is_history]


@lru_cache(maxsize=1)
def qid_map() -> dict[str, str]:
    """DDXPlus code -> neutral ID (Q001...)."""
    return {code: f"Q{i:03d}" for i, code in enumerate(menu_codes(), start=1)}


@lru_cache(maxsize=1)
def code_for_qid() -> dict[str, str]:
    return {q: c for c, q in qid_map().items()}


def render_menu(exclude: set[str] = frozenset()) -> str:
    """The question list shown to agents. `exclude` = DDXPlus codes already asked."""
    ev = kb().evidence
    lines, section = [], None
    for code in menu_codes():
        if code in exclude:
            continue
        sec = "MEDICAL HISTORY AND RISK FACTORS" if ev[code].is_history else "CURRENT SYMPTOMS"
        if sec != section:
            lines.append(f"## {sec}")
            section = sec
        lines.append(f"{qid_map()[code]}: {question_text(code)}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- answers
SCALE_HINT = {  # what the 0-10 numeric scales mean (from the DDXPlus question wording)
    "E_56": "0 = no pain, 10 = worst pain",
    "E_58": "0 = very diffuse, 10 = very precisely located",
    "E_59": "0 = very slowly, 10 = very suddenly",
    "E_134": "0 = no pain, 10 = worst pain",
    "E_136": "0 = no itching, 10 = worst itching",
}


def render_answer(a: Answer) -> str:
    q = question_text(a.code)
    if a.dtype == "B":
        ans = "Yes" if a.value else "No"
    elif a.dtype == "C":
        ans = value_text(a.code, a.value)
        if isinstance(a.value, int):
            hint = SCALE_HINT.get(a.code)
            ans = f"{a.value} on a 0-10 scale" + (f" ({hint})" if hint else "")
    else:
        ans = ", ".join(value_text(a.code, v) for v in a.value) if a.value else "None"
    return f"{q} -> {ans}"


def render_reveal(answers: list[Answer]) -> str:
    """One consultation turn: the main answer, then follow-up details indented."""
    head, *rest = answers
    out = [render_answer(head)]
    out += [f"   - {render_answer(a)}" for a in rest]
    return "\n".join(out)


# ----------------------------------------------------------------------------- opening case
def render_opening(p: Patient) -> str:
    """What every agent sees before round 1: age, sex and the opening complaint only."""
    age = "under 1 year old" if p.age == 0 else f"{p.age} years old"
    return (
        f"Patient: {SEX_WORD[p.sex]}, {age}.\n"
        f"Opening complaint: asked \"{question_text(p.initial_evidence)}\", the patient answered Yes."
        + (f" (Details not yet known: ask {qid_map()[p.initial_evidence]} to hear them.)"
           if kb().children.get(p.initial_evidence) else "")
    )


def render_condition_list() -> str:
    return "\n".join(f"- {condition_display(n)}" for n in kb().condition_names)


def render_full_record(p: Patient) -> str:
    """All positive answers in the record (used only by the check panel's 'strong' agent)."""
    from .data import reveal
    out = []
    for code in menu_codes():
        ans = reveal(p, code)
        if ans[0].positive:
            out.append(render_reveal(ans))
    return "\n".join(out)
