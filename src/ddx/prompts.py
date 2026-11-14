"""One shared prompt template for every agent in every panel.

Only the ROLE block differs between agents: one role line + one short checklist line,
kept about the same length for all roles (the generalist gets a neutral line, no specialty).
The long, identical part (rules + condition list + question list) comes first, so
providers that cache prompt prefixes can reuse it across all agents and rounds.
Checklists use generic clinical wording fixed before the pilot; they are NOT taken
from the DDXPlus marker table (paper, Appendix A).
"""
from __future__ import annotations

from functools import lru_cache

from .text import render_condition_list, render_menu

ROLES = {
    "cardiology": ("You are a cardiologist.",
                   "In this consultation, pay particular attention to chest pain character, exertion, palpitations and heart history."),
    "pulmonology": ("You are a pulmonologist.",
                    "In this consultation, pay particular attention to breathing difficulty, cough and sputum, wheezing and lung history."),
    "infectious": ("You are an infectious disease specialist.",
                   "In this consultation, pay particular attention to fever, infectious contacts, travel, vaccination and immune status."),
    "generalist": ("You are a general internal medicine physician.",
                   "In this consultation, consider the full range of conditions in the list without favouring any one organ system."),
}
ROLE_LABEL = {"cardiology": "Cardiologist", "pulmonology": "Pulmonologist",
              "infectious": "Infectious disease specialist", "generalist": "Generalist"}


@lru_cache(maxsize=1)
def system_prompt() -> str:
    return f"""You are a physician taking part in a diagnostic consultation with a patient.
You learn about the patient only through questions. The consultation works like this:
1. Each round, every physician on the panel proposes ONE question from the QUESTION LIST. The panel asks the question with the most votes, and the patient's answer is shared with everyone.
2. After the last round, each physician gives a final diagnosis, chosen from the CONDITION LIST.
3. Physicians may then see each other's final diagnoses and revise their own once.

Rules:
- Propose only question IDs that appear in the QUESTION LIST (for example "Q052"). Never propose a question that has already been asked.
- Asking about pain, skin problems or swelling also reveals the details (location, character, intensity) if the answer is yes.
- Every diagnosis you name must be copied exactly from the CONDITION LIST.
- Answer with a single JSON object and nothing else.

CONDITION LIST (the patient has exactly one of these):
{render_condition_list()}

QUESTION LIST:
{render_menu()}"""


def role_block(role: str) -> str:
    line, checklist = ROLES[role]
    return f"YOUR ROLE: {line} {checklist}"


def case_block(opening: str, transcript: list[str], full_record: str | None = None) -> str:
    lines = ["CASE SO FAR:", opening]
    if full_record:
        lines += ["", "You have also been given the patient's full answers to all questions in advance:", full_record]
    lines.append("")
    if transcript:
        lines.append("Questions asked by the panel so far:")
        lines += transcript
    else:
        lines.append("No questions have been asked yet.")
    return "\n".join(lines)


PROPOSE_TASK = """TASK: This is round {k} of {n}. The panel has {m} physician(s). Propose the single most useful next question.
Return JSON exactly in this form:
{{"question_id": "Q###", "top3": ["condition", "condition", "condition"], "confidence": <0-100, how confident you are in your first condition>, "reasoning": "<1-3 sentences>"}}"""

FINAL_TASK = """TASK: {intro} Give your final diagnosis.
Return JSON exactly in this form:
{{"diagnosis": "condition", "top3": ["condition", "condition", "condition"], "confidence": <0-100>, "reasoning": "<1-3 sentences>"}}"""

DELIB_TASK = """The panel's final diagnoses are:
{peers}

TASK: You may keep your diagnosis or change it once, using what the others said. Return JSON exactly in this form:
{{"diagnosis": "condition", "top3": ["condition", "condition", "condition"], "confidence": <0-100>, "reasoning": "<1-3 sentences>"}}"""


def propose_messages(role, opening, transcript, k, n, m, full_record=None):
    user = "\n\n".join([role_block(role), case_block(opening, transcript, full_record),
                        PROPOSE_TASK.format(k=k, n=n, m=m)])
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}]


def final_messages(role, opening, transcript, n, full_record=None):
    intro = ("No questions were asked; decide from the opening complaint only." if n == 0
             else f"All {n} questions have been asked.")
    user = "\n\n".join([role_block(role), case_block(opening, transcript, full_record),
                        FINAL_TASK.format(intro=intro)])
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}]


def delib_messages(final_msgs, own_reply: str, peers: str):
    return final_msgs + [{"role": "assistant", "content": own_reply},
                         {"role": "user", "content": DELIB_TASK.format(peers=peers)}]


# No-peer reconsideration control: same turn as DELIB_TASK, no peer answers.
NOPEER_TASK = """The other physicians' final diagnoses are not shown to you in this step.

TASK: You may keep your diagnosis or change it once, after taking a second look at the case. Return JSON exactly in this form:
{"diagnosis": "condition", "top3": ["condition", "condition", "condition"], "confidence": <0-100>, "reasoning": "<1-3 sentences>"}"""


def nopeer_messages(final_msgs, own_reply: str):
    return final_msgs + [{"role": "assistant", "content": own_reply},
                         {"role": "user", "content": NOPEER_TASK}]
