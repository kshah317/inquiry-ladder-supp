"""Build the two planned specialty maps.

1) prereg/condition_specialty.csv  (hand-written below, with a reason for each call)
2) prereg/evidence_specialty.csv   (derived from VALIDATION patients by a fixed rule)

Run once, review, then freeze. Re-running must reproduce the same files.

Rule for conditions
  primary = the specialty that would normally lead the diagnostic work-up for this
  presentation. "Other" = outside cardiology, pulmonology and infectious disease
  (ENT, GI, neuro, psych, allergy, haematology, rheumatology, ortho, oncology outside the lung).
  If two of our three specialties have an equal claim, the condition is flagged
  ambiguous=1 and dropped from the in-field / out-of-field split (it still counts
  for every overall accuracy number). The ICD-10 chapter is kept as an alternative
  map for a sensitivity check.

Rule for evidence items: TWO columns
  `specialty` (PRIMARY, used by the manipulation check) = face-validity label: a
  question is tagged to a specialty only if it names that specialty's organ system,
  one of its diseases or treatments, or a textbook sign / risk factor of it
  (e.g. palpitations, orthopnoea, chest pain at rest, cholesterol -> Cardiology).
  Questions that two specialties would claim equally (smoking, generic fatigue,
  weight loss) are "General". Reason: the manipulation check asks "did the
  cardiologist ask what a cardiologist would ask?", which is about the role as
  written, not about quirks of the DDXPlus knowledge base.
  `kb_specialty` (SECONDARY, appendix only) = data-driven label from validation
  patients, which tells us which questions are actually informative for each group:
  For each askable item e and condition k: r[e,k] = share of validation patients
  with condition k for whom e is positive. Each condition counts equally
  (prevalence does not matter). For each group G: s[e,G] = mean of r[e,k] over
  conditions in G. lift = s[e,G] / mean of r[e,k] over all unambiguous conditions.
  e is tagged to the G with the highest lift if lift >= min_lift AND
  max over k in G of r[e,k] >= min_positive_weight; otherwise "General".
  Follow-up items (pain details etc.) share their parent's tag.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddx.data import answer, kb, load_split, parse_evidences, Patient  # noqa: E402
from ddx.text import menu_codes, qid_map, question_text  # noqa: E402

CARD, PULM, ID, OTHER = "Cardiology", "Pulmonology", "Infectious disease", "Other"
GROUPS = [CARD, PULM, ID]

# name (dataset key) -> (primary specialty, ambiguous, reason)
CONDITIONS = {
    # --- Cardiology (8)
    "Possible NSTEMI / STEMI": (CARD, 0, "Acute coronary syndrome"),
    "Unstable angina": (CARD, 0, "Acute coronary syndrome"),
    "Stable angina": (CARD, 0, "Coronary artery disease"),
    "Pericarditis": (CARD, 0, "Pericardial disease"),
    "Myocarditis": (CARD, 0, "Myocardial disease (often viral cause, but cardiology leads work-up)"),
    "Atrial fibrillation": (CARD, 0, "Arrhythmia"),
    "PSVT": (CARD, 0, "Arrhythmia"),
    "Acute pulmonary edema": (CARD, 0, "Cardiogenic in the DDXPlus knowledge base (heart failure history); ICD chapter J"),
    # --- Pulmonology (7)
    "Acute COPD exacerbation / infection": (PULM, 0, "Chronic airway disease"),
    "Bronchospasm / acute asthma exacerbation": (PULM, 0, "Airway disease"),
    "Bronchiectasis": (PULM, 0, "Chronic airway disease"),
    "Bronchitis": (PULM, 0, "Lower airway; managed as respiratory, not as a systemic infection"),
    "Spontaneous pneumothorax": (PULM, 0, "Pleural disease"),
    "Pulmonary neoplasm": (PULM, 0, "Lung cancer is usually first worked up by pulmonology"),
    "Sarcoidosis": (PULM, 0, "Mostly pulmonary presentation (ICD chapter D)"),
    # --- Infectious disease (6)
    "HIV (initial infection)": (ID, 0, "Systemic viral infection"),
    "Ebola": (ID, 0, "Systemic viral infection; exposure history"),
    "Tuberculosis": (ID, 0, "Mycobacterial infection; ICD chapter A"),
    "Chagas": (ID, 0, "Parasitic infection; travel/exposure history"),
    "Whooping cough": (ID, 0, "Bacterial infection; ICD chapter A"),
    "Influenza": (ID, 0, "Systemic viral syndrome"),
    # --- Ambiguous between two of our specialties (3): excluded from in-field analysis
    "Pneumonia": (PULM, 1, "Pulmonology and infectious disease have an equal claim"),
    "Pulmonary embolism": (PULM, 1, "Pulmonology and cardiology have an equal claim (ICD chapter I)"),
    "Bronchiolitis": (PULM, 1, "Paediatric viral lower-airway infection: pulmonology vs infectious disease"),
    # --- Other: out of field for every specialist (25)
    "URTI": (OTHER, 0, "ENT / primary care"),
    "Viral pharyngitis": (OTHER, 0, "ENT / primary care"),
    "Acute laryngitis": (OTHER, 0, "ENT"),
    "Acute otitis media": (OTHER, 0, "ENT"),
    "Acute rhinosinusitis": (OTHER, 0, "ENT"),
    "Chronic rhinosinusitis": (OTHER, 0, "ENT"),
    "Allergic sinusitis": (OTHER, 0, "ENT / allergy"),
    "Epiglottitis": (OTHER, 0, "ENT / airway emergency"),
    "Croup": (OTHER, 0, "Paediatrics / ENT"),
    "Larygospasm": (OTHER, 0, "ENT / airway"),
    "GERD": (OTHER, 0, "Gastroenterology"),
    "Boerhaave": (OTHER, 0, "Gastroenterology / surgery"),
    "Inguinal hernia": (OTHER, 0, "Surgery"),
    "Pancreatic neoplasm": (OTHER, 0, "Oncology / gastroenterology"),
    "Anemia": (OTHER, 0, "Haematology"),
    "Panic attack": (OTHER, 0, "Psychiatry"),
    "Anaphylaxis": (OTHER, 0, "Allergy / emergency"),
    "Scombroid food poisoning": (OTHER, 0, "Toxicology / emergency"),
    "Guillain-Barré syndrome": (OTHER, 0, "Neurology"),
    "Myasthenia gravis": (OTHER, 0, "Neurology"),
    "Acute dystonic reactions": (OTHER, 0, "Neurology / drug reaction"),
    "Cluster headache": (OTHER, 0, "Neurology"),
    "SLE": (OTHER, 0, "Rheumatology"),
    "Localized edema": (OTHER, 0, "Internal medicine (non-pulmonary oedema)"),
    "Spontaneous rib fracture": (OTHER, 0, "Orthopaedics"),
}


# Face-validity labels (PRIMARY). Everything not listed is "General".
SEMANTIC = {
    CARD: """E_13 E_14 E_33 E_67 E_155 E_164 E_217 E_218
             E_3 E_5 E_22 E_69 E_71 E_102 E_104 E_105 E_106 E_108 E_139 E_146 E_149 E_184 E_225""",
    PULM: """E_23 E_45 E_64 E_66 E_77 E_112 E_201 E_203 E_214 E_220
             E_18 E_21 E_31 E_46 E_72 E_87 E_101 E_118 E_123 E_124 E_142 E_165 E_186 E_199 E_200 E_222 E_224""",
    ID: """E_9 E_91 E_94 E_166
           E_0 E_2 E_27 E_40 E_41 E_48 E_49 E_61 E_73 E_115 E_189 E_204 E_209 E_227""",
}
SEMANTIC_OF = {code: g for g, codes in SEMANTIC.items() for code in codes.split()}


def icd_chapter_group(icd: str) -> str:
    """Alternative map for a sensitivity check: ICD-10 chapter only."""
    first = icd.strip().upper()[0]
    if first == "I":
        return CARD
    if first in "AB":
        return ID
    if first == "J":
        num = int("".join(ch for ch in icd.split(",")[0][1:3] if ch.isdigit()) or 0)
        return OTHER if num <= 6 or 30 <= num <= 39 else PULM   # J00-J06, J30-J39 = upper airway
    return OTHER


def build_condition_map() -> pd.DataFrame:
    k = kb()
    missing = set(k.conditions) ^ set(CONDITIONS)
    assert not missing, f"condition names do not match the dataset: {missing}"
    rows = []
    for name in sorted(CONDITIONS):
        spec, amb, why = CONDITIONS[name]
        c = k.conditions[name]
        rows.append(dict(condition=name, icd10=c.icd10, severity=c.severity, specialty=spec,
                         ambiguous=amb, icd_chapter_specialty=icd_chapter_group(c.icd10), reason=why))
    return pd.DataFrame(rows)


def build_evidence_map(cmap: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    split = cfg["evidence_map"]["derive_from_split"]
    df = load_split(split)
    k = kb()
    codes = menu_codes()
    conds = sorted(df.PATHOLOGY.unique())
    # r[e, k]: share of patients with condition k for whom item e is positive
    counts = pd.DataFrame(0.0, index=codes, columns=conds)
    n_by_cond = df.PATHOLOGY.value_counts()
    for ev_str, path in zip(df.EVIDENCES, df.PATHOLOGY):
        rec = parse_evidences(ast.literal_eval(ev_str), k)
        p = Patient("x", 0, 0, "M", path, "", rec)
        for code in codes:
            if code in rec and answer(p, code).positive:
                counts.at[code, path] += 1
    r = counts / n_by_cond[conds].to_numpy()
    spec = cmap.set_index("condition")
    unamb = [c for c in conds if spec.at[c, "ambiguous"] == 0]
    base = r[unamb].mean(axis=1)
    rows = []
    for code in codes:
        lifts, maxr = {}, {}
        for g in GROUPS:
            members = [c for c in unamb if spec.at[c, "specialty"] == g]
            lifts[g] = (r.loc[code, members].mean() / base[code]) if base[code] > 0 else 0.0
            maxr[g] = r.loc[code, members].max()
        best = max(GROUPS, key=lambda g: lifts[g])
        ok = lifts[best] >= cfg["evidence_map"]["min_lift"] and maxr[best] >= cfg["evidence_map"]["min_positive_weight"]
        top_conds = r.loc[code].sort_values(ascending=False).head(3)
        rows.append(dict(
            code=code, qid=qid_map()[code], question=question_text(code),
            history=int(k.evidence[code].is_history),
            specialty=SEMANTIC_OF.get(code, "General"),
            kb_specialty=best if ok else "General",
            lift_cardio=round(lifts[CARD], 2), lift_pulm=round(lifts[PULM], 2), lift_id=round(lifts[ID], 2),
            top_conditions="; ".join(f"{c} ({v:.2f})" for c, v in top_conds.items()),
        ))
    out = pd.DataFrame(rows)
    # follow-ups inherit their parent's tag (they are only revealed with the parent)
    kids = [dict(code=c, qid="(with " + qid_map()[e.parent] + ")", question=question_text(c),
                 history=int(e.is_history), specialty=out.set_index("code").at[e.parent, "specialty"],
                 kb_specialty=out.set_index("code").at[e.parent, "kb_specialty"],
                 lift_cardio=np.nan, lift_pulm=np.nan, lift_id=np.nan, top_conditions="follow-up of " + e.parent)
            for c, e in k.evidence.items() if e.parent]
    return pd.concat([out, pd.DataFrame(kids)], ignore_index=True)


def main():
    cfg = yaml.safe_load((ROOT / "config" / "study.yaml").read_text())
    cmap = build_condition_map()
    cmap.to_csv(ROOT / "prereg" / "condition_specialty.csv", index=False)
    print(cmap.groupby(["specialty", "ambiguous"]).size())
    emap = build_evidence_map(cmap, cfg)
    emap.to_csv(ROOT / "prereg" / "evidence_specialty.csv", index=False)
    assert set(SEMANTIC_OF) <= set(emap.code), set(SEMANTIC_OF) - set(emap.code)
    print(pd.crosstab(emap.specialty, emap.kb_specialty, margins=True))
    pd.DataFrame([dict(code=c, qid=q) for c, q in qid_map().items()]).to_csv(ROOT / "prereg" / "question_ids.csv", index=False)


if __name__ == "__main__":
    main()
