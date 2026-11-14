"""Panel definitions. Each panel has 4 agents; every one of its 15 non-empty subgroups plays its own game.

An agent's seed_id is its identity: the same agent (same model, role, seed_id) appearing in two
panels makes identical requests when it plays alone, so those games are shared through the cache.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import combinations

from .game import Agent


def _a(name, model, role, sid, full=False):
    return Agent(name=name, model=model, role=role, seed_id=sid, full_record=full)


G4_GEN = _a("Physician A", "gemma4_31b", "generalist", "G4.gen.a")
Q38_GEN = _a("Physician C", "qwen38_27b", "generalist", "Q38.gen.a")

PANELS = {
    # rung 0: same model, sampled independently (also the compute-matched single-model baseline)
    "P0": [G4_GEN,
           _a("Physician B", "gemma4_31b", "generalist", "G4.gen.b"),
           _a("Physician C", "gemma4_31b", "generalist", "G4.gen.c"),
           _a("Physician D", "gemma4_31b", "generalist", "G4.gen.d")],
    # rung 1: role prompts only
    "P1": [_a("Cardiologist", "gemma4_31b", "cardiology", "G4.card"),
           _a("Pulmonologist", "gemma4_31b", "pulmonology", "G4.pulm"),
           _a("Infectious disease specialist", "gemma4_31b", "infectious", "G4.id"),
           _a("Generalist", "gemma4_31b", "generalist", "G4.gen.a")],
    "P1Q": [_a("Cardiologist", "qwen38_27b", "cardiology", "Q38.card"),
            _a("Pulmonologist", "qwen38_27b", "pulmonology", "Q38.pulm"),
            _a("Infectious disease specialist", "qwen38_27b", "infectious", "Q38.id"),
            _a("Generalist", "qwen38_27b", "generalist", "Q38.gen.a")],
    # same-model baseline on the second family: only A, B, C, D and ABCD are played
    "P0Q": [_a("Physician A", "qwen38_27b", "generalist", "Q38.gen.a"),
            _a("Physician B", "qwen38_27b", "generalist", "Q38.gen.b"),
            _a("Physician C", "qwen38_27b", "generalist", "Q38.gen.c"),
            _a("Physician D", "qwen38_27b", "generalist", "Q38.gen.d")],
    # rungs 2 and 3: different models (same family pairs and cross-family pairs within one panel)
    "P2": [G4_GEN,
           _a("Physician B", "gemma4_26b_moe", "generalist", "G4moe.gen.a"),
           Q38_GEN,
           _a("Physician D", "qwen36_27b", "generalist", "Q36.gen.a")],
    # positive control: Shapley must separate a weak and a strong agent
    "CHECK": [_a("Cardiologist", "gemma4_31b", "cardiology", "G4.card"),
              _a("Pulmonologist", "gemma4_31b", "pulmonology", "G4.pulm"),
              _a("Physician C", "gemma3_4b", "generalist", "G3s.gen.a"),
              _a("Physician D", "gemma4_31b", "generalist", "G4.full", full=True)],
    # cost/accuracy anchor: one large API model playing the solo game
    "ANCHOR": [_a("Physician A", "qwen38_max", "generalist", "Q38max.gen.a")],
}

# pair types for the error-overlap analysis (indices into the panel)
PAIR_TYPE = {
    "P0": {pair: "same model" for pair in combinations(range(4), 2)},
    "P1": {pair: "persona" for pair in combinations(range(4), 2)},
    "P1Q": {pair: "persona" for pair in combinations(range(4), 2)},
    "P0Q": {pair: "same model" for pair in combinations(range(4), 2)},
    "P2": {(0, 1): "same family", (2, 3): "same family",
           (0, 2): "cross family", (0, 3): "cross family", (1, 2): "cross family", (1, 3): "cross family"},
}


def subgroups(n: int = 4) -> list[tuple[int, ...]]:
    return [c for k in range(1, n + 1) for c in combinations(range(n), k)]


def sg_name(sg: tuple[int, ...]) -> str:
    return "".join("ABCD"[i] for i in sg)


def replicate(agents: list[Agent], replica: int) -> list[Agent]:
    """Same agents with a replica id: replica 0 = the main run, r > 0 = an independent repeat whose
    seeds (and therefore cache keys) are all new, so no cached sample is reused."""
    return [replace(a, replica=replica) for a in agents]
