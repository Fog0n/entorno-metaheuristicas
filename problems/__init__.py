"""Registro de problemas disponibles."""

from __future__ import annotations

from core.base import Problem
from problems.binary import KnapsackProblem, OneMaxProblem
from problems.continuous import RastriginProblem, SphereProblem
from problems.delivery_zapopan import DeliveryZapopanProblem
from problems.discrete import TSPProblem

PROBLEMS: dict[str, type[Problem]] = {
    "sphere": SphereProblem,
    "rastrigin": RastriginProblem,
    "onemax": OneMaxProblem,
    "knapsack": KnapsackProblem,
    "tsp": TSPProblem,
    "delivery_zapopan": DeliveryZapopanProblem,
}

PROBLEM_ORDER = ("sphere", "rastrigin", "onemax", "knapsack", "tsp", "delivery_zapopan")


def get_problem(key: str) -> type[Problem]:
    try:
        return PROBLEMS[key]
    except KeyError as exc:
        raise ValueError(f"Problema desconocido: {key}") from exc


__all__ = ["PROBLEMS", "PROBLEM_ORDER", "get_problem"]
