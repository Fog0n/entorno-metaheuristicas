"""Registro de algoritmos disponibles."""

from __future__ import annotations

from algorithms.aco import AntColonyAlgorithm
from algorithms.ais import ClonalSelectionAlgorithm
from algorithms.de import DifferentialEvolutionAlgorithm
from algorithms.ga import GeneticAlgorithm
from algorithms.pso import ParticleSwarmAlgorithm
from core.base import MetaheuristicAlgorithm

ALGORITHMS: dict[str, type[MetaheuristicAlgorithm]] = {
    "ga": GeneticAlgorithm,
    "aco": AntColonyAlgorithm,
    "pso": ParticleSwarmAlgorithm,
    "ais": ClonalSelectionAlgorithm,
    "de": DifferentialEvolutionAlgorithm,
}

ALGORITHM_ORDER = ("ga", "aco", "pso", "ais", "de")


def get_algorithm(key: str) -> type[MetaheuristicAlgorithm]:
    try:
        return ALGORITHMS[key]
    except KeyError as exc:
        raise ValueError(f"Algoritmo desconocido: {key}") from exc


def is_compatible(key: str, family: str) -> bool:
    return family in get_algorithm(key).family_support


__all__ = [
    "ALGORITHMS",
    "ALGORITHM_ORDER",
    "get_algorithm",
    "is_compatible",
]
