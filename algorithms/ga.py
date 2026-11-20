"""Algoritmo Genético (GA) con selección, cruza y mutación adaptadas.

Soporta las tres familias de representación:
- continua: cruza de 1 punto o BLX-alfa y mutación gaussiana;
- binaria: cruza de 1 punto y mutación por volteo de bits;
- permutaciones: cruza OX y mutación por intercambio.

Se implementan torneo y ruleta, elitismo y tasas configurables.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.base import (
    MetaheuristicAlgorithm,
    ParamSpec,
    Problem,
    ox_crossover,
    swap_mutation,
)


class GeneticAlgorithm(MetaheuristicAlgorithm):
    key = "ga"
    name = "Algoritmo Genético"

    @classmethod
    def param_schema(cls, family: str | None = None) -> tuple[ParamSpec, ...]:
        return (
            ParamSpec(
                "population_size", "Tamaño de población", "int", 60, minimum=4,
                maximum=1000, help="Número de individuos por generación.",
            ),
            ParamSpec(
                "selection", "Selección", "choice", "torneo",
                choices=("torneo", "ruleta"),
                help="Método para elegir padres.",
            ),
            ParamSpec(
                "tournament_size", "Tamaño del torneo", "int", 3, minimum=2,
                maximum=20, help="Participantes por torneo (solo con torneo).",
            ),
            ParamSpec(
                "crossover", "Cruza", "choice", "auto",
                choices=("auto", "1-punto", "BLX-alfa", "OX"),
                choices_by_family={
                    "continuous": ("auto", "1-punto", "BLX-alfa"),
                    "binary": ("auto", "1-punto"),
                    "permutation": ("auto", "OX"),
                },
                help="'auto' elige BLX-alfa/OX/1-punto según la familia.",
            ),
            ParamSpec(
                "crossover_rate", "Probabilidad de cruza", "float", 0.9,
                minimum=0.0, maximum=1.0,
                help="Probabilidad de cruzar una pareja de padres.",
            ),
            ParamSpec(
                "mutation_rate", "Tasa de mutación", "float", 0.12,
                minimum=0.0, maximum=1.0,
                default_by_family={"binary": 0.04, "permutation": 0.2},
                help="Probabilidad de mutar cada gen o posición.",
            ),
            ParamSpec(
                "mutation_strength", "Intensidad de mutación", "float", 0.15,
                minimum=0.0, maximum=1.0,
                families=frozenset({"continuous"}),
                help="Desviación gaussiana como fracción del rango [lb, ub].",
            ),
            ParamSpec(
                "blx_alpha", "Alpha de BLX", "float", 0.5, minimum=0.0,
                maximum=1.0, families=frozenset({"continuous"}),
                help="Ampliación del intervalo de mezcla en BLX-alfa.",
            ),
            ParamSpec(
                "elitism", "Número de élites", "int", 2, minimum=0,
                maximum=50, help="Mejores individuos que pasan intactos.",
            ),
        )

    # -- Inicialización ----------------------------------------------------

    def initialize(self) -> None:
        self.population = np.array(
            [self.problem.random_solution(self.rng) for _ in range(self.params["population_size"])]
        )
        crossover = self.params["crossover"]
        if crossover == "auto":
            crossover = {
                "continuous": "BLX-alfa",
                "binary": "1-punto",
                "permutation": "OX",
            }[self.family]
        if self.family == "continuous" and crossover == "OX":
            raise ValueError("La cruza OX solo aplica a permutaciones.")
        if self.family == "permutation" and crossover != "OX":
            raise ValueError("Las permutaciones requieren cruza OX.")
        if self.family == "binary" and crossover == "BLX-alfa":
            raise ValueError("BLX-alfa solo aplica a variables continuas.")
        self.crossover = crossover

    # -- Selección ---------------------------------------------------------

    def _tournament(self) -> int:
        k = min(self.params["tournament_size"], len(self.population))
        candidates = self.rng.choice(len(self.population), size=k, replace=False)
        values = [(self.fitness[i], i) for i in candidates]
        if self.problem.sense == "min":
            return int(min(values)[1])
        return int(max(values)[1])

    def _roulette(self) -> int:
        fitness = self.fitness.astype(float)
        if self.problem.sense == "min":
            weights = float(np.max(fitness)) - fitness + 1e-12
        else:
            weights = fitness - float(np.min(fitness)) + 1e-12
        total = float(np.sum(weights))
        if total <= 0 or not np.isfinite(total):
            return int(self.rng.integers(0, len(self.population)))
        probabilities = weights / total
        return int(self.rng.choice(len(self.population), p=probabilities))

    def _select(self) -> int:
        return self._tournament() if self.params["selection"] == "torneo" else self._roulette()

    # -- Cruza y mutación --------------------------------------------------

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        if self.family == "permutation":
            return ox_crossover(parent1, parent2, self.rng)

        if self.family == "binary" or self.crossover == "1-punto":
            n = parent1.size
            cut = int(self.rng.integers(1, n)) if n > 1 else 1
            return np.concatenate([parent1[:cut], parent2[cut:]])

        # BLX-alfa para variables continuas.
        alpha = self.params["blx_alpha"]
        low = np.minimum(parent1, parent2)
        high = np.maximum(parent1, parent2)
        spread = high - low
        child = self.rng.uniform(low - alpha * spread, high + alpha * spread)
        return self.problem.repair(child)

    def _mutate(self, child: np.ndarray) -> np.ndarray:
        rate = self.params["mutation_rate"]
        if self.family == "continuous":
            child = np.asarray(child, dtype=float).copy()
            mask = self.rng.random(child.size) < rate
            if np.any(mask):
                sigma = self.params["mutation_strength"] * (self.problem.ub - self.problem.lb)
                child[mask] += self.rng.normal(0.0, sigma, size=int(np.sum(mask)))
            return self.problem.repair(child)

        if self.family == "binary":
            child = np.asarray(child, dtype=np.int8).copy()
            mask = self.rng.random(child.size) < rate
            child[mask] ^= 1
            return child

        return swap_mutation(child, rate, self.rng)

    # -- Generación --------------------------------------------------------

    def step(self) -> None:
        population = self.population
        size = len(population)
        elite_count = min(self.params["elitism"], size)
        order = self.sorted_indices()

        new_population: list[np.ndarray] = [
            np.array(population[i], copy=True) for i in order[:elite_count]
        ]
        new_fitness: list[float] = [float(self.fitness[i]) for i in order[:elite_count]]

        while len(new_population) < size:
            parent1 = population[self._select()]
            parent2 = population[self._select()]
            if self.rng.random() < self.params["crossover_rate"]:
                child = self._crossover(parent1, parent2)
            else:
                child = np.array(parent1, copy=True)
            child = self._mutate(child)
            # Defensa adicional: toda solución debe ser válida.
            child = self.problem.repair(child)
            new_population.append(child)
            new_fitness.append(self.evaluate_individual(child))

        self.population = np.array(new_population)
        self.fitness = np.array(new_fitness, dtype=float)
