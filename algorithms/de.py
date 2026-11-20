"""Evolución Diferencial — DE/rand/1/bin.

Variantes:
- continua: mutación diferencial y cruza binomial con selección voraz;
- binaria: la prueba continua se transforma con sigmoide para generar bits;
- permutaciones: representación Random-Key y decodificación por orden.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.base import (
    MetaheuristicAlgorithm,
    ParamSpec,
    random_keys_to_permutation,
    sigmoid,
)


class DifferentialEvolutionAlgorithm(MetaheuristicAlgorithm):
    key = "de"
    name = "Evolución Diferencial (DE/rand/1/bin)"

    @classmethod
    def param_schema(cls, family: str | None = None) -> tuple[ParamSpec, ...]:
        return (
            ParamSpec(
                "population_size", "Tamaño de población", "int", 50, minimum=4,
                maximum=1000, help="Individuos; debe ser al menos 4.",
            ),
            ParamSpec(
                "f", "Factor de mutación F", "float", 0.5, minimum=0.0,
                maximum=2.0, help="Amplitud del diferencial entre individuos.",
            ),
            ParamSpec(
                "cr", "Probabilidad de cruza CR", "float", 0.9, minimum=0.0,
                maximum=1.0, help="Probabilidad de tomar el valor mutante.",
            ),
        )

    def native(self, individual: Any) -> Any:
        if self.family == "permutation":
            return random_keys_to_permutation(individual)
        return individual

    def initialize(self) -> None:
        size = self.params["population_size"]
        n = self.problem.dimension
        if self.family == "continuous":
            self.population = np.array(
                [self.problem.random_solution(self.rng) for _ in range(size)]
            )
        elif self.family == "binary":
            self.population = np.array(
                [self.problem.random_solution(self.rng) for _ in range(size)]
            ).astype(np.int8)
        else:
            self.population = self.rng.random((size, n))

        self.fitness = np.array(
            [self.evaluate_individual(ind) for ind in self.population], dtype=float
        )

    def _trial(self, index: int) -> Any:
        size = len(self.population)
        choices = [j for j in range(size) if j != index]
        r1, r2, r3 = self.rng.choice(choices, size=3, replace=False)
        f = self.params["f"]
        cr = self.params["cr"]

        if self.family == "binary":
            mutant = self.population[r1].astype(float) + f * (
                self.population[r2].astype(float) - self.population[r3].astype(float)
            )
            mask = self.rng.random(self.problem.n) < cr
            trial = np.where(mask, mutant, self.population[index].astype(float))
            # Binarización estocástica mediante sigmoide.
            return (self.rng.random(self.problem.n) < sigmoid(trial)).astype(np.int8)

        mutant = self.population[r1] + f * (self.population[r2] - self.population[r3])
        mask = self.rng.random(self.problem.dimension) < cr
        trial = np.where(mask, mutant, self.population[index])
        if self.family == "continuous":
            return self.problem.repair(trial)
        return np.clip(trial, 0.0, 1.0)

    def step(self) -> None:
        for i in range(len(self.population)):
            trial = self._trial(i)
            value = self.evaluate_individual(trial)
            if self.problem.is_better(value, self.fitness[i]):
                self.population[i] = trial
                self.fitness[i] = value
