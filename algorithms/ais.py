"""Sistema Inmune Artificial — CLONALG.

Implementa selección clonal, expansión de clones, hipermutación inversa a
la afinidad (los peores anticuerpos mutan más), reemplazo de los peores y
un pequeño aporte de anticuerpos aleatorios para mantener diversidad.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.base import MetaheuristicAlgorithm, ParamSpec, swap_mutation


class ClonalSelectionAlgorithm(MetaheuristicAlgorithm):
    key = "ais"
    name = "Sistema Inmune (CLONALG)"

    @classmethod
    def param_schema(cls, family: str | None = None) -> tuple[ParamSpec, ...]:
        return (
            ParamSpec(
                "population_size", "Tamaño de población", "int", 50, minimum=4,
                maximum=1000, help="Anticuerpos mantenidos por iteración.",
            ),
            ParamSpec(
                "n_selected", "Anticuerpos seleccionados", "int", 15, minimum=1,
                maximum=200, help="Mejores candidatos que se clonan.",
            ),
            ParamSpec(
                "clones_per_antibody", "Clones por anticuerpo", "int", 5,
                minimum=1, maximum=30, help="Expansión clonal de cada seleccionado.",
            ),
            ParamSpec(
                "mutation_rate", "Tasa de hipermutación", "float", 0.15,
                minimum=0.0, maximum=1.0,
                default_by_family={"binary": 0.08, "permutation": 0.25},
                help="Probabilidad base de mutar cada gen.",
            ),
            ParamSpec(
                "mutation_strength", "Intensidad de mutación", "float", 0.2,
                minimum=0.0, maximum=1.0, families=frozenset({"continuous"}),
                help="Desviación gaussiana como fracción del rango.",
            ),
            ParamSpec(
                "replacement_rate", "Tasa de reemplazo", "float", 0.2,
                minimum=0.0, maximum=1.0,
                help="Fracción de peores anticuerpos sustituidos al azar.",
            ),
            ParamSpec(
                "elite_count", "Élites conservados", "int", 2, minimum=0,
                maximum=50, help="Mejores anticuerpos que pasan intactos.",
            ),
        )

    # -- Inicialización ----------------------------------------------------

    def initialize(self) -> None:
        self.population = np.array(
            [self.problem.random_solution(self.rng) for _ in range(self.params["population_size"])]
        )
        self.fitness = np.array(
            [self.evaluate_individual(ind) for ind in self.population], dtype=float
        )

    # -- Hipermutación -----------------------------------------------------

    def _mutate(self, antibody: np.ndarray, factor: float) -> np.ndarray:
        """Muta un clon; ``factor`` crece para los anticuerpos menos afines."""
        rate = float(np.clip(self.params["mutation_rate"] * factor, 0.0, 1.0))

        if self.family == "continuous":
            child = np.array(antibody, dtype=float, copy=True)
            mask = self.rng.random(child.size) < rate
            if np.any(mask):
                sigma = (
                    self.params["mutation_strength"]
                    * (self.problem.ub - self.problem.lb)
                    * factor
                )
                child[mask] += self.rng.normal(0.0, sigma, size=int(np.sum(mask)))
            return self.problem.repair(child)

        if self.family == "binary":
            child = np.array(antibody, dtype=np.int8, copy=True)
            mask = self.rng.random(child.size) < rate
            child[mask] ^= 1
            return child

        return swap_mutation(antibody, rate, self.rng)

    # -- Iteración ---------------------------------------------------------

    def step(self) -> None:
        size = self.params["population_size"]
        order = self.sorted_indices()
        elite_count = min(self.params["elite_count"], size)
        n_selected = min(self.params["n_selected"], size)
        clones_per = self.params["clones_per_antibody"]

        pool: list[np.ndarray] = [
            np.array(self.population[i], copy=True) for i in order[:elite_count]
        ]
        pool_fitness: list[float] = [float(self.fitness[i]) for i in order[:elite_count]]

        # Expansión clonal de los mejores; peor rango -> mayor mutación.
        for rank in range(n_selected):
            idx = int(order[rank])
            affinity_factor = 0.3 + 0.7 * (rank / max(1, n_selected - 1))
            for _ in range(clones_per):
                clone = self._mutate(self.population[idx], affinity_factor)
                if self.family == "permutation":
                    clone = self.problem.repair(clone)
                pool.append(clone)
                pool_fitness.append(self.evaluate_individual(clone))

        # Completar con anticuerpos aleatorios si faltan individuos.
        while len(pool) < size:
            candidate = self.problem.random_solution(self.rng)
            pool.append(candidate)
            pool_fitness.append(self.evaluate_individual(candidate))

        # Selección de los mejores del pool (incluye élites y clones).
        pool_fitness_array = np.asarray(pool_fitness, dtype=float)
        if self.problem.sense == "min":
            best = np.argsort(pool_fitness_array)[:size]
        else:
            best = np.argsort(-pool_fitness_array)[:size]

        new_population = np.array([pool[int(i)] for i in best])
        new_fitness = pool_fitness_array[best]

        # Reemplazo: los peores se sustituyen por anticuerpos nuevos.
        replacement = int(round(self.params["replacement_rate"] * size))
        if replacement > 0:
            # Nunca se reemplazan los élites, para no perder la mejor solución.
            protected = set(int(i) for i in best[:elite_count])
            candidates = np.array([i for i in range(size) if i not in protected], dtype=int)
            if candidates.size:
                replacement = min(replacement, candidates.size)
                worst_order = (
                    candidates[np.argsort(new_fitness[candidates])]
                    if self.problem.sense == "min"
                    else candidates[np.argsort(-new_fitness[candidates])]
                )
                for j in worst_order[-replacement:]:
                    candidate = self.problem.random_solution(self.rng)
                    new_population[int(j)] = candidate
                    new_fitness[int(j)] = self.evaluate_individual(candidate)

        self.population = new_population
        self.fitness = np.asarray(new_fitness, dtype=float)
