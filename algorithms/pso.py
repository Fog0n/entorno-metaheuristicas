"""Optimización por Enjambre de Partículas (PSO).

Variantes implementadas:
- continua: PSO clásico con inercia w, coeficientes c1/c2 y v_max;
- binaria: BPSO, donde la velocidad se transforma con una sigmoide para
  obtener la probabilidad de cada bit;
- permutaciones: Random-Key, cada partícula es un vector continuo que se
  decodifica ordenando sus claves.
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


class ParticleSwarmAlgorithm(MetaheuristicAlgorithm):
    key = "pso"
    name = "Enjambre de Partículas (PSO)"

    @classmethod
    def param_schema(cls, family: str | None = None) -> tuple[ParamSpec, ...]:
        return (
            ParamSpec(
                "swarm_size", "Tamaño del enjambre", "int", 40, minimum=2,
                maximum=1000, help="Número de partículas.",
            ),
            ParamSpec(
                "w", "Inercia w", "float", 0.72, minimum=0.0, maximum=2.0,
                help="Peso de la velocidad anterior.",
            ),
            ParamSpec(
                "c1", "Coeficiente c1", "float", 1.6, minimum=0.0, maximum=5.0,
                help="Atracción hacia el mejor personal.",
            ),
            ParamSpec(
                "c2", "Coeficiente c2", "float", 1.6, minimum=0.0, maximum=5.0,
                help="Atracción hacia el mejor global.",
            ),
            ParamSpec(
                "v_max", "Velocidad máxima", "float", 0.2, minimum=0.001,
                maximum=100.0,
                default_by_family={"binary": 6.0, "continuous": 0.2, "permutation": 0.2},
                help=(
                    "Continua: fracción del rango. Binaria: límite de velocidad. "
                    "Permutación: límite en el espacio de claves."
                ),
            ),
        )

    def native(self, individual: Any) -> Any:
        if self.family == "permutation":
            return random_keys_to_permutation(individual)
        return individual

    # -- Inicialización ----------------------------------------------------

    def initialize(self) -> None:
        size = self.params["swarm_size"]
        n = self.problem.dimension

        if self.family == "continuous":
            self.population = np.array(
                [self.problem.random_solution(self.rng) for _ in range(size)]
            )
            span = self.problem.ub - self.problem.lb
            limit = self.params["v_max"] * span
            self.velocity = self.rng.uniform(-limit, limit, size=(size, n))
        elif self.family == "binary":
            self.population = np.array(
                [self.problem.random_solution(self.rng) for _ in range(size)]
            ).astype(np.int8)
            self.velocity = self.rng.uniform(
                -self.params["v_max"], self.params["v_max"], size=(size, n)
            )
        else:
            # Espacio de claves en [0, 1] para Random-Key.
            self.population = self.rng.random((size, n))
            self.velocity = self.rng.uniform(
                -self.params["v_max"], self.params["v_max"], size=(size, n)
            )

        self.fitness = np.array(
            [self.evaluate_individual(ind) for ind in self.population], dtype=float
        )
        self.pbest = np.array(self.population, copy=True)
        self.pbest_fitness = self.fitness.copy()
        best_idx = self.problem.best_index(self.fitness)
        self.gbest = np.array(self.population[best_idx], copy=True)
        self.gbest_fitness = float(self.fitness[best_idx])

    # -- Iteración ---------------------------------------------------------

    def step(self) -> None:
        size = len(self.population)
        w = self.params["w"]
        c1 = self.params["c1"]
        c2 = self.params["c2"]
        vmax = self.params["v_max"]

        for i in range(size):
            r1 = self.rng.random(self.problem.dimension)
            r2 = self.rng.random(self.problem.dimension)
            self.velocity[i] = (
                w * self.velocity[i]
                + c1 * r1 * (self.pbest[i] - self.population[i])
                + c2 * r2 * (self.gbest - self.population[i])
            )

            if self.family == "continuous":
                span = self.problem.ub - self.problem.lb
                limit = vmax * span
                self.velocity[i] = np.clip(self.velocity[i], -limit, limit)
                position = self.population[i] + self.velocity[i]
                self.population[i] = self.problem.repair(position)
            elif self.family == "binary":
                self.velocity[i] = np.clip(self.velocity[i], -vmax, vmax)
                probability = sigmoid(self.velocity[i])
                self.population[i] = (self.rng.random(self.problem.n) < probability).astype(np.int8)
            else:
                self.velocity[i] = np.clip(self.velocity[i], -vmax, vmax)
                self.population[i] = np.clip(self.population[i] + self.velocity[i], 0.0, 1.0)

            value = self.evaluate_individual(self.population[i])
            self.fitness[i] = value
            if self.problem.is_better(value, self.pbest_fitness[i]):
                self.pbest_fitness[i] = value
                self.pbest[i] = np.array(self.population[i], copy=True)
            if self.problem.is_better(value, self.gbest_fitness):
                self.gbest_fitness = value
                self.gbest = np.array(self.population[i], copy=True)
