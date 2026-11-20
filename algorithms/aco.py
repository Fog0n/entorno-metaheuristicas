"""Colonia de Hormigas — Ant System (ACO) para problemas de permutación.

Cada hormiga construye una solución guiada por la feromona ``tau`` y la
visibilidad ``eta = 1/distancia``. Después se evapora la feromona y cada
hormiga deposita una cantidad inversamente proporcional al costo de su
solución. Se implementa desde cero, sin librerías externas.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.base import MetaheuristicAlgorithm, ParamSpec


class AntColonyAlgorithm(MetaheuristicAlgorithm):
    key = "aco"
    name = "Colonia de Hormigas (Ant System)"
    family_support = frozenset({"permutation"})

    @classmethod
    def param_schema(cls, family: str | None = None) -> tuple[ParamSpec, ...]:
        return (
            ParamSpec(
                "n_ants", "Número de hormigas", "int", 30, minimum=2,
                maximum=500, help="Soluciones construidas por iteración.",
            ),
            ParamSpec(
                "alpha", "Alpha (peso de feromona)", "float", 1.0, minimum=0.0,
                maximum=10.0, help="Importancia de tau en la probabilidad.",
            ),
            ParamSpec(
                "beta", "Beta (peso de visibilidad)", "float", 3.0, minimum=0.0,
                maximum=10.0, help="Importancia de 1/distancia.",
            ),
            ParamSpec(
                "rho", "Evaporación rho", "float", 0.1, minimum=0.0,
                maximum=1.0, help="Fracción de feromona que se evapora.",
            ),
            ParamSpec(
                "q", "Q (depósito)", "float", 100.0, minimum=0.001,
                maximum=1e6, help="Constante del depósito Q/costo.",
            ),
            ParamSpec(
                "tau_init", "Feromona inicial", "float", 1.0, minimum=1e-6,
                maximum=1e6, help="Valor inicial de todas las aristas.",
            ),
        )

    def initialize(self) -> None:
        n = self.problem.n
        self.tau = np.full((n, n), float(self.params["tau_init"]))
        np.fill_diagonal(self.tau, 0.0)
        distances = self.problem.aco_distance_matrix()
        self.eta = 1.0 / (np.asarray(distances, dtype=float) + 1e-9)
        np.fill_diagonal(self.eta, 0.0)
        self._initial_weights = np.asarray(self.problem.aco_initial_weights(), dtype=float)
        self._initial_weights = self._initial_weights / np.sum(self._initial_weights)

        # Población inicial aleatoria para que la plantilla base tenga datos.
        self.population = np.array(
            [self.problem.random_solution(self.rng) for _ in range(self.params["n_ants"])]
        )

    # -- Construcción de una solución -------------------------------------

    def _construct(self) -> np.ndarray:
        n = self.problem.n
        alpha = self.params["alpha"]
        beta = self.params["beta"]

        # Primera ciudad: en TSP uniforme; en reparto, cercanía a la base.
        probabilities = np.power(self._initial_weights, max(beta, 1e-9))
        probabilities = probabilities / np.sum(probabilities)
        start = int(self.rng.choice(n, p=probabilities))

        tour = [start]
        visited = np.zeros(n, dtype=bool)
        visited[start] = True
        current = start

        for _ in range(n - 1):
            candidates = np.flatnonzero(~visited)
            pheromone = np.power(self.tau[current, candidates], alpha)
            visibility = np.power(self.eta[current, candidates], beta)
            weights = pheromone * visibility
            total = float(np.sum(weights))
            if total <= 0 or not np.isfinite(total):
                weights = np.ones(len(candidates), dtype=float)
                total = float(len(candidates))
            next_node = int(self.rng.choice(candidates, p=weights / total))
            tour.append(next_node)
            visited[next_node] = True
            current = next_node

        return np.array(tour, dtype=int)

    # -- Iteración ---------------------------------------------------------

    def step(self) -> None:
        n_ants = self.params["n_ants"]
        rho = self.params["rho"]
        q = self.params["q"]

        ants = np.array([self._construct() for _ in range(n_ants)])
        fitness = np.array([self.evaluate_individual(ant) for ant in ants], dtype=float)

        # Evaporación.
        self.tau *= 1.0 - rho
        np.clip(self.tau, 1e-12, None, out=self.tau)
        np.fill_diagonal(self.tau, 0.0)

        # Depósito proporcional a la calidad: Q / costo.
        for ant, cost in zip(ants, fitness):
            if not np.isfinite(cost) or cost <= 0:
                continue
            deposit = q / cost
            for a, b in self.problem.aco_edges(ant):
                self.tau[a, b] += deposit
                self.tau[b, a] += deposit

        self.population = ants
        self.fitness = fitness
