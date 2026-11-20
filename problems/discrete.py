"""Problema del agente viajero (TSP) con ciudades en el plano."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.base import ParamSpec, PermutationProblem


class TSPProblem(PermutationProblem):
    """TSP simétrico: recorrer todas las ciudades y volver al origen.

    Las coordenadas se generan de forma determinista a partir de
    ``coord_seed`` para que el escenario sea reproducible.
    """

    key = "tsp"
    name = "TSP (agente viajero)"
    sense = "min"
    description = "Minimiza la longitud de un recorrido cerrado por n ciudades."

    def __init__(
        self,
        n: int = 20,
        coord_seed: int = 7,
        area: float = 100.0,
        **extra: Any,
    ) -> None:
        super().__init__(n=n, **extra)
        if area <= 0:
            raise ValueError("El área debe ser positiva.")
        self.coord_seed = int(coord_seed)
        self.area = float(area)
        rng = np.random.default_rng(self.coord_seed)
        self.coordinates = rng.uniform(0.0, self.area, size=(self.n, 2))
        # Matriz de distancias euclidianas precalculada.
        diff = self.coordinates[:, None, :] - self.coordinates[None, :, :]
        self._distances = np.sqrt(np.sum(diff * diff, axis=2))

    @classmethod
    def param_schema(cls) -> tuple[ParamSpec, ...]:
        return (
            ParamSpec(
                "n", "Número de ciudades", "int", 20, minimum=5, maximum=120,
                help="Ciudades del recorrido; se generan de forma reproducible.",
            ),
            ParamSpec(
                "coord_seed", "Semilla de coordenadas", "int", 7, minimum=0,
                maximum=10_000,
                help="Cambia el mapa sin cambiar la semilla de la corrida.",
            ),
            ParamSpec(
                "area", "Tamaño del mapa", "float", 100.0, minimum=1.0,
                maximum=10_000.0,
                help="Lado del cuadrado donde se ubican las ciudades.",
            ),
        )

    def evaluate(self, solution: Any) -> float:
        tour = np.asarray(solution, dtype=int)
        nxt = np.roll(tour, -1)
        return float(np.sum(self._distances[tour, nxt]))

    def describe(self, solution: Any) -> dict[str, Any]:
        tour = np.asarray(solution, dtype=int)
        length = self.evaluate(tour)
        return {
            "Distancia total": f"{length:.2f}",
            "Ciudades": str(self.n),
            "Recorrido": " → ".join(f"C{i + 1:02d}" for i in tour),
        }

    # -- Ganchos para ACO --------------------------------------------------

    def aco_distance_matrix(self) -> np.ndarray:
        return self._distances

    def aco_initial_weights(self) -> np.ndarray:
        return np.ones(self.n, dtype=float)

    def aco_edges(self, solution: Any) -> list[tuple[int, int]]:
        tour = np.asarray(solution, dtype=int)
        return [
            (int(tour[i]), int(tour[(i + 1) % self.n]))
            for i in range(self.n)
        ]
