"""Problema real: rutas de reparto locales en Zapopan.

Se carga un dataset de 12 nodos (1 base CEDIS + 11 puntos de entrega).
Las distancias viales se aproximan con la fórmula de Haversine multiplicada
por un factor de rodeo (1.25). La carga de cada vehículo no puede superar
``Q_max``; una permutación de clientes se divide en rutas factibles y se
minimiza la distancia total más un costo fijo por ruta.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from core.base import ParamSpec, PermutationProblem


class DeliveryZapopanProblem(PermutationProblem):
    """Rutas de reparto con capacidad desde una base en Zapopan."""

    key = "delivery_zapopan"
    name = "Rutas de reparto Zapopan"
    sense = "min"
    description = (
        "Minimiza distancia y número de rutas desde el CEDIS, "
        "respetando una capacidad de 180 kg por vehículo."
    )

    def __init__(
        self,
        q_max: float = 180.0,
        road_factor: float = 1.25,
        fixed_cost: float = 8.0,
        **extra: Any,
    ) -> None:
        dataset_path = Path(__file__).resolve().parents[1] / "data" / "rutas_reparto_zapopan.json"
        if not dataset_path.exists():
            raise FileNotFoundError(f"No se encontró el dataset: {dataset_path}")
        with dataset_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        nodes = data["nodos"]
        super().__init__(n=len(nodes), **extra)
        if q_max <= 0:
            raise ValueError("La capacidad Q_max debe ser positiva.")
        self.q_max = float(q_max)
        self.road_factor = float(road_factor)
        self.fixed_cost = float(fixed_cost)
        self.dataset_path = dataset_path
        self.dataset = data
        self.depot = data["deposito"]
        self.nodes = nodes
        self.demands = np.array([float(n["demanda_kg"]) for n in nodes])
        self.names = [str(n["nombre"]) for n in nodes]
        self.lats = np.array([float(n["lat"]) for n in nodes])
        self.lons = np.array([float(n["lon"]) for n in nodes])

        if float(np.max(self.demands)) > self.q_max:
            raise ValueError("Hay una demanda individual mayor que Q_max.")

        # Matrices viales aproximadas (km).
        self._client_distances = self._haversine_matrix(self.lats, self.lons)
        self._depot_distances = self._haversine(
            np.full(self.n, float(self.depot["lat"])),
            np.full(self.n, float(self.depot["lon"])),
            self.lats,
            self.lons,
        )
        self._client_distances *= self.road_factor
        self._depot_distances *= self.road_factor
        np.fill_diagonal(self._client_distances, 0.0)

    # -- Distancias --------------------------------------------------------

    @staticmethod
    def _haversine(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> np.ndarray:
        """Distancia esférica en km entre pares de coordenadas."""
        lat1 = np.radians(np.asarray(lat1, dtype=float))
        lon1 = np.radians(np.asarray(lon1, dtype=float))
        lat2 = np.radians(np.asarray(lat2, dtype=float))
        lon2 = np.radians(np.asarray(lon2, dtype=float))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    @classmethod
    def _haversine_matrix(cls, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        lat1 = np.radians(lats)[:, None]
        lon1 = np.radians(lons)[:, None]
        lat2 = np.radians(lats)[None, :]
        lon2 = np.radians(lons)[None, :]
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    # -- Configuración -----------------------------------------------------

    @classmethod
    def param_schema(cls) -> tuple[ParamSpec, ...]:
        return (
            ParamSpec(
                "q_max", "Capacidad Q_max (kg)", "float", 180.0,
                minimum=10.0, maximum=10_000.0,
                help="Carga máxima por vehículo.",
            ),
            ParamSpec(
                "road_factor", "Factor vial", "float", 1.25,
                minimum=1.0, maximum=3.0,
                help="Multiplicador sobre la distancia Haversine.",
            ),
            ParamSpec(
                "fixed_cost", "Costo fijo por ruta (km eq.)", "float", 8.0,
                minimum=0.0, maximum=500.0,
                help="Penaliza abrir vehículos adicionales.",
            ),
        )

    # -- Decodificación y evaluación --------------------------------------

    def decode_routes(self, solution: Any) -> list[dict[str, Any]]:
        """Divide la permutación en rutas factibles por capacidad."""
        perm = np.asarray(solution, dtype=int)
        routes: list[list[int]] = []
        current: list[int] = []
        load = 0.0
        for idx in perm:
            demand = float(self.demands[idx])
            if current and load + demand > self.q_max + 1e-9:
                routes.append(current)
                current = []
                load = 0.0
            current.append(int(idx))
            load += demand
        if current:
            routes.append(current)

        result = []
        for route in routes:
            distance = float(self._depot_distances[route[0]])
            for a, b in zip(route, route[1:]):
                distance += float(self._client_distances[a, b])
            distance += float(self._depot_distances[route[-1]])
            result.append(
                {
                    "nodos": route,
                    "nombres": [self.names[i] for i in route],
                    "carga": float(np.sum(self.demands[route])),
                    "distancia": distance,
                }
            )
        return result

    def evaluate(self, solution: Any) -> float:
        routes = self.decode_routes(solution)
        distance = sum(route["distancia"] for route in routes)
        return float(distance + self.fixed_cost * len(routes))

    def validate(self, solution: Any) -> tuple[bool, str]:
        valid, message = super().validate(solution)
        if not valid:
            return valid, message
        if float(np.max(self.demands)) > self.q_max:
            return False, "Existe una demanda individual mayor que Q_max."
        return True, "Permutación válida y demandas acotadas por Q_max."

    def describe(self, solution: Any) -> dict[str, Any]:
        routes = self.decode_routes(solution)
        total_distance = sum(route["distancia"] for route in routes)
        details = []
        for i, route in enumerate(routes, start=1):
            details.append(
                f"Ruta {i}: {' → '.join(route['nombres'])} "
                f"({route['carga']:.0f} kg, {route['distancia']:.1f} km)"
            )
        return {
            "Rutas": str(len(routes)),
            "Distancia de rutas": f"{total_distance:.1f} km",
            "Costo fijo": f"{self.fixed_cost * len(routes):.1f} km eq.",
            "Carga total": f"{float(np.sum(self.demands)):.0f} kg",
            "Detalle": "\n".join(details),
        }

    def route_points(self, solution: Any) -> list[list[tuple[float, float]]]:
        """Polilíneas (lon, lat) por ruta, incluyendo la base al inicio y fin."""
        depot = (float(self.depot["lon"]), float(self.depot["lat"]))
        lines = []
        for route in self.decode_routes(solution):
            line = [depot]
            for idx in route["nodos"]:
                line.append((float(self.lons[idx]), float(self.lats[idx])))
            line.append(depot)
            lines.append(line)
        return lines

    # -- Ganchos para ACO --------------------------------------------------

    def aco_distance_matrix(self) -> np.ndarray:
        return self._client_distances

    def aco_initial_weights(self) -> np.ndarray:
        # Al inicio conviene visitar primero lo más cercano a la base.
        return 1.0 / (self._depot_distances + 1e-6)

    def aco_edges(self, solution: Any) -> list[tuple[int, int]]:
        edges: list[tuple[int, int]] = []
        for route in self.decode_routes(solution):
            nodes = route["nodos"]
            edges.extend((int(a), int(b)) for a, b in zip(nodes, nodes[1:]))
        return edges

    def plot_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordenadas (lon, lat) de la base y de los clientes."""
        depot = np.array([[float(self.depot["lon"]), float(self.depot["lat"])]])
        return depot, np.column_stack((self.lons, self.lats))
