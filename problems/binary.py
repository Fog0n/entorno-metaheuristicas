"""Problemas binarios: OneMax y Mochila 0-1."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.base import BinaryProblem, ParamSpec


class OneMaxProblem(BinaryProblem):
    """OneMax: maximizar el número de unos en una cadena binaria."""

    key = "onemax"
    name = "OneMax"
    sense = "max"
    description = "Maximiza el número de bits en 1; óptimo conocido: n."

    @classmethod
    def param_schema(cls) -> tuple[ParamSpec, ...]:
        return (
            ParamSpec(
                "n", "Longitud de la cadena", "int", 40, minimum=5, maximum=500,
                help="Número de bits de cada solución.",
            ),
        )

    def evaluate(self, solution: Any) -> float:
        return float(np.sum(np.asarray(solution, dtype=int)))

    def describe(self, solution: Any) -> dict[str, Any]:
        bits = np.asarray(solution, dtype=int)
        ones = int(np.sum(bits))
        return {
            "Bits en 1": f"{ones} / {bits.size}",
            "Porcentaje": f"{100.0 * ones / bits.size:.1f} %",
            "Cadena": "".join(str(int(b)) for b in bits),
        }


class KnapsackProblem(BinaryProblem):
    """Mochila 0-1 con penalización por exceso de capacidad.

    Los artículos se generan de forma determinista a partir de ``n`` para
    que una misma configuración produzca siempre la misma instancia.
    La función objetivo es:

        valor_total - penalización * max(0, peso_total - capacidad)
    """

    key = "knapsack"
    name = "Mochila 0-1"
    sense = "max"
    description = "Selecciona artículos maximizando valor con penalización por peso."

    def __init__(
        self,
        n: int = 30,
        capacity: float = 180.0,
        penalty: float = 25.0,
        **extra: Any,
    ) -> None:
        super().__init__(n=n, **extra)
        if capacity <= 0:
            raise ValueError("La capacidad debe ser positiva.")
        if penalty < 0:
            raise ValueError("La penalización no puede ser negativa.")
        self.capacity = float(capacity)
        self.penalty = float(penalty)

        # Instancia reproducible: no depende del RNG de la corrida.
        rng = np.random.default_rng(2024 + self.n)
        self.weights = rng.integers(3, 21, size=self.n).astype(float)
        self.values = rng.integers(5, 36, size=self.n).astype(float)
        self.names = [f"Artículo {i + 1:02d}" for i in range(self.n)]

    @classmethod
    def param_schema(cls) -> tuple[ParamSpec, ...]:
        return (
            ParamSpec(
                "n", "Número de artículos", "int", 30, minimum=5, maximum=200,
                help="Tamaño de la instancia generada de forma reproducible.",
            ),
            ParamSpec(
                "capacity", "Capacidad (kg/unidades)", "float", 180.0,
                minimum=1.0, maximum=1e6,
                help="Peso máximo sin penalización.",
            ),
            ParamSpec(
                "penalty", "Penalización por unidad excedida", "float", 25.0,
                minimum=0.0, maximum=1e6,
                help="Cuánto se resta al valor por cada unidad de sobrepeso.",
            ),
        )

    def evaluate(self, solution: Any) -> float:
        bits = np.asarray(solution, dtype=int)
        total_weight = float(np.dot(bits, self.weights))
        total_value = float(np.dot(bits, self.values))
        excess = max(0.0, total_weight - self.capacity)
        return total_value - self.penalty * excess

    def describe(self, solution: Any) -> dict[str, Any]:
        bits = np.asarray(solution, dtype=int)
        weight = float(np.dot(bits, self.weights))
        value = float(np.dot(bits, self.values))
        selected = int(np.sum(bits))
        excess = max(0.0, weight - self.capacity)
        return {
            "Valor total": f"{value:.1f}",
            "Peso total": f"{weight:.1f} / {self.capacity:.1f}",
            "Artículos": f"{selected} / {self.n}",
            "Sobrepeso": f"{excess:.1f}",
            "Penalización": f"{self.penalty * excess:.1f}",
            "Seleccionados": ", ".join(
                self.names[i] for i in np.flatnonzero(bits)
            ) or "(ninguno)",
        }
