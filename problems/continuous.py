"""Problemas de optimización continua: Sphere y Rastrigin."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.base import ContinuousProblem, ParamSpec


class SphereProblem(ContinuousProblem):
    """Esfera: f(x) = sum(x_i^2). Óptimo global en 0 con f=0."""

    key = "sphere"
    name = "Sphere"
    description = "Función convexa y unimodal; ideal para validar convergencia."

    @classmethod
    def param_schema(cls) -> tuple[ParamSpec, ...]:
        return (
            ParamSpec(
                "dim", "Dimensión", "int", 10, minimum=2, maximum=100,
                help="Número de variables reales de la solución.",
            ),
            ParamSpec(
                "lb", "Límite inferior", "float", -5.12, minimum=-1e6, maximum=1e6,
                help="Cota inferior de cada coordenada.",
            ),
            ParamSpec(
                "ub", "Límite superior", "float", 5.12, minimum=-1e6, maximum=1e6,
                help="Cota superior de cada coordenada.",
            ),
        )

    def evaluate(self, solution: Any) -> float:
        x = np.asarray(solution, dtype=float)
        return float(np.sum(x * x))


class RastriginProblem(ContinuousProblem):
    """Rastrigin: multimodal con muchos mínimos locales.

    f(x) = 10n + sum(x_i^2 - 10 cos(2 pi x_i)); óptimo global en 0.
    """

    key = "rastrigin"
    name = "Rastrigin"
    description = "Función multimodal con numerosos mínimos locales."

    @classmethod
    def param_schema(cls) -> tuple[ParamSpec, ...]:
        return (
            ParamSpec(
                "dim", "Dimensión", "int", 10, minimum=2, maximum=100,
                help="Número de variables reales de la solución.",
            ),
            ParamSpec(
                "lb", "Límite inferior", "float", -5.12, minimum=-1e6, maximum=1e6,
                help="Cota inferior de cada coordenada.",
            ),
            ParamSpec(
                "ub", "Límite superior", "float", 5.12, minimum=-1e6, maximum=1e6,
                help="Cota superior de cada coordenada.",
            ),
        )

    def evaluate(self, solution: Any) -> float:
        x = np.asarray(solution, dtype=float)
        n = x.size
        return float(10.0 * n + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))
