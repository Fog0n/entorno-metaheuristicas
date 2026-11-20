"""Núcleo del entorno de metaheurísticas.

Define los contratos que comparten todos los problemas y algoritmos, el
historial de snapshots que consume la interfaz y operadores genéticos
básicos. Este módulo no importa Tkinter ni Matplotlib: puede usarse desde
scripts headless y desde el motor en hilo secundario.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

MINIMIZE = "min"
MAXIMIZE = "max"


# ---------------------------------------------------------------------------
# Especificación de parámetros (alimenta los paneles dinámicos de la UI)
# ---------------------------------------------------------------------------


@dataclass
class ParamSpec:
    """Describe un parámetro configurable de un problema o algoritmo.

    La interfaz gráfica construye los formularios a partir de estas
    especificaciones, por lo que añadir un parámetro no exige tocar la UI.
    """

    key: str
    label: str
    kind: str  # "int", "float", "choice" o "bool"
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    help: str = ""
    families: frozenset[str] = frozenset()
    default_by_family: dict[str, Any] = field(default_factory=dict)
    choices_by_family: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def applies_to(self, family: str | None) -> bool:
        return not self.families or family in self.families

    def resolve_default(self, family: str | None = None) -> Any:
        if family is not None and family in self.default_by_family:
            return self.default_by_family[family]
        return self.default

    def resolve_choices(self, family: str | None = None) -> tuple[str, ...]:
        if family is not None and family in self.choices_by_family:
            return self.choices_by_family[family]
        return self.choices

    def validate(self, value: Any) -> Any:
        """Convierte y valida un valor, lanzando ValueError con un mensaje claro."""
        if self.kind == "int":
            try:
                result = int(round(float(value)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{self.label}: se esperaba un número entero.") from exc
            if self.minimum is not None and result < self.minimum:
                raise ValueError(f"{self.label}: el mínimo es {self.minimum}.")
            if self.maximum is not None and result > self.maximum:
                raise ValueError(f"{self.label}: el máximo es {self.maximum}.")
            return result

        if self.kind == "float":
            try:
                result = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{self.label}: se esperaba un número.") from exc
            if not np.isfinite(result):
                raise ValueError(f"{self.label}: el valor debe ser finito.")
            if self.minimum is not None and result < self.minimum:
                raise ValueError(f"{self.label}: el mínimo es {self.minimum}.")
            if self.maximum is not None and result > self.maximum:
                raise ValueError(f"{self.label}: el máximo es {self.maximum}.")
            return result

        if self.kind == "choice":
            result = str(value)
            options = self.resolve_choices()
            if options and result not in options:
                raise ValueError(
                    f"{self.label}: opción inválida ({result}). "
                    f"Opciones: {', '.join(options)}."
                )
            return result

        if self.kind == "bool":
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "sí", "si", "yes", "on"}
            return bool(value)

        raise ValueError(f"Tipo de parámetro desconocido: {self.kind}")


def defaults_for(schema: Sequence[ParamSpec], family: str | None = None) -> dict[str, Any]:
    """Devuelve los valores por defecto de un esquema para una familia."""
    return {
        spec.key: spec.resolve_default(family)
        for spec in schema
        if spec.applies_to(family)
    }


def validate_params(
    schema: Sequence[ParamSpec],
    values: dict[str, Any],
    family: str | None = None,
) -> dict[str, Any]:
    """Mezcla los valores recibidos con los defaults y valida cada campo."""
    merged = defaults_for(schema, family)
    for key, value in values.items():
        if key not in merged:
            raise ValueError(f"Parámetro desconocido: {key}")
        merged[key] = value
    for spec in schema:
        if spec.applies_to(family) and spec.key in merged:
            merged[spec.key] = spec.validate(merged[spec.key])
    return merged


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


@dataclass
class Snapshot:
    """Estado resumido de una iteración, apto para graficar y reproducir."""

    iteration: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    diversity: float
    best_solution: Any
    evaluations: int
    elapsed: float
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Problemas
# ---------------------------------------------------------------------------


class Problem(ABC):
    """Contrato base de un problema de optimización.

    ``sense`` indica si el objetivo se minimiza o se maximiza. Los
    algoritmos nunca comparan números directamente: usan ``is_better``.
    """

    key: str = ""
    name: str = ""
    family: str = "continuous"  # continuous | binary | permutation
    sense: str = MINIMIZE
    description: str = ""

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    # -- Configuración -----------------------------------------------------

    @classmethod
    def param_schema(cls) -> tuple[ParamSpec, ...]:
        return ()

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return defaults_for(cls.param_schema())

    # -- Interfaz obligatoria ---------------------------------------------

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Tamaño de la representación de una solución."""

    @abstractmethod
    def evaluate(self, solution: Any) -> float:
        """Evalúa una solución en la escala propia del problema."""

    def random_solution(self, rng: np.random.Generator) -> Any:
        raise NotImplementedError

    @abstractmethod
    def validate(self, solution: Any) -> tuple[bool, str]:
        """Comprueba que una solución sea válida para el problema."""

    # -- Utilidades --------------------------------------------------------

    def is_better(self, a: float, b: float) -> bool:
        if a is None or not np.isfinite(a):
            return False
        if b is None or not np.isfinite(b):
            return True
        return a < b if self.sense == MINIMIZE else a > b

    def best_index(self, fitness: Sequence[float]) -> int:
        values = np.asarray(fitness, dtype=float)
        return int(np.argmin(values) if self.sense == MINIMIZE else np.argmax(values))

    def decode(self, solution: Any) -> Any:
        """Convierte una solución interna en la representación para la UI."""
        return np.array(solution, copy=True)

    def diversity(self, population: Sequence[Any]) -> float:
        """Diversidad normalizada en [0, 1]; cada familia la redefine."""
        return 0.0

    def describe(self, solution: Any) -> dict[str, Any]:
        """Resumen legible de una solución para el panel de la interfaz."""
        return {}

    @property
    def objective_label(self) -> str:
        return "Costo (menor es mejor)" if self.sense == MINIMIZE else "Fitness (mayor es mejor)"

    def __repr__(self) -> str:  # pragma: no cover - ayuda de depuración
        return f"{self.__class__.__name__}({self.params})"


class ContinuousProblem(Problem):
    """Base para problemas de variables reales con caja [lb, ub]^dim."""

    family = "continuous"
    sense = MINIMIZE

    def __init__(self, dim: int = 10, lb: float = -5.12, ub: float = 5.12, **extra: Any) -> None:
        if dim < 1:
            raise ValueError("La dimensión debe ser al menos 1.")
        if not lb < ub:
            raise ValueError("Se requiere lb < ub.")
        self.dim = int(dim)
        self.lb = float(lb)
        self.ub = float(ub)
        super().__init__(dim=self.dim, lb=self.lb, ub=self.ub, **extra)

    @property
    def dimension(self) -> int:
        return self.dim

    def random_solution(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lb, self.ub, size=self.dim)

    def repair(self, solution: Sequence[float]) -> np.ndarray:
        return np.clip(np.asarray(solution, dtype=float), self.lb, self.ub)

    def validate(self, solution: Any) -> tuple[bool, str]:
        try:
            arr = np.asarray(solution, dtype=float)
        except (TypeError, ValueError):
            return False, "La solución no es numérica."
        if arr.shape != (self.dim,):
            return False, f"Se esperaban {self.dim} coordenadas y llegaron {arr.size}."
        if not np.all(np.isfinite(arr)):
            return False, "La solución contiene valores no finitos."
        if np.any(arr < self.lb - 1e-9) or np.any(arr > self.ub + 1e-9):
            return False, f"La solución sale de los límites [{self.lb}, {self.ub}]."
        return True, "Solución válida."

    def diversity(self, population: Sequence[Any]) -> float:
        arr = np.asarray(population, dtype=float)
        if arr.ndim != 2 or len(arr) < 2:
            return 0.0
        # Desviación media relativa al rango de cada dimensión.
        spread = float(np.mean(np.std(arr, axis=0))) / (self.ub - self.lb)
        return float(np.clip(2.0 * spread, 0.0, 1.0))


class BinaryProblem(Problem):
    """Base para problemas con representación binaria de longitud n."""

    family = "binary"

    def __init__(self, n: int = 40, **extra: Any) -> None:
        if n < 1:
            raise ValueError("La longitud binaria debe ser al menos 1.")
        self.n = int(n)
        super().__init__(n=self.n, **extra)

    @property
    def dimension(self) -> int:
        return self.n

    def random_solution(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, 2, size=self.n).astype(np.int8)

    def repair(self, solution: Sequence[float]) -> np.ndarray:
        return (np.asarray(solution) > 0.5).astype(np.int8)

    def validate(self, solution: Any) -> tuple[bool, str]:
        try:
            arr = np.asarray(solution)
        except (TypeError, ValueError):
            return False, "La solución no es interpretable como bits."
        if arr.shape != (self.n,):
            return False, f"Se esperaban {self.n} bits y llegaron {arr.size}."
        if not np.all(np.isin(arr, (0, 1))):
            return False, "La solución contiene valores distintos de 0 y 1."
        return True, "Solución válida."

    def diversity(self, population: Sequence[Any]) -> float:
        arr = np.asarray(population, dtype=float)
        if arr.ndim != 2 or len(arr) < 2:
            return 0.0
        p = np.mean(arr, axis=0)
        return float(np.clip(np.mean(2.0 * p * (1.0 - p)), 0.0, 1.0))


class PermutationProblem(Problem):
    """Base para problemas cuya solución es una permutación de 0..n-1."""

    family = "permutation"

    def __init__(self, n: int = 20, **extra: Any) -> None:
        if n < 2:
            raise ValueError("Se requieren al menos 2 nodos.")
        self.n = int(n)
        super().__init__(n=self.n, **extra)

    @property
    def dimension(self) -> int:
        return self.n

    def random_solution(self, rng: np.random.Generator) -> np.ndarray:
        return rng.permutation(self.n)

    def repair(self, solution: Sequence[float]) -> np.ndarray:
        return repair_permutation(solution)

    def validate(self, solution: Any) -> tuple[bool, str]:
        try:
            arr = np.asarray(solution)
        except (TypeError, ValueError):
            return False, "La solución no es interpretable como permutación."
        if arr.shape != (self.n,):
            return False, f"Se esperaba una permutación de {self.n} elementos."
        if not np.array_equal(np.sort(arr.astype(int)), np.arange(self.n)):
            return False, "La solución no es una permutación válida de 0..n-1."
        return True, "Solución válida."

    def diversity(self, population: Sequence[Any]) -> float:
        perms = [np.asarray(p) for p in population]
        if len(perms) < 2:
            return 0.0
        # Distancia de Hamming normalizada entre pares (muestra acotada).
        sample = perms if len(perms) <= 40 else perms[:: max(1, len(perms) // 40)][:40]
        total = 0.0
        pairs = 0
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                total += float(np.mean(sample[i] != sample[j]))
                pairs += 1
        return float(np.clip(total / pairs, 0.0, 1.0)) if pairs else 0.0

    # -- Ganchos para ACO (solo problemas de permutación) ------------------

    def aco_distance_matrix(self) -> np.ndarray:
        """Matriz de distancias entre los nodos de la permutación."""
        raise NotImplementedError

    def aco_initial_weights(self) -> np.ndarray:
        """Atractivo inicial de cada nodo (por ejemplo, cercanía a la base)."""
        return np.ones(self.n, dtype=float)

    def aco_edges(self, solution: Sequence[int]) -> list[tuple[int, int]]:
        """Aristas donde se deposita feromona para una solución."""
        perm = np.asarray(solution, dtype=int)
        n = perm.size
        return [(int(perm[i]), int(perm[(i + 1) % n])) for i in range(n)]


# ---------------------------------------------------------------------------
# Operadores compartidos (implementados desde cero)
# ---------------------------------------------------------------------------


def sigmoid(x: Any) -> np.ndarray:
    """Sigmoide numéricamente estable."""
    arr = np.clip(np.asarray(x, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-arr))


def repair_permutation(solution: Sequence[float]) -> np.ndarray:
    """Convierte cualquier vector en una permutación válida.

    Un ``argsort`` estable garantiza una permutación de 0..n-1 incluso si
    hay valores repetidos (random-key).
    """
    arr = np.asarray(solution, dtype=float)
    if arr.ndim != 1:
        arr = arr.ravel()
    n = arr.size
    if n == 0:
        return np.array([], dtype=int)
    return np.argsort(arr, kind="stable").astype(int)


def random_keys_to_permutation(keys: Sequence[float]) -> np.ndarray:
    """Decodificación Random-Key: ordena las claves y devuelve los índices."""
    return np.argsort(np.asarray(keys, dtype=float), kind="stable").astype(int)


def ox_crossover(
    parent1: Sequence[int],
    parent2: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Cruza Order Crossover (OX) para permutaciones.

    Copia un segmento contiguo del primer padre y rellena el resto con el
    orden relativo del segundo padre, evitando duplicados.
    """
    p1 = np.asarray(parent1, dtype=int)
    p2 = np.asarray(parent2, dtype=int)
    n = p1.size
    if n < 2:
        return p1.copy()
    a, b = sorted(rng.integers(0, n, size=2))
    if a == b:
        b = min(n, a + 1)
    child = np.full(n, -1, dtype=int)
    child[a:b] = p1[a:b]
    used = set(int(v) for v in child[a:b])
    fill = [int(v) for v in p2 if int(v) not in used]
    positions = list(range(b, n)) + list(range(0, a))
    for pos, value in zip(positions, fill):
        child[pos] = value
    return child


def swap_mutation(
    permutation: Sequence[int],
    rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Muta una permutación con intercambios aleatorios."""
    child = np.asarray(permutation, dtype=int).copy()
    n = child.size
    if n < 2 or rate <= 0:
        return child
    for i in range(n):
        if rng.random() < rate:
            j = int(rng.integers(0, n))
            child[i], child[j] = child[j], child[i]
    return child


def inversion_mutation(
    permutation: Sequence[int],
    rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Invierte un segmento aleatorio si se supera la probabilidad ``rate``."""
    child = np.asarray(permutation, dtype=int).copy()
    n = child.size
    if n < 2 or rng.random() >= rate:
        return child
    a, b = sorted(rng.integers(0, n, size=2))
    if a == b:
        b = min(n, a + 1)
    child[a:b] = child[a:b][::-1]
    return child


def hamming_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Número de posiciones distintas entre dos soluciones."""
    return int(np.sum(np.asarray(a) != np.asarray(b)))


# ---------------------------------------------------------------------------
# Algoritmos
# ---------------------------------------------------------------------------


class MetaheuristicAlgorithm(ABC):
    """Plantilla común para todos los algoritmos metaheurísticos.

    Las subclases solo implementan ``initialize`` y ``step``. La clase base
    se encarga del conteo de evaluaciones, la parada cooperativa, la pausa,
    los snapshots y el mejor global.
    """

    key: str = ""
    name: str = ""
    description: str = ""
    family_support: frozenset[str] = frozenset({"continuous", "binary", "permutation"})

    def __init__(
        self,
        problem: Problem,
        params: dict[str, Any] | None = None,
        seed: int = 0,
        snapshot_every: int = 1,
    ) -> None:
        if problem.family not in self.family_support:
            raise ValueError(
                f"{self.name} no está diseñado para problemas de familia "
                f"'{problem.family}'. Familias soportadas: "
                f"{', '.join(sorted(self.family_support))}."
            )
        self.problem = problem
        self.family = problem.family
        self.params = validate_params(
            self.param_schema(problem.family), params or {}, problem.family
        )
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.snapshot_every = max(1, int(snapshot_every))
        self.population: Any = None
        self.fitness: np.ndarray = np.empty(0, dtype=float)
        self.evaluations = 0
        self.history: list[Snapshot] = []
        self.best_fitness: float | None = None
        self.best_solution: Any = None
        self.iteration = 0
        self.elapsed = 0.0
        self._started_at = 0.0
        self._last_snapshot_iteration = -1
        self._log: list[str] = []

    # -- Esquema -----------------------------------------------------------

    @classmethod
    def param_schema(cls, family: str | None = None) -> tuple[ParamSpec, ...]:
        return ()

    # -- Ganchos de las subclases -----------------------------------------

    @abstractmethod
    def initialize(self) -> None:
        """Crea la población inicial (self.population)."""

    @abstractmethod
    def step(self) -> None:
        """Avanza una generación o iteración."""

    # -- Utilidades para subclases ----------------------------------------

    def native(self, individual: Any) -> Any:
        """Convierte un individuo interno a la solución evaluable.

        Por defecto es la identidad. PSO/DE lo sobrescriben para Random-Key
        y para la variante binaria con sigmoide.
        """
        return individual

    def evaluate_individual(self, individual: Any) -> float:
        value = float(self.problem.evaluate(self.native(individual)))
        self.evaluations += 1
        if not np.isfinite(value):
            return float("inf") if self.problem.sense == MINIMIZE else float("-inf")
        return value

    def evaluate_population(self) -> None:
        self.fitness = np.array(
            [self.evaluate_individual(ind) for ind in self.population], dtype=float
        )

    def refresh_best(self) -> None:
        if len(self.fitness) == 0:
            return
        idx = self.problem.best_index(self.fitness)
        value = float(self.fitness[idx])
        if self.best_fitness is None or self.problem.is_better(value, self.best_fitness):
            self.best_fitness = value
            self.best_solution = np.array(self.native(self.population[idx]), copy=True)

    def sorted_indices(self) -> np.ndarray:
        """Índices ordenados de mejor a peor según el sentido del problema."""
        return np.argsort(self.fitness) if self.problem.sense == MINIMIZE else np.argsort(-self.fitness)

    def sample_population(self, limit: int = 80) -> list[Any]:
        population = list(self.population)
        if len(population) <= limit:
            return population
        step = max(1, len(population) // limit)
        return population[::step][:limit]

    # -- Ejecución ---------------------------------------------------------

    def _wait_if_paused(
        self,
        should_stop: Callable[[], bool] | None,
        should_pause: Callable[[], bool] | None,
    ) -> None:
        # Breve retraso para que el event de pausa se haya propagado
        # entre hilos antes de empezar a esperar.
        if should_pause is not None:
            time.sleep(0.05)
        while should_pause is not None and should_pause():
            if should_stop is not None and should_stop():
                return
            time.sleep(0.05)

    def should_stop_now(
        self,
        should_stop: Callable[[], bool] | None,
        time_limit: float | None,
    ) -> bool:
        if should_stop is not None and should_stop():
            return True
        if time_limit is not None and time_limit > 0:
            if time.perf_counter() - self._started_at >= time_limit:
                return True
        return False

    def build_snapshot(self) -> Snapshot:
        idx = self.problem.best_index(self.fitness)
        mean = float(np.mean(self.fitness)) if len(self.fitness) else float("nan")
        worst = float(
            np.max(self.fitness) if self.problem.sense == MINIMIZE else np.min(self.fitness)
        )
        native_population = [self.native(ind) for ind in self.sample_population()]
        diversity = float(self.problem.diversity(native_population))
        extra: dict[str, Any] = {}
        if self.family in {"continuous", "binary"}:
            extra["population"] = np.array(native_population, copy=True)
        else:
            # Guardar solo un resumen; la mejor permutación ya viaja aparte.
            extra["population_size"] = int(len(self.population))

        assert self.best_fitness is not None
        return Snapshot(
            iteration=self.iteration,
            best_fitness=float(self.best_fitness),
            mean_fitness=mean,
            worst_fitness=worst,
            diversity=diversity,
            best_solution=np.array(self.best_solution, copy=True),
            evaluations=self.evaluations,
            elapsed=float(self.elapsed),
            extra=extra,
        )

    def record_snapshot(self, on_snapshot: Callable[[Snapshot], None] | None) -> None:
        self.elapsed = time.perf_counter() - self._started_at
        snapshot = self.build_snapshot()
        self.history.append(snapshot)
        self._last_snapshot_iteration = self.iteration
        if on_snapshot is not None:
            on_snapshot(snapshot)

    def run(
        self,
        max_iterations: int = 100,
        time_limit: float | None = None,
        on_snapshot: Callable[[Snapshot], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        should_pause: Callable[[], bool] | None = None,
        on_iteration: Callable[[int], None] | None = None,
    ) -> tuple[Any, float]:
        """Ejecuta el algoritmo con parada y pausa cooperativas."""
        if max_iterations < 1:
            raise ValueError("Las iteraciones deben ser al menos 1.")
        self._started_at = time.perf_counter()
        self.iteration = 0
        self.evaluations = 0
        self.history.clear()
        self.best_fitness = None
        self.best_solution = None
        self.population = None
        self.fitness = np.empty(0, dtype=float)

        self.initialize()
        if self.population is None or len(self.population) == 0:
            raise RuntimeError("El algoritmo no generó una población inicial.")
        # Algunos algoritmos (PSO, AIS, DE) ya evalúan al inicializar porque
        # necesitan la afinidad para construir sus estructuras internas.
        if len(self.fitness) != len(self.population):
            self.evaluate_population()
        self.refresh_best()
        self.record_snapshot(on_snapshot)

        for iteration in range(1, max_iterations + 1):
            self._wait_if_paused(should_stop, should_pause)
            if self.should_stop_now(should_stop, time_limit):
                break
            self.step()
            self.iteration = iteration
            self.refresh_best()
            if iteration % self.snapshot_every == 0 or iteration == max_iterations:
                self.record_snapshot(on_snapshot)
            if on_iteration is not None:
                on_iteration(iteration)

        if self._last_snapshot_iteration != self.iteration:
            self.record_snapshot(on_snapshot)

        assert self.best_solution is not None
        return self.best_solution, float(self.best_fitness)

    def summary(self) -> dict[str, Any]:
        return {
            "algorithm": self.name,
            "problem": self.problem.name,
            "best_fitness": self.best_fitness,
            "evaluations": self.evaluations,
            "iterations": self.iteration,
            "elapsed": round(self.elapsed, 4),
            "seed": self.seed,
            "params": dict(self.params),
        }
