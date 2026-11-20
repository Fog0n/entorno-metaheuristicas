"""Motor de ejecución en hilo secundario.

La interfaz nunca ejecuta el algoritmo en el hilo de Tkinter. Este motor
crea un hilo daemon, publica snapshots en una cola y expone operaciones
seguras de iniciar, pausar, reanudar, detener y reiniciar. La comunicación
con los widgets se hace exclusivamente mediante la cola y ``root.after``.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from core.base import MetaheuristicAlgorithm, Problem, Snapshot


@dataclass
class RunConfig:
    """Configuración común de una corrida."""

    max_iterations: int = 150
    seed: int = 42
    time_limit: float | None = None
    snapshot_every: int = 1

    def validate(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("Las iteraciones deben ser al menos 1.")
        if self.time_limit is not None and self.time_limit <= 0:
            raise ValueError("El límite de tiempo debe ser positivo o vacío.")
        if self.snapshot_every < 1:
            raise ValueError("La frecuencia de snapshot debe ser al menos 1.")


@dataclass
class EngineEvent:
    kind: str  # started | snapshot | finished | stopped | error
    payload: Any = None


class ExperimentEngine:
    """Orquesta una corrida de un algoritmo sobre un problema."""

    def __init__(
        self,
        problem_class: type[Problem],
        problem_params: dict[str, Any],
        algorithm_class: type[MetaheuristicAlgorithm],
        algorithm_params: dict[str, Any],
        config: RunConfig,
    ) -> None:
        if problem_class.family not in algorithm_class.family_support:
            raise ValueError(
                f"{algorithm_class.name} no admite problemas de familia "
                f"'{problem_class.family}'."
            )
        config.validate()
        self.problem_class = problem_class
        self.problem_params = dict(problem_params)
        self.algorithm_class = algorithm_class
        self.algorithm_params = dict(algorithm_params)
        self.config = config

        self.events: queue.Queue[EngineEvent] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._lock = threading.Lock()
        self._algorithm: MetaheuristicAlgorithm | None = None
        self.last_summary: dict[str, Any] | None = None

    # -- Estado ------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_paused(self) -> bool:
        return self._pause.is_set()

    @property
    def algorithm(self) -> MetaheuristicAlgorithm | None:
        return self._algorithm

    # -- Control -----------------------------------------------------------

    def start(self) -> bool:
        """Inicia una corrida nueva. Devuelve False si ya hay una activa."""
        if self.is_running:
            return False
        self._stop = threading.Event()
        self._pause = threading.Event()
        self.last_summary = None
        self._thread = threading.Thread(
            target=self._worker,
            name="metaheuristic-engine",
            daemon=True,
        )
        self._thread.start()
        return True

    def pause(self) -> None:
        if self.is_running:
            self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def toggle_pause(self) -> bool:
        if self._pause.is_set():
            self.resume()
        else:
            self.pause()
        return self._pause.is_set()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._pause.clear()
        thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=timeout)

    def restart(self) -> bool:
        """Detiene la corrida actual y lanza una nueva con la misma configuración."""
        if self.is_running:
            self.stop()
            # Pequeña espera para que el hilo libere el estado.
            deadline = time.perf_counter() + 2.0
            while self.is_running and time.perf_counter() < deadline:
                time.sleep(0.02)
        return self.start()

    def drain(self) -> list[EngineEvent]:
        """Extrae todos los eventos disponibles sin bloquear."""
        items: list[EngineEvent] = []
        while True:
            try:
                items.append(self.events.get_nowait())
            except queue.Empty:
                break
        return items

    # -- Hilo --------------------------------------------------------------

    def _worker(self) -> None:
        try:
            problem = self.problem_class(**self.problem_params)
            algorithm = self.algorithm_class(
                problem,
                params=self.algorithm_params,
                seed=self.config.seed,
                snapshot_every=self.config.snapshot_every,
            )
            with self._lock:
                self._algorithm = algorithm
            self.events.put(
                EngineEvent(
                    "started",
                    {
                        "problem": problem.name,
                        "algorithm": algorithm.name,
                        "seed": self.config.seed,
                        "iterations": self.config.max_iterations,
                    },
                )
            )
            algorithm.run(
                max_iterations=self.config.max_iterations,
                time_limit=self.config.time_limit,
                on_snapshot=lambda snap: self.events.put(EngineEvent("snapshot", snap)),
                should_stop=self._stop.is_set,
                should_pause=self._pause.is_set,
            )
            self.last_summary = algorithm.summary()
            kind = "stopped" if self._stop.is_set() else "finished"
            self.events.put(EngineEvent(kind, self.last_summary))
        except Exception:
            self.events.put(EngineEvent("error", traceback.format_exc()))
        finally:
            # Garantiza que is_running pase a False incluso si el hilo termina.
            self._thread = None


def run_sync(
    problem: Problem,
    algorithm_class: type[MetaheuristicAlgorithm],
    algorithm_params: dict[str, Any],
    config: RunConfig,
    on_snapshot: Callable[[Snapshot], None] | None = None,
) -> MetaheuristicAlgorithm:
    """Ejecuta una corrida en el hilo actual (para scripts y pruebas)."""
    algorithm = algorithm_class(
        problem,
        params=algorithm_params,
        seed=config.seed,
        snapshot_every=config.snapshot_every,
    )
    algorithm.run(
        max_iterations=config.max_iterations,
        time_limit=config.time_limit,
        on_snapshot=on_snapshot,
    )
    return algorithm
