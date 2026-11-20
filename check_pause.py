"""Prueba aislada del ciclo pausar / reanudar / detener del motor."""

from __future__ import annotations

import time

from algorithms import get_algorithm
from core.engine import ExperimentEngine, RunConfig
from problems import get_problem


def main() -> None:
    problem_cls = get_problem("sphere")
    algo_cls = get_algorithm("ga")
    engine = ExperimentEngine(
        problem_cls,
        problem_cls.default_params(),
        algo_cls,
        {"population_size": 30, "elitism": 2},
        RunConfig(max_iterations=100000, seed=1, snapshot_every=5),
    )
    engine.start()
    time.sleep(0.4)

    engine.pause()
    time.sleep(0.15)
    for _ in engine.drain():  # vaciar lo pendiente
        pass
    before = len(engine.drain())
    base = 0
    time.sleep(1.0)
    after = len(engine.drain())
    print(f"pausado: eventos nuevos en 1s = {after} (esperado 0)", flush=True)
    assert after == 0, f"la pausa no detuvo el motor: {after} eventos"

    engine.resume()
    time.sleep(0.5)
    resumed = len(engine.drain())
    print(f"reanudado: eventos en 0.5s = {resumed} (esperado > 0)", flush=True)
    assert resumed > 0, "no se reanudó"

    t0 = time.perf_counter()
    engine.stop()
    dt = time.perf_counter() - t0
    print(f"detenido en {dt:.2f}s, is_running={engine.is_running}", flush=True)
    assert not engine.is_running
    print("ENGINE PAUSE OK", flush=True)


if __name__ == "__main__":
    main()
