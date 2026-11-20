"""Experimentos reproducibles y generación de fig-3.png ... fig-10.png.

Uso:
    python run_experiments.py
    python run_experiments.py --quick
    python run_experiments.py --outdir resultados

El script no abre ventanas: usa el backend Agg de Matplotlib y ejecuta los
algoritmos en el hilo principal a través de ``core.engine.run_sync``.
Todas las corridas usan semillas fijas y parámetros explícitos.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from algorithms import ALGORITHMS, ALGORITHM_ORDER, get_algorithm
from core.base import defaults_for
from core.engine import RunConfig, run_sync
from problems import get_problem
from problems.delivery_zapopan import DeliveryZapopanProblem
from problems.discrete import TSPProblem

ROOT = Path(__file__).resolve().parent
ALGO_LABEL = {"ga": "GA", "aco": "ACO", "pso": "PSO", "ais": "AIS", "de": "DE"}
ALGO_COLOR = {
    "ga": "#2f9e63",
    "aco": "#e0a33a",
    "pso": "#3b6fb5",
    "ais": "#b05a9e",
    "de": "#c0504d",
}

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


class ExperimentRunner:
    """Ejecuta corridas con caché para reutilizarlas entre figuras."""

    def __init__(self, quick: bool = False) -> None:
        self.quick = quick
        self.cache: dict[tuple[Any, ...], Any] = {}
        self.records: list[dict[str, Any]] = []
        self.started = time.perf_counter()

    # -- Utilidades de parámetros -----------------------------------------

    @staticmethod
    def algo_params(
        algorithm_key: str,
        family: str,
        **overrides: Any,
    ) -> dict[str, Any]:
        cls = get_algorithm(algorithm_key)
        params = defaults_for(cls.param_schema(family), family)
        params.update(overrides)
        return params

    # -- Ejecución ---------------------------------------------------------

    def run(
        self,
        problem_key: str,
        problem_params: dict[str, Any],
        algorithm_key: str,
        algorithm_params: dict[str, Any],
        seed: int,
        iterations: int,
    ) -> Any:
        key = (
            problem_key,
            tuple(sorted(problem_params.items())),
            algorithm_key,
            tuple(sorted(algorithm_params.items())),
            seed,
            iterations,
        )
        if key in self.cache:
            return self.cache[key]

        problem = get_problem(problem_key)(**problem_params)
        config = RunConfig(
            max_iterations=iterations,
            seed=seed,
            snapshot_every=1,
        )
        algorithm = run_sync(
            problem,
            get_algorithm(algorithm_key),
            algorithm_params,
            config,
        )
        self.cache[key] = algorithm
        self.records.append(
            {
                "problema": problem.name,
                "algoritmo": ALGO_LABEL[algorithm_key],
                "semilla": seed,
                "iteraciones": iterations,
                "mejor_objetivo": algorithm.best_fitness,
                "evaluaciones": algorithm.evaluations,
                "tiempo_s": round(algorithm.elapsed, 4),
            }
        )
        print(
            f"  {problem.name:<26} {ALGO_LABEL[algorithm_key]:<4} "
            f"seed={seed:<5} mejor={algorithm.best_fitness:>12.4f}",
            flush=True,
        )
        return algorithm

    def curve(self, algorithm: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        iterations = np.array([s.iteration for s in algorithm.history])
        best = np.array([s.best_fitness for s in algorithm.history])
        mean = np.array([s.mean_fitness for s in algorithm.history])
        return iterations, best, mean


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, outdir: Path, name: str) -> None:
    path = outdir / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  guardada {path.name}", flush=True)


def _median_curve(algorithms: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    iterations = np.array([s.iteration for s in algorithms[0].history])
    curves = np.vstack(
        [np.array([s.best_fitness for s in algo.history]) for algo in algorithms]
    )
    return iterations, np.median(curves, axis=0)


def _plot_ga_convergence(runner: ExperimentRunner, outdir: Path) -> None:
    sphere_runs = [
        runner.run(
            "sphere",
            {"dim": 10, "lb": -5.12, "ub": 5.12},
            "ga",
            runner.algo_params("ga", "continuous", population_size=40, elitism=2),
            seed=2026,
            iterations=120 if not runner.quick else 40,
        )
    ]
    rastrigin_runs = [
        runner.run(
            "rastrigin",
            {"dim": 10, "lb": -5.12, "ub": 5.12},
            "ga",
            runner.algo_params("ga", "continuous", population_size=40, elitism=2),
            seed=2027,
            iterations=120 if not runner.quick else 40,
        )
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, runs, title in (
        (axes[0], sphere_runs, "Sphere · GA"),
        (axes[1], rastrigin_runs, "Rastrigin · GA"),
    ):
        iterations, best, mean = runner.curve(runs[0])
        ax.plot(iterations, best, color="#2f9e63", linewidth=2.0, label="Mejor")
        ax.plot(iterations, mean, color="#e0a33a", linewidth=1.2, label="Media")
        ax.set_title(title)
        ax.set_xlabel("Iteración")
        ax.set_ylabel("Fitness (menor es mejor)")
        ax.legend()
    fig.suptitle("Convergencia del Algoritmo Genético", fontsize=12)
    fig.tight_layout()
    _save(fig, outdir, "fig-3.png")


def _comparison_scenario(
    runner: ExperimentRunner,
    problem_key: str,
    problem_params: dict[str, Any],
    label: str,
    seeds: int,
    iterations: int,
    population: int,
    family: str = "continuous",
) -> dict[str, list[Any]]:
    results: dict[str, list[Any]] = {}
    for algo_key in ("ga", "pso", "ais", "de"):
        overrides: dict[str, Any] = {"population_size": population} if algo_key in {"ga", "ais", "de"} else {"swarm_size": population}
        if algo_key == "ga":
            overrides["elitism"] = 2
        if algo_key == "ais":
            overrides.update({"n_selected": min(12, population // 3), "clones_per_antibody": 4})
        results[algo_key] = [
            runner.run(
                problem_key,
                problem_params,
                algo_key,
                runner.algo_params(algo_key, family, **overrides),
                seed=2026 + i,
                iterations=iterations,
            )
            for i in range(seeds)
        ]
        print(f"  {label}: {ALGO_LABEL[algo_key]} listo ({seeds} semillas)", flush=True)
    return results


def _plot_continuous_comparison(
    runner: ExperimentRunner,
    outdir: Path,
    problem_key: str,
    problem_params: dict[str, Any],
    title: str,
    seeds: int,
    iterations: int,
    population: int,
    figure_name: str,
) -> dict[str, list[Any]]:
    results = _comparison_scenario(
        runner, problem_key, problem_params, title, seeds, iterations, population
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    for algo_key, runs in results.items():
        iterations_axis, median = _median_curve(runs)
        axes[0].plot(
            iterations_axis, median, label=ALGO_LABEL[algo_key],
            color=ALGO_COLOR[algo_key], linewidth=1.8,
        )
    axes[0].set_yscale("log")
    axes[0].set_title(f"{title} · mejor de la población (mediana)")
    axes[0].set_xlabel("Iteración")
    axes[0].set_ylabel("Fitness (escala log)")
    axes[0].legend(ncol=2)

    box_data = [[algo.best_fitness for algo in runs] for runs in results.values()]
    axes[1].boxplot(
        box_data,
        tick_labels=[ALGO_LABEL[k] for k in results],
        patch_artist=True,
        boxprops={"facecolor": "#d8e8f5", "edgecolor": "#3b6fb5"},
        medianprops={"color": "#c0504d"},
    )
    axes[1].set_title("Dispersión final por semilla")
    axes[1].set_ylabel("Mejor objetivo")
    fig.suptitle(f"Comparación de metaheurísticas · {title}", fontsize=12)
    fig.tight_layout()
    _save(fig, outdir, figure_name)
    return results


def _plot_binary(runner: ExperimentRunner, outdir: Path) -> None:
    onemax_params = {"n": 40}
    knapsack_params = {"n": 30, "capacity": 180.0, "penalty": 25.0}
    seeds = 3 if not runner.quick else 2
    iterations = 100 if not runner.quick else 40
    population = 40 if not runner.quick else 24

    onemax = _comparison_scenario(
        runner, "onemax", onemax_params, "OneMax", seeds, iterations, population, "binary"
    )
    knapsack = _comparison_scenario(
        runner, "knapsack", knapsack_params, "Mochila", seeds, iterations, population, "binary"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for algo_key, runs in onemax.items():
        it, median = _median_curve(runs)
        axes[0].plot(it, median, label=ALGO_LABEL[algo_key], color=ALGO_COLOR[algo_key])
    axes[0].axhline(40, color="#777777", linestyle="--", linewidth=1, label="Óptimo (40)")
    axes[0].set_title("OneMax · 40 bits")
    axes[0].set_xlabel("Iteración")
    axes[0].set_ylabel("Bits en 1")
    axes[0].legend(ncol=2)

    for algo_key, runs in knapsack.items():
        it, median = _median_curve(runs)
        axes[1].plot(it, median, label=ALGO_LABEL[algo_key], color=ALGO_COLOR[algo_key])
    axes[1].set_title("Mochila 0-1 · 30 artículos, W=180")
    axes[1].set_xlabel("Iteración")
    axes[1].set_ylabel("Valor con penalización")
    axes[1].legend(ncol=2)
    fig.suptitle("Problemas binarios: OneMax y Mochila", fontsize=12)
    fig.tight_layout()
    _save(fig, outdir, "fig-6.png")


def _tsp_runs(runner: ExperimentRunner) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    problem_params = {"n": 20, "coord_seed": 7, "area": 100.0}
    seeds = 8 if not runner.quick else 2
    iterations = 120 if not runner.quick else 40
    population = 30 if not runner.quick else 20
    overrides = {
        "ga": {"population_size": population, "elitism": 2},
        "aco": {"n_ants": population},
        "pso": {"swarm_size": population},
        "ais": {"population_size": population, "n_selected": 10, "clones_per_antibody": 4},
        "de": {"population_size": population},
    }
    results: dict[str, list[Any]] = {}
    for algo_key in ALGORITHM_ORDER:
        results[algo_key] = [
            runner.run(
                "tsp",
                problem_params,
                algo_key,
                runner.algo_params(algo_key, "permutation", **overrides[algo_key]),
                seed=3000 + i,
                iterations=iterations,
            )
            for i in range(seeds)
        ]
        print(f"  TSP: {ALGO_LABEL[algo_key]} listo ({seeds} semillas)", flush=True)
    return results, problem_params


def _plot_tsp(runner: ExperimentRunner, outdir: Path, tsp_results: dict[str, list[Any]], params: dict[str, Any]) -> None:
    problem = TSPProblem(**params)
    best_algorithm = min(
        (algo for runs in tsp_results.values() for algo in runs),
        key=lambda algo: algo.best_fitness,
    )
    route = np.asarray(best_algorithm.best_solution, dtype=int)
    closed = np.append(route, route[0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    ax.plot(problem.coordinates[closed, 0], problem.coordinates[closed, 1], color="#2f9e63", linewidth=1.8)
    ax.scatter(problem.coordinates[:, 0], problem.coordinates[:, 1], s=28, c="#3b6fb5", zorder=3)
    ax.scatter(
        problem.coordinates[route[0], 0], problem.coordinates[route[0], 1],
        s=120, marker="*", c="#e0a33a", zorder=4,
    )
    for i, (x, y) in enumerate(problem.coordinates):
        ax.annotate(str(i + 1), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_title(f"Mejor ruta encontrada · {best_algorithm.best_fitness:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")

    for algo_key, runs in tsp_results.items():
        it, median = _median_curve(runs)
        axes[1].plot(it, median, label=ALGO_LABEL[algo_key], color=ALGO_COLOR[algo_key])
    axes[1].set_title("Convergencia mediana · TSP 20 ciudades")
    axes[1].set_xlabel("Iteración")
    axes[1].set_ylabel("Distancia")
    axes[1].legend(ncol=2)
    fig.suptitle("Agente viajero: calidad de la ruta", fontsize=12)
    fig.tight_layout()
    _save(fig, outdir, "fig-7.png")


def _plot_tsp_boxplots(outdir: Path, tsp_results: dict[str, list[Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    data = [[algo.best_fitness for algo in runs] for runs in tsp_results.values()]
    bp = ax.boxplot(
        data,
        tick_labels=[ALGO_LABEL[k] for k in tsp_results],
        patch_artist=True,
        boxprops={"facecolor": "#d8e8f5", "edgecolor": "#3b6fb5"},
        medianprops={"color": "#c0504d", "linewidth": 1.6},
        whiskerprops={"color": "#3b6fb5"},
        capprops={"color": "#3b6fb5"},
    )
    for patch, key in zip(bp["boxes"], tsp_results):
        patch.set_facecolor(ALGO_COLOR[key])
        patch.set_alpha(0.35)
    ax.set_title("Robustez en TSP · 8 semillas independientes")
    ax.set_ylabel("Mejor distancia")
    ax.set_xlabel("Algoritmo")
    fig.tight_layout()
    _save(fig, outdir, "fig-9.png")


def _delivery_runs(runner: ExperimentRunner) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    params = {"q_max": 180.0, "road_factor": 1.25, "fixed_cost": 8.0}
    seeds = 4 if not runner.quick else 2
    iterations = 100 if not runner.quick else 40
    population = 30 if not runner.quick else 20
    overrides = {
        "ga": {"population_size": population, "elitism": 2},
        "aco": {"n_ants": population},
        "pso": {"swarm_size": population},
        "ais": {"population_size": population, "n_selected": 10, "clones_per_antibody": 4},
        "de": {"population_size": population},
    }
    results: dict[str, list[Any]] = {}
    for algo_key in ALGORITHM_ORDER:
        results[algo_key] = [
            runner.run(
                "delivery_zapopan",
                params,
                algo_key,
                runner.algo_params(algo_key, "permutation", **overrides[algo_key]),
                seed=4000 + i,
                iterations=iterations,
            )
            for i in range(seeds)
        ]
        print(f"  Reparto: {ALGO_LABEL[algo_key]} listo ({seeds} semillas)", flush=True)
    return results, params


def _plot_delivery(runner: ExperimentRunner, outdir: Path, results: dict[str, list[Any]], params: dict[str, Any]) -> None:
    problem = DeliveryZapopanProblem(**params)
    best_algorithm = min(
        (algo for runs in results.values() for algo in runs),
        key=lambda algo: algo.best_fitness,
    )
    solution = np.asarray(best_algorithm.best_solution, dtype=int)
    routes = problem.decode_routes(solution)
    lines = problem.route_points(solution)
    depot, clients = problem.plot_coordinates()
    cmap = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(13, 7))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1])
    ax_map = fig.add_subplot(grid[:, 0])
    ax_conv = fig.add_subplot(grid[0, 1])
    ax_load = fig.add_subplot(grid[1, 1])

    ax_map.scatter(clients[:, 0], clients[:, 1], s=34, c="#777777", zorder=3)
    for i, (x, y) in enumerate(clients):
        ax_map.annotate(problem.names[i], (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    for idx, line in enumerate(lines):
        arr = np.asarray(line)
        color = cmap(idx % 10)
        ax_map.plot(arr[:, 0], arr[:, 1], color=color, linewidth=2.0)
        ax_map.scatter(arr[1:-1, 0], arr[1:-1, 1], s=26, color=color, zorder=4)
    ax_map.scatter(depot[:, 0], depot[:, 1], marker="*", s=220, c="#e0a33a", zorder=5, label="Base CEDIS")
    ax_map.set_title(
        f"Rutas de reparto Zapopan · {len(routes)} rutas · objetivo {best_algorithm.best_fitness:.2f}"
    )
    ax_map.set_xlabel("Longitud")
    ax_map.set_ylabel("Latitud")
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.legend(loc="best")

    for algo_key, runs in results.items():
        it, median = _median_curve(runs)
        ax_conv.plot(it, median, label=ALGO_LABEL[algo_key], color=ALGO_COLOR[algo_key])
    ax_conv.set_title("Convergencia mediana · rutas Zapopan")
    ax_conv.set_xlabel("Iteración")
    ax_conv.set_ylabel("Costo (km + rutas)")
    ax_conv.legend(ncol=2)

    loads = [route["carga"] for route in routes]
    positions = np.arange(len(routes))
    ax_load.bar(positions, loads, color=[cmap(i % 10) for i in positions])
    ax_load.axhline(problem.q_max, color="#c0504d", linestyle="--", label=f"Q_max={problem.q_max:.0f} kg")
    ax_load.set_xticks(positions)
    ax_load.set_xticklabels([f"R{i + 1}" for i in positions])
    ax_load.set_ylabel("Carga (kg)")
    ax_load.set_title("Carga por ruta")
    ax_load.legend()

    fig.suptitle("Caso real: reparto local en Zapopan/Guadalajara", fontsize=12)
    fig.tight_layout()
    _save(fig, outdir, "fig-8.png")


def _plot_scalability(runner: ExperimentRunner, outdir: Path) -> None:
    sizes = [10, 15, 20, 25] if not runner.quick else [10, 20]
    seeds = 3 if not runner.quick else 2
    iterations = 80 if not runner.quick else 30
    algorithms = ("ga", "aco")
    objective: dict[str, list[float]] = {key: [] for key in algorithms}
    times: dict[str, list[float]] = {key: [] for key in algorithms}

    for size in sizes:
        for algo_key in algorithms:
            overrides = (
                {"population_size": 25, "elitism": 2}
                if algo_key == "ga"
                else {"n_ants": 20}
            )
            runs = [
                runner.run(
                    "tsp",
                    {"n": size, "coord_seed": 7, "area": 100.0},
                    algo_key,
                    runner.algo_params(algo_key, "permutation", **overrides),
                    seed=5000 + i,
                    iterations=iterations,
                )
                for i in range(seeds)
            ]
            objective[algo_key].append(float(np.median([r.best_fitness for r in runs])))
            times[algo_key].append(float(np.median([r.elapsed for r in runs])))
        print(f"  Escalabilidad n={size} lista", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for algo_key in algorithms:
        axes[0].plot(sizes, objective[algo_key], marker="o", label=ALGO_LABEL[algo_key], color=ALGO_COLOR[algo_key])
        axes[1].plot(sizes, times[algo_key], marker="o", label=ALGO_LABEL[algo_key], color=ALGO_COLOR[algo_key])
    axes[0].set_title("Calidad vs número de ciudades")
    axes[0].set_xlabel("Ciudades")
    axes[0].set_ylabel("Mejor distancia (mediana)")
    axes[0].legend()
    axes[1].set_title("Tiempo de cómputo vs tamaño")
    axes[1].set_xlabel("Ciudades")
    axes[1].set_ylabel("Tiempo (s)")
    axes[1].legend()
    fig.suptitle("Escalabilidad en TSP", fontsize=12)
    fig.tight_layout()
    _save(fig, outdir, "fig-10.png")


# ---------------------------------------------------------------------------
# Salidas y orquestación
# ---------------------------------------------------------------------------


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _write_outputs(runner: ExperimentRunner, outdir: Path) -> None:
    csv_path = outdir / "resultados_experimentos.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(runner.records[0].keys()))
        writer.writeheader()
        writer.writerows(runner.records)

    summary = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "modo": "quick" if runner.quick else "completo",
        "corridas": len(runner.records),
        "tiempo_total_s": round(time.perf_counter() - runner.started, 2),
        "resultados": runner.records,
    }
    json_path = outdir / "resultados_experimentos.json"
    json_path.write_text(
        json.dumps(_to_builtin(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nResultados: {csv_path.name}, {json_path.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Corridas cortas para verificar el script.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT,
        help="Carpeta de salida (por defecto la raíz del proyecto).",
    )
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Entorno de Metaheurísticas · experimentos reproducibles")
    print(f"Modo: {'quick' if args.quick else 'completo'} · salida: {outdir}")
    print("=" * 72, flush=True)

    runner = ExperimentRunner(quick=args.quick)
    seeds = 5 if not args.quick else 2
    iterations = 120 if not args.quick else 40
    population = 40 if not args.quick else 24

    print("\n[1/7] Convergencia GA (fig-3)…", flush=True)
    _plot_ga_convergence(runner, outdir)

    print("\n[2/7] Comparación en Sphere (fig-4)…", flush=True)
    _plot_continuous_comparison(
        runner, outdir, "sphere", {"dim": 10, "lb": -5.12, "ub": 5.12},
        "Sphere (10 dimensiones)", seeds, iterations, population, "fig-4.png",
    )

    print("\n[3/7] Comparación en Rastrigin (fig-5)…", flush=True)
    _plot_continuous_comparison(
        runner, outdir, "rastrigin", {"dim": 10, "lb": -5.12, "ub": 5.12},
        "Rastrigin (10 dimensiones)", seeds, iterations, population, "fig-5.png",
    )

    print("\n[4/7] Problemas binarios (fig-6)…", flush=True)
    _plot_binary(runner, outdir)

    print("\n[5/7] TSP: ruta y convergencia (fig-7)…", flush=True)
    tsp_results, tsp_params = _tsp_runs(runner)
    _plot_tsp(runner, outdir, tsp_results, tsp_params)

    print("\n[6/7] Reparto Zapopan (fig-8)…", flush=True)
    delivery_results, delivery_params = _delivery_runs(runner)
    _plot_delivery(runner, outdir, delivery_results, delivery_params)

    print("\n[7/7] Robustez y escalabilidad (fig-9, fig-10)…", flush=True)
    _plot_tsp_boxplots(outdir, tsp_results)
    _plot_scalability(runner, outdir)

    _write_outputs(runner, outdir)
    print("\nListo. Figuras fig-3.png … fig-10.png generadas.", flush=True)


if __name__ == "__main__":
    main()
