"""Interfaz gráfica completa en CustomTkinter.

La ventana se organiza en una barra lateral con formularios dinámicos y un
panel principal con métricas, convergencia, visualización de la solución y
registro. Las gráficas de Matplotlib se embeben con FigureCanvasTkAgg y se
actualizan solo desde el hilo principal de Tkinter.
"""

from __future__ import annotations

import queue
import time
from datetime import datetime
from typing import Any

import customtkinter as ctk
import matplotlib

matplotlib.use("TkAgg")

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from algorithms import ALGORITHM_ORDER, ALGORITHMS, get_algorithm, is_compatible
from core.base import ParamSpec, Problem, Snapshot
from core.engine import EngineEvent, ExperimentEngine, RunConfig
from problems import PROBLEM_ORDER, PROBLEMS, get_problem


class MetaheuristicApp(ctk.CTk):
    """Aplicación de escritorio del entorno de metaheurísticas."""

    def __init__(self) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        super().__init__()

        self.title("Entorno Interactivo para Algoritmos Metaheurísticos")
        self.geometry("1480x920")
        self.minsize(1220, 760)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.font_title = ctk.CTkFont(size=24, weight="bold")
        self.font_section = ctk.CTkFont(size=13, weight="bold")
        self.font_small = ctk.CTkFont(size=10)
        self.font_metric = ctk.CTkFont(size=20, weight="bold")

        self.problem_key = PROBLEM_ORDER[0]
        self.algorithm_key = ALGORITHM_ORDER[0]
        self.problem_fields: dict[str, tuple[ParamSpec, Any, Any]] = {}
        self.algorithm_fields: dict[str, tuple[ParamSpec, Any, Any]] = {}
        self.current_problem: Problem | None = None
        self.engine: ExperimentEngine | None = None
        self.snapshots: list[Snapshot] = []
        self.view_index = 0
        self.follow_latest = True
        self._run_token = 0
        self._ui_running = False
        self._last_draw = 0.0
        self._dirty = False
        self._updating_slider = False
        self._contour_cache: dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}
        self._max_iterations = 150

        self._build_layout()
        self._build_sidebar()
        self._build_main()
        self._select_problem(self.problem_key, initial=True)
        self._select_algorithm(self.algorithm_key, initial=True)
        self._draw_empty()
        self._log(
            "Listo. Elige un problema y un algoritmo, ajusta sus parámetros "
            "y pulsa Iniciar."
        )
        self.after(200, self._refresh_preview)

    # ------------------------------------------------------------------
    # Construcción de la ventana
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkScrollableFrame(
            self, width=360, corner_radius=0, fg_color=("gray92", "gray12")
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_columnconfigure(0, weight=1)

        self.main = ctk.CTkFrame(self, corner_radius=0, fg_color=("gray97", "gray15"))
        self.main.grid(row=0, column=1, sticky="nsew", padx=(1, 0))
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(2, weight=1)

    def _section(self, text: str) -> None:
        label = ctk.CTkLabel(
            self.sidebar, text=text, font=self.font_section, anchor="w"
        )
        label.grid(row=self._next_sidebar_row(), column=0, sticky="ew", pady=(14, 2))
        self._sidebar_separator()

    def _sidebar_separator(self) -> None:
        sep = ctk.CTkFrame(self.sidebar, height=1, fg_color=("gray75", "gray30"))
        sep.grid(row=self._next_sidebar_row(), column=0, sticky="ew", pady=(0, 6))

    def _next_sidebar_row(self) -> int:
        if not hasattr(self, "_sidebar_row"):
            self._sidebar_row = 0
        row = self._sidebar_row
        self._sidebar_row += 1
        return row

    # -- Barra lateral -----------------------------------------------------

    def _build_sidebar(self) -> None:
        ctk.CTkLabel(
            self.sidebar,
            text="METAHEURÍSTICAS",
            font=self.font_title,
            anchor="w",
        ).grid(row=self._next_sidebar_row(), column=0, sticky="ew", padx=0, pady=(4, 0))
        ctk.CTkLabel(
            self.sidebar,
            text="Entorno interactivo de experimentación",
            font=self.font_small,
            text_color=("gray35", "gray65"),
            anchor="w",
        ).grid(row=self._next_sidebar_row(), column=0, sticky="ew", pady=(0, 8))

        # Problema ---------------------------------------------------------
        self._section("PROBLEMA")
        self.problem_names = [PROBLEMS[k].name for k in PROBLEM_ORDER]
        self.problem_key_by_name = {PROBLEMS[k].name: k for k in PROBLEM_ORDER}
        self.problem_name_by_key = {k: PROBLEMS[k].name for k in PROBLEM_ORDER}
        self.problem_var = ctk.StringVar(value=self.problem_name_by_key[self.problem_key])
        self.problem_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=self.problem_names,
            variable=self.problem_var,
            command=self._on_problem_change,
        )
        self.problem_menu.grid(row=self._next_sidebar_row(), column=0, sticky="ew", pady=(2, 2))
        self.family_label = ctk.CTkLabel(
            self.sidebar, text="", font=self.font_small,
            text_color=("gray35", "gray65"), anchor="w",
        )
        self.family_label.grid(row=self._next_sidebar_row(), column=0, sticky="ew")
        self.problem_form = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.problem_form.grid(row=self._next_sidebar_row(), column=0, sticky="ew")
        self.problem_form.grid_columnconfigure(0, weight=1)

        # Algoritmo --------------------------------------------------------
        self._section("ALGORITMO")
        self.algorithm_names = [ALGORITHMS[k].name for k in ALGORITHM_ORDER]
        self.algorithm_key_by_name = {ALGORITHMS[k].name: k for k in ALGORITHM_ORDER}
        self.algorithm_name_by_key = {k: ALGORITHMS[k].name for k in ALGORITHM_ORDER}
        self.algorithm_var = ctk.StringVar(value=self.algorithm_name_by_key[self.algorithm_key])
        self.algorithm_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=self.algorithm_names,
            variable=self.algorithm_var,
            command=self._on_algorithm_change,
        )
        self.algorithm_menu.grid(row=self._next_sidebar_row(), column=0, sticky="ew", pady=(2, 2))
        self.compat_label = ctk.CTkLabel(
            self.sidebar, text="", font=self.font_small,
            text_color=("gray35", "gray65"), anchor="w", wraplength=320,
            justify="left",
        )
        self.compat_label.grid(row=self._next_sidebar_row(), column=0, sticky="ew")
        self.algorithm_form = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.algorithm_form.grid(row=self._next_sidebar_row(), column=0, sticky="ew")
        self.algorithm_form.grid_columnconfigure(0, weight=1)
        self.restore_button = ctk.CTkButton(
            self.sidebar, text="Restaurar parámetros", height=26,
            fg_color="transparent", border_width=1, command=self._restore_defaults,
        )
        self.restore_button.grid(row=self._next_sidebar_row(), column=0, sticky="ew", pady=(4, 0))

        # Corrida ----------------------------------------------------------
        self._section("CORRIDA")
        run_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        run_frame.grid(row=self._next_sidebar_row(), column=0, sticky="ew")
        run_frame.grid_columnconfigure(1, weight=1)
        self.iterations_var = ctk.StringVar(value="150")
        self.seed_var = ctk.StringVar(value="42")
        self.time_limit_var = ctk.StringVar(value="")
        self.snapshot_var = ctk.StringVar(value="1")
        self._run_entry(run_frame, 0, "Iteraciones", self.iterations_var, "150")
        self._run_entry(run_frame, 1, "Semilla", self.seed_var, "42")
        self._run_entry(run_frame, 2, "Límite (s)", self.time_limit_var, "vacío = sin límite")
        self._run_entry(run_frame, 3, "Snapshot cada", self.snapshot_var, "1")

        # Botones ----------------------------------------------------------
        actions = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        actions.grid(row=self._next_sidebar_row(), column=0, sticky="ew", pady=(12, 4))
        actions.grid_columnconfigure((0, 1), weight=1)
        self.start_button = ctk.CTkButton(
            actions, text="Iniciar", command=self._on_start,
            fg_color="#2f9e63", hover_color="#278a56", height=36,
        )
        self.start_button.grid(row=0, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        self.pause_button = ctk.CTkButton(
            actions, text="Pausar", command=self._on_pause, height=32,
            fg_color="#b07d2b", hover_color="#946a25",
        )
        self.pause_button.grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        self.stop_button = ctk.CTkButton(
            actions, text="Detener", command=self._on_stop, height=32,
            fg_color="#b34545", hover_color="#963b3b",
        )
        self.stop_button.grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        self.restart_button = ctk.CTkButton(
            actions, text="Reiniciar", command=self._on_restart, height=32,
            fg_color="#3b6fb5", hover_color="#325f9b",
        )
        self.restart_button.grid(row=2, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        self.status_label = ctk.CTkLabel(
            self.sidebar, text="Estado: listo", font=self.font_small,
            text_color=("gray35", "gray65"), anchor="w",
        )
        self.status_label.grid(row=self._next_sidebar_row(), column=0, sticky="ew", pady=(4, 0))

    def _run_entry(
        self,
        parent: Any,
        row: int,
        label: str,
        variable: ctk.StringVar,
        placeholder: str,
    ) -> None:
        ctk.CTkLabel(parent, text=label, font=self.font_small, anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 6), pady=2
        )
        entry = ctk.CTkEntry(parent, textvariable=variable, height=26, placeholder_text=placeholder)
        entry.grid(row=row, column=1, sticky="ew", pady=2)
        self._run_entries = getattr(self, "_run_entries", [])
        self._run_entries.append(entry)

    # -- Panel principal ---------------------------------------------------

    def _build_main(self) -> None:
        header = ctk.CTkFrame(self.main, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 4))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Panel de experimentación",
            font=ctk.CTkFont(size=22, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header,
            text="GA · ACO · PSO · AIS · DE  |  continuo, binario, TSP y reparto real en Zapopan",
            font=self.font_small,
            text_color=("gray40", "gray60"),
            anchor="w",
        ).grid(row=1, column=0, sticky="w")

        self.appearance_var = ctk.StringVar(value="Oscuro")
        ctk.CTkSegmentedButton(
            header, values=["Oscuro", "Claro"], variable=self.appearance_var,
            command=self._on_theme_change, width=150,
        ).grid(row=0, column=1, rowspan=2, sticky="e", padx=(8, 0))

        # Métricas ---------------------------------------------------------
        metrics = ctk.CTkFrame(self.main, fg_color="transparent")
        metrics.grid(row=1, column=0, sticky="ew", padx=18, pady=(8, 4))
        for col in range(5):
            metrics.grid_columnconfigure(col, weight=1)
        self.metric_labels: dict[str, ctk.CTkLabel] = {}
        cards = [
            ("best", "Mejor objetivo"),
            ("iteration", "Iteración"),
            ("evaluations", "Evaluaciones"),
            ("elapsed", "Tiempo"),
            ("diversity", "Diversidad"),
        ]
        for col, (key, title) in enumerate(cards):
            card = ctk.CTkFrame(metrics, corner_radius=12, fg_color=("gray90", "gray18"))
            card.grid(row=0, column=col, sticky="ew", padx=4)
            ctk.CTkLabel(
                card, text=title, font=self.font_small,
                text_color=("gray40", "gray65"), anchor="w",
            ).pack(anchor="w", padx=12, pady=(8, 0))
            value = ctk.CTkLabel(card, text="—", font=self.font_metric, anchor="w")
            value.pack(anchor="w", padx=12, pady=(0, 8))
            self.metric_labels[key] = value

        # Progreso ---------------------------------------------------------
        progress_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        progress_frame.grid(row=2, column=0, sticky="new", padx=18, pady=(0, 2))
        progress_frame.grid_columnconfigure(0, weight=1)
        self.progress = ctk.CTkProgressBar(progress_frame, height=8)
        self.progress.grid(row=0, column=0, sticky="ew")
        self.progress.set(0)
        self.progress_label = ctk.CTkLabel(
            progress_frame, text="0 / 0", font=self.font_small,
            text_color=("gray40", "gray60"),
        )
        self.progress_label.grid(row=0, column=1, padx=(8, 0))

        # Pestañas ---------------------------------------------------------
        self.tabs = ctk.CTkTabview(self.main, corner_radius=10)
        self.tabs.grid(row=3, column=0, sticky="nsew", padx=18, pady=(4, 16))
        self.main.grid_rowconfigure(3, weight=1)

        self.tab_convergence = self.tabs.add("Convergencia")
        self.tab_solution = self.tabs.add("Solución")
        self.tab_log = self.tabs.add("Registro")
        for tab in (self.tab_convergence, self.tab_solution, self.tab_log):
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)

        # Convergencia
        self.conv_figure = Figure(figsize=(8, 5), dpi=100)
        self.conv_canvas = FigureCanvasTkAgg(self.conv_figure, master=self.tab_convergence)
        self.conv_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Solución
        self.tab_solution.grid_columnconfigure(0, weight=1)
        self.tab_solution.grid_rowconfigure(0, weight=1)
        solution_body = ctk.CTkFrame(self.tab_solution, fg_color="transparent")
        solution_body.grid(row=0, column=0, sticky="nsew")
        solution_body.grid_columnconfigure(0, weight=1)
        solution_body.grid_columnconfigure(1, weight=0, minsize=310)
        solution_body.grid_rowconfigure(0, weight=1)

        self.solution_figure = Figure(figsize=(8, 5), dpi=100)
        self.solution_canvas = FigureCanvasTkAgg(self.solution_figure, master=solution_body)
        self.solution_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.info_box = ctk.CTkTextbox(solution_body, width=300, wrap="word")
        self.info_box.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.info_box.configure(state="disabled")

        controls = ctk.CTkFrame(self.tab_solution, fg_color="transparent")
        controls.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        controls.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(controls, text="Historial", font=self.font_small).grid(
            row=0, column=0, padx=(0, 8)
        )
        self.history_slider = ctk.CTkSlider(
            controls, from_=0, to=1, command=self._on_history_slider
        )
        self.history_slider.grid(row=0, column=1, sticky="ew")
        self.history_slider.set(0)
        self.history_slider.configure(state="disabled")
        self.history_label = ctk.CTkLabel(
            controls, text="—", font=self.font_small, width=110,
            text_color=("gray40", "gray60"),
        )
        self.history_label.grid(row=0, column=2, padx=8)
        self.latest_button = ctk.CTkButton(
            controls, text="Ver última", width=90, height=26,
            fg_color="transparent", border_width=1,
            command=self._on_latest,
        )
        self.latest_button.grid(row=0, column=3)

        # Registro
        self.log_box = ctk.CTkTextbox(self.tab_log, wrap="word")
        self.log_box.grid(row=0, column=0, sticky="nsew")
        self.log_box.configure(state="disabled")

    # ------------------------------------------------------------------
    # Formularios dinámicos
    # ------------------------------------------------------------------

    def _rebuild_form(
        self,
        parent: Any,
        fields: dict[str, tuple[ParamSpec, Any, Any]],
        schema: tuple[ParamSpec, ...],
        family: str,
        values: dict[str, Any] | None = None,
    ) -> None:
        for widget in parent.winfo_children():
            widget.destroy()
        fields.clear()
        parent.grid_columnconfigure(0, weight=1)
        values = values or {}
        row = 0
        for spec in schema:
            if not spec.applies_to(family):
                continue
            default = values.get(spec.key, spec.resolve_default(family))
            ctk.CTkLabel(
                parent, text=spec.label, font=self.font_small,
                anchor="w", text_color=("gray25", "gray80"),
            ).grid(row=row, column=0, sticky="ew", pady=(4, 0))
            row += 1

            if spec.kind == "choice":
                var = ctk.StringVar(value=str(default))
                widget = ctk.CTkOptionMenu(
                    parent,
                    values=list(spec.resolve_choices(family)),
                    variable=var,
                    height=26,
                )
            elif spec.kind == "bool":
                var = ctk.BooleanVar(value=bool(default))
                widget = ctk.CTkSwitch(parent, text="", variable=var)
            else:
                var = ctk.StringVar(value=str(default))
                widget = ctk.CTkEntry(parent, textvariable=var, height=26)
            widget.grid(row=row, column=0, sticky="ew", pady=(0, 1))
            row += 1
            fields[spec.key] = (spec, var, widget)

            if spec.help:
                ctk.CTkLabel(
                    parent, text=spec.help, font=self.font_small,
                    text_color=("gray45", "gray55"), anchor="w",
                    justify="left", wraplength=310,
                ).grid(row=row, column=0, sticky="ew", pady=(0, 3))
                row += 1

    def _collect_fields(self, fields: dict[str, tuple[ParamSpec, Any, Any]]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for key, (spec, var, _widget) in fields.items():
            values[key] = spec.validate(var.get())
        return values

    # ------------------------------------------------------------------
    # Selección de problema / algoritmo
    # ------------------------------------------------------------------

    @property
    def problem_family(self) -> str:
        return get_problem(self.problem_key).family

    def _on_problem_change(self, display_name: str) -> None:
        key = self.problem_key_by_name[display_name]
        if key == self.problem_key:
            return
        self.problem_key = key
        self._select_problem(key)
        self._select_algorithm(self.algorithm_key)

    def _select_problem(self, key: str, initial: bool = False) -> None:
        self.problem_key = key
        self.problem_var.set(self.problem_name_by_key[key])
        cls = get_problem(key)
        self.family_label.configure(
            text=f"Familia: {cls.family} · {'maximizar' if cls.sense == 'max' else 'minimizar'}"
        )
        self._rebuild_form(
            self.problem_form, self.problem_fields, cls.param_schema(), cls.family
        )
        self._refresh_preview()
        if not initial:
            self._clear_run_views()
            self._draw_empty()
            self._log(f"Problema seleccionado: {cls.name}.")
        self._refresh_compat()

    def _on_algorithm_change(self, display_name: str) -> None:
        key = self.algorithm_key_by_name[display_name]
        if key == self.algorithm_key:
            return
        self.algorithm_key = key
        self._select_algorithm(key)

    def _select_algorithm(self, key: str, initial: bool = False) -> None:
        self.algorithm_key = key
        self.algorithm_var.set(self.algorithm_name_by_key[key])
        cls = get_algorithm(key)
        family = self.problem_family
        self._rebuild_form(
            self.algorithm_form,
            self.algorithm_fields,
            cls.param_schema(family),
            family,
        )
        self._refresh_compat()
        if not initial:
            self._log(f"Algoritmo seleccionado: {cls.name}.")

    def _refresh_compat(self) -> None:
        cls = get_algorithm(self.algorithm_key)
        family = self.problem_family
        if is_compatible(self.algorithm_key, family):
            self.compat_label.configure(
                text="Compatible con el problema actual.",
                text_color=("#2f7d52", "#6fcf97"),
            )
        else:
            self.compat_label.configure(
                text=(
                    f"{cls.name} solo aplica a permutaciones (TSP y rutas de Zapopan). "
                    "Selecciona GA, PSO, AIS o DE para esta familia."
                ),
                text_color=("#b06020", "#f2b06b"),
            )
        self._update_controls()

    def _restore_defaults(self) -> None:
        if self._ui_running:
            return
        self._select_problem(self.problem_key, initial=True)
        self._select_algorithm(self.algorithm_key, initial=True)
        self._log("Parámetros restaurados a sus valores por defecto.")

    def _refresh_preview(self) -> None:
        """Construye una instancia del problema con los valores actuales."""
        try:
            values = self._collect_fields(self.problem_fields)
            self.current_problem = get_problem(self.problem_key)(**values)
        except Exception:
            self.current_problem = None

    # ------------------------------------------------------------------
    # Ejecución
    # ------------------------------------------------------------------

    def _read_run_config(self) -> RunConfig:
        try:
            iterations = int(self.iterations_var.get().strip())
        except ValueError as exc:
            raise ValueError("Las iteraciones deben ser un entero.") from exc
        try:
            seed = int(self.seed_var.get().strip())
        except ValueError as exc:
            raise ValueError("La semilla debe ser un entero.") from exc
        raw_time = self.time_limit_var.get().strip()
        try:
            time_limit = float(raw_time) if raw_time else None
        except ValueError as exc:
            raise ValueError("El límite de tiempo debe ser numérico o estar vacío.") from exc
        try:
            snapshot_every = int(self.snapshot_var.get().strip())
        except ValueError as exc:
            raise ValueError("La frecuencia de snapshot debe ser un entero.") from exc
        config = RunConfig(
            max_iterations=iterations,
            seed=seed,
            time_limit=time_limit,
            snapshot_every=snapshot_every,
        )
        config.validate()
        return config

    def _on_start(self) -> None:
        self._start_run(restart=False)

    def _start_run(self, restart: bool) -> None:
        if self.engine is not None and self.engine.is_running:
            self.engine.stop()
        try:
            problem_params = self._collect_fields(self.problem_fields)
            algorithm_params = self._collect_fields(self.algorithm_fields)
            config = self._read_run_config()
            problem_cls = get_problem(self.problem_key)
            algorithm_cls = get_algorithm(self.algorithm_key)
            if not is_compatible(self.algorithm_key, problem_cls.family):
                raise ValueError(
                    "El algoritmo seleccionado no admite este problema. "
                    "Elige GA, PSO, AIS o DE."
                )
            # Validación temprana de los parámetros del problema.
            problem = problem_cls(**problem_params)
            self.current_problem = problem
            engine = ExperimentEngine(
                problem_cls,
                problem_params,
                algorithm_cls,
                algorithm_params,
                config,
            )
        except Exception as exc:
            self._show_error("No se pudo iniciar la corrida", str(exc))
            return

        self._max_iterations = config.max_iterations
        self.engine = engine
        self.snapshots.clear()
        self.view_index = 0
        self.follow_latest = True
        self._clear_run_views()
        self._run_token += 1
        self._ui_running = True
        self._dirty = False
        self._update_controls()
        self.status_label.configure(text="Estado: ejecutando…")
        self.pause_button.configure(text="Pausar")
        self._log(
            ("Reinicio" if restart else "Inicio")
            + f": {algorithm_cls.name} sobre {problem.name} · "
            f"{config.max_iterations} iteraciones · semilla {config.seed}."
        )
        engine.start()
        token = self._run_token
        self.after(50, lambda: self._poll_events(token))

    def _on_pause(self) -> None:
        if self.engine is None or not self.engine.is_running:
            return
        paused = self.engine.toggle_pause()
        self.pause_button.configure(text="Reanudar" if paused else "Pausar")
        self.status_label.configure(text="Estado: pausado" if paused else "Estado: ejecutando…")
        self._log("Corrida pausada." if paused else "Corrida reanudada.")

    def _on_stop(self) -> None:
        if self.engine is None or not self.engine.is_running:
            return
        self.status_label.configure(text="Estado: deteniendo…")
        self._log("Solicitud de detención enviada…")
        self.engine.stop()

    def _on_restart(self) -> None:
        self._log("Reiniciando la corrida con la configuración actual…")
        self._start_run(restart=True)

    def _poll_events(self, token: int) -> None:
        if token != self._run_token or self.engine is None:
            return
        finished = False
        for event in self.engine.drain():
            self._handle_event(event)
            if event.kind in {"finished", "stopped", "error"}:
                finished = True

        if self.snapshots:
            self._sync_history_controls()
            self._update_metrics(self.snapshots[-1])

        if self._dirty and (time.perf_counter() - self._last_draw >= 0.1):
            self._draw_all()
            self._dirty = False

        if self.engine.is_running or not self.engine.events.empty():
            self.after(60, lambda: self._poll_events(token))
        elif finished or self._ui_running:
            self._ui_running = False
            self._draw_all(force=True)
            self._update_controls()
            if self.status_label.cget("text") in {"Estado: ejecutando…", "Estado: deteniendo…"}:
                self.status_label.configure(text="Estado: listo")

    def _handle_event(self, event: EngineEvent) -> None:
        if event.kind == "started":
            data = event.payload
            self._log(
                f"Motor iniciado: {data['algorithm']} / {data['problem']} "
                f"(semilla {data['seed']})."
            )
        elif event.kind == "snapshot":
            snapshot: Snapshot = event.payload
            self.snapshots.append(snapshot)
            if self.follow_latest:
                self.view_index = len(self.snapshots) - 1
            self._dirty = True
        elif event.kind == "finished":
            summary = event.payload or {}
            self._log(
                "Corrida finalizada · mejor objetivo: "
                f"{self._format_number(summary.get('best_fitness'))} · "
                f"evaluaciones: {summary.get('evaluations', '—')} · "
                f"tiempo: {summary.get('elapsed', '—')} s."
            )
            self.status_label.configure(text="Estado: finalizado")
        elif event.kind == "stopped":
            self._log("Corrida detenida por el usuario; se conserva el histórico.")
            self.status_label.configure(text="Estado: detenido")
        elif event.kind == "error":
            self._log("ERROR durante la corrida:\n" + str(event.payload))
            self.status_label.configure(text="Estado: error")
            self._show_error(
                "La corrida terminó con un error",
                "Revisa el panel Registro para ver el traceback completo.",
            )

    # ------------------------------------------------------------------
    # Estado de los controles
    # ------------------------------------------------------------------

    def _update_controls(self) -> None:
        compatible = is_compatible(self.algorithm_key, self.problem_family)
        self.start_button.configure(
            state="normal" if (compatible and not self._ui_running) else "disabled"
        )
        self.pause_button.configure(state="normal" if self._ui_running else "disabled")
        self.stop_button.configure(state="normal" if self._ui_running else "disabled")
        self.restart_button.configure(state="normal")
        self.problem_menu.configure(state="disabled" if self._ui_running else "normal")
        self.algorithm_menu.configure(state="disabled" if self._ui_running else "normal")
        self.restore_button.configure(state="disabled" if self._ui_running else "normal")
        for entry in getattr(self, "_run_entries", []):
            entry.configure(state="disabled" if self._ui_running else "normal")
        state = "disabled" if self._ui_running else "normal"
        for fields in (self.problem_fields, self.algorithm_fields):
            for _spec, _var, widget in fields.values():
                try:
                    widget.configure(state=state)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Dibujo
    # ------------------------------------------------------------------

    def _theme(self) -> dict[str, str]:
        dark = ctk.get_appearance_mode() == "Dark"
        return {
            "bg": "#1b1b1b" if dark else "#ffffff",
            "fg": "#e6e6e6" if dark else "#202020",
            "grid": "#3a3a3a" if dark else "#e2e2e2",
            "muted": "#9a9a9a" if dark else "#666666",
            "accent": "#2cc985",
            "secondary": "#f2a93b",
            "third": "#5aa9e6",
            "bad": "#e06c75",
        }

    def _style_axes(self, fig: Figure, ax: Any) -> None:
        theme = self._theme()
        fig.patch.set_facecolor(theme["bg"])
        ax.set_facecolor(theme["bg"])
        ax.tick_params(colors=theme["fg"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(theme["grid"])
        ax.xaxis.label.set_color(theme["fg"])
        ax.yaxis.label.set_color(theme["fg"])
        ax.title.set_color(theme["fg"])
        ax.grid(True, color=theme["grid"], alpha=0.45, linewidth=0.6)

    def _draw_empty(self) -> None:
        self._draw_convergence()
        self._draw_solution()
        self._update_info(None)
        self._update_metrics(None)

    def _clear_run_views(self) -> None:
        self.snapshots.clear()
        self.view_index = 0
        self.follow_latest = True
        self._updating_slider = True
        try:
            self.history_slider.configure(to=1, state="disabled")
            self.history_slider.set(0)
        finally:
            self._updating_slider = False
        self.history_label.configure(text="—")
        self._last_draw = 0.0
        self._dirty = False

    def _draw_all(self, force: bool = False) -> None:
        if not force and time.perf_counter() - self._last_draw < 0.08:
            return
        self._last_draw = time.perf_counter()
        self._draw_convergence()
        self._draw_solution()

    # -- Convergencia ------------------------------------------------------

    def _draw_convergence(self) -> None:
        theme = self._theme()
        fig = self.conv_figure
        fig.clear()
        fig.patch.set_facecolor(theme["bg"])

        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        for ax in (ax1, ax2):
            self._style_axes(fig, ax)

        if not self.snapshots:
            ax1.text(
                0.5, 0.5, "Sin datos todavía\npulsa Iniciar para comenzar",
                ha="center", va="center", color=theme["muted"], fontsize=11,
                transform=ax1.transAxes,
            )
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])
            fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.09, hspace=0.35)
            self.conv_canvas.draw_idle()
            return

        iterations = np.array([s.iteration for s in self.snapshots])
        best = np.array([s.best_fitness for s in self.snapshots])
        mean = np.array([s.mean_fitness for s in self.snapshots])
        diversity = np.array([s.diversity for s in self.snapshots])

        ax1.plot(iterations, best, color=theme["accent"], linewidth=2.0, label="Mejor")
        ax1.plot(iterations, mean, color=theme["secondary"], linewidth=1.2, alpha=0.9, label="Media")
        ax1.set_ylabel("Objetivo", fontsize=9)
        ax1.set_title("Convergencia", fontsize=11, loc="left")
        ax1.legend(loc="best", fontsize=8, facecolor=theme["bg"], edgecolor=theme["grid"], labelcolor=theme["fg"])
        ax1.set_xlim(left=0, right=max(1, iterations[-1]))
        if self.current_problem is not None:
            ax1.set_ylabel(self.current_problem.objective_label, fontsize=8)

        ax2.plot(iterations, diversity, color=theme["third"], linewidth=1.6)
        ax2.fill_between(iterations, 0, diversity, color=theme["third"], alpha=0.18)
        ax2.set_ylim(0, 1.02)
        ax2.set_xlabel("Iteración", fontsize=9)
        ax2.set_ylabel("Diversidad", fontsize=9)
        ax2.set_xlim(left=0, right=max(1, iterations[-1]))
        ax2.set_title("Diversidad de la población", fontsize=11, loc="left")

        fig.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.10, hspace=0.42)
        self.conv_canvas.draw_idle()

    # -- Solución ----------------------------------------------------------

    def _visible_snapshot(self) -> Snapshot | None:
        if not self.snapshots:
            return None
        index = min(max(0, self.view_index), len(self.snapshots) - 1)
        return self.snapshots[index]

    def _draw_solution(self) -> None:
        fig = self.solution_figure
        fig.clear()
        fig.patch.set_facecolor(self._theme()["bg"])
        snapshot = self._visible_snapshot()
        if self.current_problem is None:
            self._solution_message("Configura un problema válido para visualizar su solución.")
            self._update_info(None)
            return
        if snapshot is None:
            self._solution_message(
                "Aún no hay solución.\nInicia una corrida para ver la evolución."
            )
            self._update_info(None)
            return

        problem = self.current_problem
        try:
            if problem.family == "continuous":
                self._plot_continuous(fig, snapshot)
            elif problem.family == "binary":
                self._plot_binary(fig, snapshot)
            elif problem.key == "tsp":
                self._plot_tsp(fig, snapshot)
            elif problem.key == "delivery_zapopan":
                self._plot_delivery(fig, snapshot)
            else:
                self._solution_message("Visualización no disponible para este problema.")
        except Exception as exc:  # una visualización nunca debe tumbar la app
            self._solution_message(f"No se pudo dibujar la solución:\n{exc}")
        self._update_info(snapshot)
        self.solution_canvas.draw_idle()

    def _solution_message(self, message: str) -> None:
        theme = self._theme()
        fig = self.solution_figure
        fig.clear()
        fig.patch.set_facecolor(theme["bg"])
        ax = fig.add_subplot(111)
        self._style_axes(fig, ax)
        ax.text(
            0.5, 0.5, message, ha="center", va="center", fontsize=11,
            color=theme["muted"], transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)

    def _plot_continuous(self, fig: Figure, snapshot: Snapshot) -> None:
        problem = self.current_problem
        assert problem is not None
        solution = np.asarray(snapshot.best_solution, dtype=float)
        population = snapshot.extra.get("population")

        if problem.dimension == 2:
            ax = fig.add_subplot(111)
            self._style_axes(fig, ax)
            key = (problem.key, problem.dim, problem.lb, problem.ub)
            if key not in self._contour_cache:
                xs = np.linspace(problem.lb, problem.ub, 90)
                ys = np.linspace(problem.lb, problem.ub, 90)
                xx, yy = np.meshgrid(xs, ys)
                zz = np.array(
                    [
                        [problem.evaluate(np.array([x, y])) for x in xs]
                        for y in ys
                    ]
                )
                self._contour_cache[key] = (xx, yy, zz)
            xx, yy, zz = self._contour_cache[key]
            contour = ax.contourf(xx, yy, zz, levels=28, cmap="viridis", alpha=0.9)
            if population is not None and len(population):
                pop = np.asarray(population)
                ax.scatter(
                    pop[:, 0], pop[:, 1], s=14, c="#ffffff", edgecolors="#1b1b1b",
                    linewidths=0.4, alpha=0.75, label="Población",
                )
            ax.scatter(
                [solution[0]], [solution[1]], marker="*", s=220,
                c=self._theme()["accent"], edgecolors="black", linewidths=0.6,
                zorder=5, label="Mejor",
            )
            ax.set_xlabel("x₁")
            ax.set_ylabel("x₂")
            ax.set_title(
                f"{problem.name} · iteración {snapshot.iteration} · "
                f"objetivo {snapshot.best_fitness:.4f}",
                fontsize=10, loc="left",
            )
            ax.legend(
                loc="upper right", fontsize=8, facecolor=self._theme()["bg"],
                edgecolor=self._theme()["grid"], labelcolor=self._theme()["fg"],
            )
            fig.colorbar(contour, ax=ax, fraction=0.045, pad=0.02)
            fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.09)
            return

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        for ax in (ax1, ax2):
            self._style_axes(fig, ax)
        shown = min(problem.dimension, 40)
        ax1.bar(
            np.arange(shown), solution[:shown], color=self._theme()["accent"], alpha=0.9
        )
        ax1.axhline(0, color=self._theme()["muted"], linewidth=0.8)
        ax1.set_title(
            f"Mejor vector (primeras {shown} de {problem.dimension} dims)",
            fontsize=10, loc="left",
        )
        ax1.set_xlabel("Dimensión")
        ax1.set_ylabel("Valor")
        ax1.set_xlim(-0.8, shown - 0.2)

        if population is not None and len(population) > 1:
            pop = np.asarray(population)
            ax2.scatter(
                pop[:, 0], pop[:, 1], s=16, alpha=0.6,
                c=self._theme()["third"], edgecolors="none",
            )
            ax2.set_title("Población (dimensiones 1 y 2)", fontsize=10, loc="left")
            ax2.set_xlabel("x₁")
            ax2.set_ylabel("x₂")
        else:
            ax2.text(
                0.5, 0.5, "Sin muestra de población", ha="center", va="center",
                transform=ax2.transAxes, color=self._theme()["muted"],
            )
        fig.suptitle(
            f"{problem.name} · iteración {snapshot.iteration} · "
            f"objetivo {snapshot.best_fitness:.4f}",
            fontsize=11, color=self._theme()["fg"], x=0.02, ha="left",
        )
        fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.10, wspace=0.25)

    def _plot_binary(self, fig: Figure, snapshot: Snapshot) -> None:
        problem = self.current_problem
        assert problem is not None
        solution = np.asarray(snapshot.best_solution, dtype=int)
        population = snapshot.extra.get("population")

        if problem.key == "onemax":
            ax = fig.add_subplot(111)
            self._style_axes(fig, ax)
            if population is not None and len(population):
                sample = np.asarray(population)
                ax.imshow(
                    sample, aspect="auto", interpolation="nearest",
                    cmap=ListedColormap(["#2b2b2b", self._theme()["accent"]]),
                )
                ax.set_ylabel("Individuos (muestra)")
            ax.set_xlabel("Bit")
            ax.set_title(
                f"OneMax · iteración {snapshot.iteration} · "
                f"{int(solution.sum())} / {solution.size} bits en 1",
                fontsize=10, loc="left",
            )
            fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.10)
            return

        # Mochila: valores de los artículos seleccionados y uso de capacidad.
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        for ax in (ax1, ax2):
            self._style_axes(fig, ax)
        selected = np.flatnonzero(solution)
        if selected.size:
            colors = [
                self._theme()["accent"] if solution[i] else self._theme()["muted"]
                for i in selected
            ]
            ax1.bar(
                np.arange(selected.size), problem.values[selected],
                color=colors, alpha=0.9,
            )
            for pos, i in enumerate(selected):
                ax1.text(
                    pos, problem.values[i] + 0.5, f"{problem.weights[i]:.0f}",
                    ha="center", va="bottom", fontsize=7, color=self._theme()["fg"],
                )
            ax1.set_xticks(np.arange(selected.size))
            ax1.set_xticklabels([f"A{int(i) + 1:02d}" for i in selected], rotation=60, fontsize=7)
        else:
            ax1.text(
                0.5, 0.5, "Ningún artículo seleccionado", ha="center", va="center",
                transform=ax1.transAxes, color=self._theme()["muted"],
            )
        ax1.set_ylabel("Valor")
        ax1.set_title("Artículos seleccionados (número = peso)", fontsize=10, loc="left")

        weight = float(np.dot(solution, problem.weights))
        capacity = float(problem.capacity)
        ax2.barh([0], [min(weight, capacity)], color=self._theme()["accent"])
        if weight > capacity:
            ax2.barh(
                [0], [weight - capacity], left=[capacity],
                color=self._theme()["bad"], alpha=0.9,
            )
        ax2.axvline(capacity, color=self._theme()["secondary"], linestyle="--", linewidth=1.5)
        ax2.set_yticks([0])
        ax2.set_yticklabels(["Carga"])
        ax2.set_xlabel("Peso")
        ax2.set_title(
            f"Capacidad: {weight:.1f} / {capacity:.1f}", fontsize=10, loc="left"
        )
        ax2.set_xlim(0, max(capacity, weight) * 1.08)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.18, wspace=0.25)

    def _plot_tsp(self, fig: Figure, snapshot: Snapshot) -> None:
        problem = self.current_problem
        assert problem is not None
        tour = np.asarray(snapshot.best_solution, dtype=int)
        coords = problem.coordinates
        closed = np.append(tour, tour[0])
        theme = self._theme()

        ax = fig.add_subplot(111)
        self._style_axes(fig, ax)
        ax.plot(
            coords[closed, 0], coords[closed, 1],
            color=theme["accent"], linewidth=1.8, alpha=0.95, zorder=2,
        )
        ax.scatter(
            coords[:, 0], coords[:, 1], s=34, c=theme["third"],
            edgecolors=theme["bg"], linewidths=0.8, zorder=3,
        )
        start = int(tour[0])
        ax.scatter(
            [coords[start, 0]], [coords[start, 1]], s=150, marker="*",
            c=theme["secondary"], edgecolors=theme["bg"], linewidths=0.8, zorder=4,
        )
        if problem.n <= 30:
            for i, (x, y) in enumerate(coords):
                ax.annotate(
                    str(i + 1), (x, y), textcoords="offset points", xytext=(4, 4),
                    fontsize=7, color=theme["fg"],
                )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"TSP · {problem.n} ciudades · distancia {snapshot.best_fitness:.2f} · "
            f"iteración {snapshot.iteration}",
            fontsize=10, loc="left",
        )
        fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.08)

    def _plot_delivery(self, fig: Figure, snapshot: Snapshot) -> None:
        problem = self.current_problem
        assert problem is not None
        solution = np.asarray(snapshot.best_solution, dtype=int)
        routes = problem.decode_routes(solution)
        lines = problem.route_points(solution)
        depot, clients = problem.plot_coordinates()
        theme = self._theme()
        cmap = colormaps.get_cmap("tab10")

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        for ax in (ax1, ax2):
            self._style_axes(fig, ax)

        ax1.scatter(
            clients[:, 0], clients[:, 1], s=42, c=theme["muted"],
            edgecolors=theme["bg"], linewidths=0.7, zorder=3,
        )
        for i, (x, y) in enumerate(clients):
            ax1.annotate(
                problem.names[i], (x, y), textcoords="offset points", xytext=(4, 4),
                fontsize=7, color=theme["fg"],
            )
        for idx, line in enumerate(lines):
            arr = np.asarray(line)
            color = cmap(idx % 10)
            ax1.plot(arr[:, 0], arr[:, 1], color=color, linewidth=2.0, alpha=0.95)
            ax1.scatter(
                arr[1:-1, 0], arr[1:-1, 1], s=30, color=color,
                edgecolors=theme["bg"], linewidths=0.5, zorder=4,
            )
        ax1.scatter(
            depot[:, 0], depot[:, 1], marker="*", s=260,
            c=theme["secondary"], edgecolors=theme["bg"], linewidths=0.7,
            zorder=5, label="Base CEDIS",
        )
        ax1.set_aspect("equal", adjustable="datalim")
        ax1.set_xlabel("Longitud")
        ax1.set_ylabel("Latitud")
        ax1.set_title(
            f"Rutas Zapopan · {len(routes)} rutas · "
            f"objetivo {snapshot.best_fitness:.2f}",
            fontsize=10, loc="left",
        )
        ax1.legend(
            loc="best", fontsize=8, facecolor=theme["bg"],
            edgecolor=theme["grid"], labelcolor=theme["fg"],
        )

        loads = [route["carga"] for route in routes]
        positions = np.arange(len(routes))
        ax2.bar(
            positions, loads, color=[cmap(i % 10) for i in positions], alpha=0.9
        )
        ax2.axhline(
            problem.q_max, color=theme["bad"], linestyle="--", linewidth=1.4,
            label=f"Q_max = {problem.q_max:.0f} kg",
        )
        ax2.set_xticks(positions)
        ax2.set_xticklabels([f"R{i + 1}" for i in positions], fontsize=8)
        ax2.set_ylabel("Carga (kg)")
        ax2.set_ylim(0, problem.q_max * 1.15)
        ax2.set_title("Carga por ruta", fontsize=10, loc="left")
        ax2.legend(
            loc="upper right", fontsize=8, facecolor=theme["bg"],
            edgecolor=theme["grid"], labelcolor=theme["fg"],
        )
        fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.12, wspace=0.25)

    # -- Panel de información ---------------------------------------------

    def _update_info(self, snapshot: Snapshot | None) -> None:
        self.info_box.configure(state="normal")
        self.info_box.delete("1.0", "end")
        if snapshot is None or self.current_problem is None:
            self.info_box.insert("1.0", "Sin solución para mostrar.")
            self.info_box.configure(state="disabled")
            return
        problem = self.current_problem
        lines = [
            f"Iteración: {snapshot.iteration}",
            f"Objetivo: {self._format_number(snapshot.best_fitness)}",
            f"Diversidad: {snapshot.diversity:.3f}",
            f"Evaluaciones: {snapshot.evaluations}",
            "",
        ]
        if problem.family == "continuous":
            solution = np.asarray(snapshot.best_solution, dtype=float)
            lines.extend(
                [
                    f"Dimensión: {problem.dimension}",
                    f"Norma euclidiana: {float(np.linalg.norm(solution)):.4f}",
                    "Mejor vector:",
                    ", ".join(f"{v:.4f}" for v in solution[:12])
                    + ("…" if solution.size > 12 else ""),
                ]
            )
        else:
            description = problem.describe(snapshot.best_solution)
            for key, value in description.items():
                if "\n" in str(value):
                    lines.append(f"{key}:")
                    lines.append(str(value))
                else:
                    lines.append(f"{key}: {value}")
        self.info_box.insert("1.0", "\n".join(lines))
        self.info_box.configure(state="disabled")

    def _update_metrics(self, snapshot: Snapshot | None) -> None:
        if snapshot is None:
            for label in self.metric_labels.values():
                label.configure(text="—")
            self.progress.set(0)
            self.progress_label.configure(text=f"0 / {self._max_iterations}")
            return
        self.metric_labels["best"].configure(
            text=self._format_number(snapshot.best_fitness)
        )
        self.metric_labels["iteration"].configure(
            text=f"{snapshot.iteration} / {self._max_iterations}"
        )
        self.metric_labels["evaluations"].configure(text=f"{snapshot.evaluations:,}")
        self.metric_labels["elapsed"].configure(text=f"{snapshot.elapsed:.1f} s")
        self.metric_labels["diversity"].configure(text=f"{snapshot.diversity:.3f}")
        self.progress.set(min(1.0, snapshot.iteration / max(1, self._max_iterations)))
        self.progress_label.configure(
            text=f"{snapshot.iteration} / {self._max_iterations}"
        )

    def _sync_history_controls(self) -> None:
        """Actualiza el slider una sola vez por sondeo, no por snapshot.

        CustomTkinter redibuja el canvas completo del slider en cada ``set``;
        hacerlo por cada snapshot congela la interfaz en corridas largas.
        """
        if not self.snapshots:
            return
        last = len(self.snapshots) - 1
        self._updating_slider = True
        try:
            self.history_slider.configure(to=max(1, last), state="normal")
            if self.follow_latest:
                self.view_index = last
                self.history_slider.set(last)
        finally:
            self._updating_slider = False
        snapshot = self.snapshots[min(self.view_index, last)]
        self.history_label.configure(
            text=f"iter {snapshot.iteration} / {self._max_iterations}"
        )

    # ------------------------------------------------------------------
    # Interacción auxiliar
    # ------------------------------------------------------------------

    def _on_history_slider(self, value: float) -> None:
        if self._updating_slider or not self.snapshots:
            return
        index = int(round(float(value)))
        index = min(max(0, index), len(self.snapshots) - 1)
        self.view_index = index
        self.follow_latest = index == len(self.snapshots) - 1
        snapshot = self.snapshots[index]
        self.history_label.configure(
            text=f"iter {snapshot.iteration} / {self._max_iterations}"
        )
        self._draw_solution()

    def _on_latest(self) -> None:
        if not self.snapshots:
            return
        self.follow_latest = True
        self.view_index = len(self.snapshots) - 1
        self._updating_slider = True
        try:
            self.history_slider.set(self.view_index)
        finally:
            self._updating_slider = False
        snapshot = self.snapshots[-1]
        self.history_label.configure(
            text=f"iter {snapshot.iteration} / {self._max_iterations}"
        )
        self._draw_solution()
        self._update_metrics(snapshot)

    def _on_theme_change(self, value: str) -> None:
        ctk.set_appearance_mode("dark" if value == "Oscuro" else "light")
        self._draw_convergence()
        self._draw_solution()

    # ------------------------------------------------------------------
    # Registro y errores
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{stamp}] {message}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _show_error(self, title: str, message: str) -> None:
        self._log(f"{title}: {message}")
        try:
            from tkinter import messagebox

            messagebox.showerror(title, message, parent=self)
        except Exception:
            pass

    @staticmethod
    def _format_number(value: Any) -> str:
        if value is None:
            return "—"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(number):
            return "∞" if number > 0 else "−∞"
        magnitude = abs(number)
        if magnitude != 0 and (magnitude >= 1e4 or magnitude < 1e-3):
            return f"{number:.3e}"
        return f"{number:.4f}"

    # ------------------------------------------------------------------
    # Cierre
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        try:
            if self.engine is not None:
                self.engine.stop(timeout=1.0)
        finally:
            self.destroy()


def main() -> None:
    app = MetaheuristicApp()
    app.mainloop()


if __name__ == "__main__":
    main()
